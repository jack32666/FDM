# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Main training loop."""

import os
import time
import copy
import json
import pickle
import psutil
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc
from training import networks
from torch.distributions import Beta, Normal
from collections import deque

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import beta

def feature_selector(kl_diff_lasso, kl_diff_sum_lasso, logger, global_steps, dist):

    X = kl_diff_lasso.cpu().numpy()  # Feature matrix
    y = kl_diff_sum_lasso.cpu().numpy()  # Target vector

    # Select the best features using SelectKBest
    k_best = SelectKBest(score_func=f_regression, k=5)
    k_best.fit(X, y)
    selected_features_indices = k_best.get_support(indices=True)
    
    # Create a linear regression model with the selected features
    model = LinearRegression()
    model.fit(X[:, selected_features_indices], y)
    y_pred = model.predict(X[:, selected_features_indices])
    r2 = r2_score(y, y_pred)
    
    if dist.get_rank() == 0:
        logger.log({"feature_selection/feature_selection_acheived_r2": r2}, step=global_steps)
        logger.log({"feature_selection/demanded_n_features": 3}, step=global_steps)
        logger.log({"feature_selection/selected_n_features": selected_features_indices.shape[0]}, step=global_steps)
        
    return selected_features_indices


#----------------------------------------------------------------------------

def training_loop(
    run_dir             = '.',      # Output directory.
    dataset_kwargs      = {},       # Options for training set.
    data_loader_kwargs  = {},       # Options for torch.utils.data.DataLoader.
    network_kwargs      = {},       # Options for model and preconditioning.
    loss_kwargs         = {},       # Options for loss function.
    optimizer_kwargs    = {},       # Options for optimizer.
    augment_kwargs      = None,     # Options for augmentation pipeline, None = disable.
    actor_network_kwargs = {},  # Add this parameter
    actor_optimizer_kwargs = {},  # Add this parameter
    seed                = 0,        # Global random seed.
    batch_size          = 512,      # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU, None = no limit.
    total_kimg          = 200000,   # Training duration, measured in thousands of training images.
    ema_halflife_kimg   = 500,      # Half-life of the exponential moving average (EMA) of model weights.
    ema_rampup_ratio    = 0.05,     # EMA ramp-up coefficient, None = no rampup.
    lr_rampup_kimg      = 10000,    # Learning rate ramp-up duration.
    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    kimg_per_tick       = 50,       # Interval of progress prints.
    snapshot_ticks      = 50,       # How often to save network snapshots, None = disable.
    state_dump_ticks    = 500,      # How often to dump training state, None = disable.
    resume_pkl          = None,     # Start from the given network snapshot, None = random initialization.
    resume_state_dump   = None,     # Start from the given training state, None = reset training state.
    resume_kimg         = 0,        # Start from the given training progress.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'),
    logger              = None,
):
    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    non_zero_coef_timesteps = [1e-8, 0.00001, 0.001, 0.1, 0.3, 0.5, 0.8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    kl_diff_lasso_queue = deque(maxlen=20)
    kl_diff_sum_lasso_queue = deque(maxlen=20)
    update_policy_interval = 1
    global_steps = 0

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # Load dataset.
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # subclass of training.dataset.Dataset
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))

    # Construct network.
    dist.print0('Constructing network...')
    interface_kwargs = dict(img_resolution=dataset_obj.resolution, img_channels=dataset_obj.num_channels, label_dim=dataset_obj.label_dim)
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs) # subclass of torch.nn.Module
    net.train().requires_grad_(True).to(device)
    if dist.get_rank() == 0:
        with torch.no_grad():
            images = torch.zeros([batch_gpu, net.img_channels, net.img_resolution, net.img_resolution], device=device)
            sigma = torch.ones([batch_gpu], device=device)
            labels = torch.zeros([batch_gpu, net.label_dim], device=device)
            misc.print_module_summary(net, [images, sigma, labels], max_nesting=2)

    # Setup optimizer.
    dist.print0('Setting up optimizer...')
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs) # training.loss.(VP|VE|EDM)Loss
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs) # subclass of torch.optim.Optimizer
    augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs) if augment_kwargs is not None else None # training.augment.AugmentPipe
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device], broadcast_buffers=False)
    ema = copy.deepcopy(net).eval().requires_grad_(False)

    # Construct ActorNetwork
    dist.print0('Constructing ActorNetwork...')
    actor_net = networks.ActorNetwork().to(device)
    actor_net.train().requires_grad_(True)
    if dist.get_rank() == 0:
        misc.print_module_summary(actor_net, [images], max_nesting=2)

    # Setup ActorNetwork optimizer
    dist.print0('Setting up ActorNetwork optimizer...')
    actor_optimizer = dnnlib.util.construct_class_by_name(params=actor_net.parameters(), **actor_optimizer_kwargs)
    actor_ddp = torch.nn.parallel.DistributedDataParallel(actor_net, device_ids=[device], broadcast_buffers=False)
    actor_ema = copy.deepcopy(actor_net).eval().requires_grad_(False)


    # Resume training from previous snapshot.
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier() # rank 0 goes first
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier() # other ranks follow
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=net, require_all=False)
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=ema, require_all=False)
        del data # conserve memory
    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location=torch.device('cpu'))
        misc.copy_params_and_buffers(src_module=data['net'], dst_module=net, require_all=True)
        optimizer.load_state_dict(data['optimizer_state'])
        del data # conserve memory

    # Train.
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None
    while True:

        # Accumulate gradients.
        optimizer.zero_grad(set_to_none=True)
        actor_optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                images, labels = next(dataset_iterator)
                images = images.to(device).to(torch.float32) / 127.5 - 1
                labels = labels.to(device)
                
                B = images.shape[0]
                T = 21
                #################### BEFORE DIFFUSION UPDATE ####################
                if cur_nimg % 4992 == 0:
                    # KL_sum before for lasso
                    range_T = torch.tensor([1e-8, 0.00001, 0.001, 0.1, 0.3, 0.5, 0.8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], device=device).view(-1, 1, 1, 1)
                    random_index = torch.randint(0, images.shape[0], (1,))
                    x_sampled = images[random_index]
                    x_sampled = x_sampled.repeat(range_T.shape[0], 1, 1, 1)
                    with torch.no_grad():
                        kl_before_for_lasso = loss_fn(net=ddp, images=x_sampled, sigma=range_T, labels=labels, augment_pipe=augment_pipe).sum(dim=[1, 2, 3])
                    
                    # KL_sum before for reward 
                    kl_divergence_tensor_list = []
                    for j in non_zero_coef_timesteps:
                        t_full = torch.full((images.shape[0],), j, device=device).view(-1, 1, 1, 1)
                        with torch.no_grad():
                            kl_divergence_values = loss_fn(net=ddp, images=images, sigma=t_full, labels=labels, augment_pipe=augment_pipe)
                        kl_divergence_tensor_list.append(kl_divergence_values)
                    kl_divergence_tensor_before = torch.stack(kl_divergence_tensor_list).transpose(0, 1).sum(dim=[1, 2, 3, 4])
                    
                    
                    #print("kl_divergence_tensor_before", kl_divergence_tensor_before, "non_zero_coef_timesteps", non_zero_coef_timesteps)
                
                #################### SAMPLE TIMESTEP ####################
                # alpha_value, beta_value = actor_ddp(images)
                # alpha_value = alpha_value.squeeze()
                # beta_value = beta_value.squeeze()

                # beta_dist = Beta(alpha_value, beta_value)
                # dist_sampled = beta_dist.sample()
            
                # timesteps = dist_sampled * 20  #Scale to [0, 20]
                # timesteps = timesteps.view(images.shape[0], 1, 1, 1)
                # timestep = torch.clamp(timesteps, min=0.1)
                
                # log_probs = beta_dist.log_prob(dist_sampled)
                # entropy = beta_dist.entropy()
                
                normal_mean, normal_std = actor_ddp(images)
                normal_mean = normal_mean.squeeze()
                normal_std = normal_std.squeeze()
                
                normal_dist = Normal(normal_mean, normal_std)
                dist_sampled = normal_dist.sample()
                
                log_probs = normal_dist.log_prob(dist_sampled)
                entropy = normal_dist.entropy()
                
                rnd_normal = normal_mean.view(-1,1,1,1) + normal_std.view(-1,1,1,1) * torch.randn([images.shape[0], 1, 1, 1], device=images.device)
                timestep = (rnd_normal *1.2 - 1.2).exp() # log normal of [0,20]
                timestep = torch.clamp((rnd_normal * 1.2 - 1.2).exp(), max=20)
                
                #################### DIFFUSION UPDATE ####################
                loss = loss_fn(net=ddp, images=images, sigma=timestep.detach(), labels=labels, augment_pipe=augment_pipe)
                training_stats.report('Loss/loss', loss)
                loss.sum().mul(loss_scaling / batch_gpu_total).backward()
                for g in optimizer.param_groups:
                    g['lr'] = optimizer_kwargs['lr'] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)
                for param in net.parameters():
                    if param.grad is not None:
                        torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
                optimizer.step()
                
                #################### AFTER DIFFUSION UPDATE ####################
                if cur_nimg % 4992 == 0:
                    # KL_sum after for lasso
                    with torch.no_grad():
                        kl_after_for_lasso = loss_fn(net=ddp, images=x_sampled, sigma=range_T, labels=labels, augment_pipe=augment_pipe).sum(dim=[1, 2, 3])
                    kl_diff_lasso = kl_before_for_lasso - kl_after_for_lasso
                    kl_diff_sum_lasso = kl_diff_lasso.sum()
                    
                    # Add to queue
                    kl_diff_lasso_queue.append(kl_diff_lasso)
                    kl_diff_sum_lasso_queue.append(kl_diff_sum_lasso)
                    
                    # KL_sum after for reward
                    kl_divergence_tensor_list = []
                    for j in non_zero_coef_timesteps:
                        t_full = torch.full((images.shape[0],), j, device=device).view(-1, 1, 1, 1)
                        with torch.no_grad():
                            kl_divergence_values = loss_fn(net=ddp, images=images, sigma=t_full, labels=labels, augment_pipe=augment_pipe)
                        kl_divergence_tensor_list.append(kl_divergence_values)
                    kl_divergence_tensor_after = torch.stack(kl_divergence_tensor_list).transpose(0, 1).sum(dim=[1, 2, 3, 4])
                    kl_diff_sum = kl_divergence_tensor_before - kl_divergence_tensor_after
                    
                    # Stack the tensors in each queue
                    stacked_kl_diff_lasso = torch.stack(list(kl_diff_lasso_queue), dim=0)
                    stacked_kl_diff_sum_lasso = torch.stack(list(kl_diff_sum_lasso_queue), dim=0)
                            
                    # Featrue selection
                    if stacked_kl_diff_lasso.shape[0] == 1:
                        pass
                    else:
                        # gets error when non_zero_coef_timesteps is 0
                        range_T_array = np.array([1e-8, 0.00001, 0.001, 0.1, 0.3, 0.5, 0.8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
                        non_zero_coef_timesteps_indices = feature_selector(stacked_kl_diff_lasso, stacked_kl_diff_sum_lasso, logger, global_steps, dist).tolist()
                        non_zero_coef_timesteps = range_T_array[non_zero_coef_timesteps_indices].tolist()
                        print(non_zero_coef_timesteps)
                    
                    # REINFORCE 
                    reward = copy.deepcopy(kl_diff_sum.detach())
                    modified_reward = reward
                    modified_reward += 0 * entropy 
                    # Update the moving average
                    # alpha = 0.5  
                    # reward_moving_avg = reward_moving_avg if 'reward_moving_avg' in locals() else reward.mean().repeat(images.shape[0])
                    # reward_moving_avg = alpha * reward.mean().repeat(images.shape[0]) + (1 - alpha) * reward_moving_avg
                    # modified_reward = reward - reward_moving_avg
                    # modified_reward = (modified_reward - modified_reward.mean()) / (modified_reward.std() + 1e-8)
                    # modified_reward += 1e-2 * entropy 
                    
                    actor_loss = -(log_probs * modified_reward).mean()
                    actor_loss.mul(loss_scaling / batch_gpu_total).backward()
                    # for g in actor_optimizer.param_groups:
                    #     g['lr'] = actor_optimizer_kwargs['lr']
                    actor_optimizer.step()
                    

        # Update EMA.
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))
        for p_ema, p_net in zip(actor_ema.parameters(), actor_net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        # Perform maintenance tasks once per tick.
        cur_nimg += batch_size
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        if dist.get_rank() == 0:
            logger.log({
                'diffusion/loss': loss.sum().item(),
                'diffusion/lr': optimizer_kwargs['lr'] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1),
                "kl_divergence/kl_diff_sum": kl_diff_sum.mean().item(),
                "kl_divergence/kl_divergence_tensor_before": kl_divergence_tensor_before.mean().item(),
                "kl_divergence/kl_divergence_tensor_after": kl_divergence_tensor_after.mean().item(),
                "policy/actor_loss": actor_loss.item(),
                "policy/log_prob": log_probs.mean().item(),
                "policy/reward": reward.mean(),
                "policy/entropy": entropy.mean(),
                "policy/normal_mean": normal_mean.mean(),
                "policy/normal_std": normal_std.mean(),
            }, step=global_steps)


            # Create a figure with subplots
            fig, axes = plt.subplots(2, 1, figsize=(18, 6))
            (ax1, ax2) = axes

            # Plot histogram of timesteps
            ax1.hist(timestep.view(-1,).detach().cpu().numpy(), bins=50, alpha=0.7)
            ax1.set_xlabel('Timesteps')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Histogram of Timesteps')

            # Plot normal distributions of policy
            x_axis = np.linspace(0, 1, 1000)
            for a, b in zip(normal_mean, normal_std):
                a = a.item()  # Convert tensor value to a scalar
                b = b.item()  # Convert tensor value to a scalar
                y = norm.pdf(x_axis, a, b)  # Compute the normal distribution's PDF
                ax2.plot(x_axis, y, alpha=0.1)  # Plot with some transparency
            ax2.set_xlabel('x_axis')
            ax2.set_ylabel('Density')
            ax2.set_title('Normal Distributions')

            # Save the figure
            plt.savefig("combined_plot.png")

            # Log the figure using logger
            logger.log({"KL_diveregence, timestep histogram, and normal distributions": logger.Image(plt)}, step=global_steps)
            plt.close()



        global_steps += 1

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        fields += [f"loss {loss.sum().item():<10.2f}"]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(' '.join(fields))

        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0('Aborting...')

        # Save network snapshot.
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            data = dict(ema=ema, actor_ema=actor_ema, loss_fn=loss_fn, augment_pipe=augment_pipe, dataset_kwargs=dict(dataset_kwargs))
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value # conserve memory
            if dist.get_rank() == 0:
                with open(os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl'), 'wb') as f:
                    pickle.dump(data, f)
            del data # conserve memory

        # Save full dump of the training state.
        if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and cur_tick != 0 and dist.get_rank() == 0:
            torch.save(dict(net=net, actor_net=actor_net, optimizer_state=optimizer.state_dict()), os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt'))
        
        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
            stats_jsonl.write(json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n')
            stats_jsonl.flush()
        dist.update_progress(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    dist.print0()
    dist.print0('Exiting...')

#----------------------------------------------------------------------------
