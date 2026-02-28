"""
Fine-tuning Training Script for Molecular Property Prediction

This script implements training pipelines for both regression and classification tasks
on molecular property prediction using Graph Neural Networks (GNNs). It supports:

- Transfer learning with pre-trained model loading
- Distributed training across multiple GPUs/nodes
- Automatic Mixed Precision (AMP) training with bfloat16
- Model Exponential Moving Average (EMA) for improved generalization
- Flexible learning rate scheduling and optimization
- Comprehensive logging and checkpoint management

Usage:
    python main.py --mode regression --name <property_name> [additional arguments]
    python main.py --mode classification --name <property_name> [additional arguments]

Author: JunyiAn
Date: 2026-02-28
"""

import argparse
import os
import time
from datetime import datetime

import numpy as np
import torch
from torch_geometric.loader import DataLoader

# Logging and utilities
from logger import FileLogger
import utils

# Model training and optimization utilities from timm (PyTorch Image Models)
from timm.utils import NativeScaler, ModelEmaV2
from timm.scheduler import create_scheduler
from optim_factory import create_optimizer

# Custom dataset and training modules
from suiren_datasets.org_mol2d import PP_smiles_2d
from engine import (
    train_one_epoch, evaluate, compute_stats,
    train_cls_one_epoch, evaluate_cls
)

def get_args_parser():
    """
    Parse command line arguments for model fine-tuning.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser for the training script
    """
    parser = argparse.ArgumentParser(
        'Fine-tuning for molecular property prediction',
        add_help=False
    )
    
    # ========================================================================
    # Model Configuration
    # ========================================================================
    parser.add_argument('--checkpoint-pretrain', type=str, default=None,
                        help='Path to pre-trained model checkpoint')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to resume training from checkpoint')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for checkpoints and logs')
    
    # ========================================================================
    # Task Configuration
    # ========================================================================
    parser.add_argument('--mode', type=str, default='regression',
                        help='Run mode: regression or classification')
    parser.add_argument('--name', type=str, required=True,
                        help='Property name for dataset and experiment naming')
    parser.add_argument('--loss', type=str, default='l1', choices=['l1', 'l2'],
                        help='Loss function: l1 (MAE) or l2 (MSE) for regression; Cross-entropy is default for classification tast')
    parser.add_argument('--main-metric', type=str, default='MAE',
                        help='Primary metric for model selection (MAE, R2, ACC, AUROC, AUPRC)')
    parser.add_argument('--class-num', type=int, default=2,
                        help='Number of classes for classification tasks')
    
    # ========================================================================
    # Data Configuration
    # ========================================================================
    parser.add_argument('--data-mode', type=str, default='smiles_random',
                        choices=['smiles_random', 'smiles_defined'],
                        help='Data loading mode: random split or predefined split')
    parser.add_argument('--tvt', action='store_true',
                        help='Use train/val/test split (default: train/val only)')
    parser.add_argument('--ratio', type=float, default=0.8,
                        help='Train/val split ratio for random splitting')
    parser.add_argument('--compute-stats', action='store_true', dest='compute_stats',
                        help='Compute and display dataset statistics only')
    
    # ========================================================================
    # Training Hyperparameters
    # ========================================================================
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size per GPU')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducibility')
    
    # ========================================================================
    # Optimizer Configuration
    # ========================================================================
    parser.add_argument('--opt', type=str, default='adamw',
                        help='Optimizer type (adamw, sgd, etc.)')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='L2 regularization coefficient')
    parser.add_argument('--opt-eps', type=float, default=1e-8,
                        help='Optimizer epsilon for numerical stability')
    parser.add_argument('--opt-betas', type=float, nargs='+', default=None,
                        help='Adam optimizer betas')
    parser.add_argument('--clip-grad', type=float, default=None,
                        help='Gradient clipping norm threshold')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum coefficient')
    
    # ========================================================================
    # Learning Rate Schedule
    # ========================================================================
    parser.add_argument('--sched', type=str, default='cosine',
                        help='Learning rate scheduler (cosine, step, etc.)')
    parser.add_argument('--warmup-epochs', type=int, default=0,
                        help='Number of warmup epochs')
    parser.add_argument('--warmup-lr', type=float, default=1e-6,
                        help='Warmup initial learning rate')
    parser.add_argument('--min-lr', type=float, default=1e-6,
                        help='Minimum learning rate for scheduler')
    parser.add_argument('--decay-epochs', type=float, default=30,
                        help='Epoch interval to decay learning rate')
    parser.add_argument('--decay-rate', type=float, default=0.1,
                        help='Learning rate decay rate')
    parser.add_argument('--cooldown-epochs', type=int, default=10,
                        help='Cooldown epochs after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10,
                        help='Patience for ReduceLROnPlateau scheduler')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None,
                        help='LR noise schedule (epoch percentages)')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67,
                        help='LR noise amplitude as percentage')
    parser.add_argument('--lr-noise-std', type=float, default=1.0,
                        help='LR noise standard deviation')
    
    # ========================================================================
    # Model EMA (Exponential Moving Average)
    # ========================================================================
    parser.add_argument('--model-ema', action='store_true',
                        help='Enable exponential moving average model')
    parser.add_argument('--model-ema-decay', type=float, default=0.9999,
                        help='EMA decay coefficient')
    parser.add_argument('--model-ema-force-cpu', action='store_true',
                        help='Force EMA model to CPU')
    
    # ========================================================================
    # Regularization
    # ========================================================================
    parser.add_argument('--drop-path', type=float, default=0.0,
                        help='Drop path rate for stochastic depth')
    
    # ========================================================================
    # Data Loading
    # ========================================================================
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of data loading workers')
    parser.add_argument('--pin-mem', action='store_true', default=True,
                        help='Pin memory in DataLoader')
    
    # ========================================================================
    # Mixed Precision Training (AMP)
    # ========================================================================
    parser.add_argument('--amp', action='store_true',
                        help='Enable automatic mixed precision (bfloat16) training')
    
    # ========================================================================
    # Distributed Training
    # ========================================================================
    parser.add_argument('--world_size', type=int, default=1,
                        help='Number of distributed processes')
    parser.add_argument('--local-rank', type=int, default=0,
                        help='Local rank in distributed training')
    parser.add_argument('--dist_url', type=str, default='env://',
                        help='URL for distributed training setup')
    
    # ========================================================================
    # Logging
    # ========================================================================
    parser.add_argument('--print-freq', type=int, default=100,
                        help='Print frequency (batches)')
    
    return parser


def train_regression(args):
    """
    Training pipeline for regression tasks (continuous property prediction).
    
    This function implements the complete training workflow including:
    - Dataset loading and preprocessing
    - Model initialization with pre-trained weights
    - Optimizer and learning rate scheduler setup
    - Training loop with validation and EMA evaluation
    - Checkpoint saving and best model tracking
    
    Args:
        args (Namespace): Parsed command line arguments containing all training configurations
    
    Returns:
        None: Results are saved to checkpoints and logged to files
    """

    # ========================================================================
    # Stage 1: Initialization
    # ========================================================================
    # Generate timestamp for experiment tracking
    exp_time = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    
    # Initialize distributed training environment
    utils.init_distributed_mode(args)
    is_main_process = (args.rank == 0)
    
    # Create logging directory and logger
    if is_main_process:
        if not os.path.exists('logs/' + args.name):
            os.makedirs('logs/' + args.name)
    
    _log = FileLogger(
        is_master=is_main_process,
        is_rank0=is_main_process,
        output_dir='logs/' + args.name + '/',
        time_name=exp_time
    )
    _log.info(args)
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ========================================================================
    # Stage 2: Dataset Loading and Preprocessing
    # ========================================================================
    if args.data_mode == 'smiles_random':
        _log.info(f'Random spliting dataset to train dataset and valid dataset with ratio {args.ratio}')
        data_path = 'data/' + args.name
        train_dataset = PP_smiles_2d(data_path, 'train', args.name, args.ratio)
        val_dataset   = PP_smiles_2d(data_path, 'valid', args.name, args.ratio)
    elif args.data_mode == 'smiles_defined':
        if args.tvt:
            _log.info('Using defined train, valid, test dataset')
            data_path = 'data/' + args.name
            train_dataset = PP_smiles_2d(data_path, 'train', args.name, args.ratio)
            val_dataset    = PP_smiles_2d(data_path, 'valid', args.name, args.ratio)
            test_dataset   = PP_smiles_2d(data_path, 'test', args.name, defined=True)
        else:
            _log.info('Using defined train, valid dataset')
            data_path = 'data/' + args.name
            train_dataset = PP_smiles_2d(data_path, 'train', args.name, defined=True)
            val_dataset   = PP_smiles_2d(data_path, 'valid', args.name, defined=True)
    else:
        raise ValueError('Unseen data file.')
    
    # Log dataset warnings and statistics
    if train_dataset.exceed_ele is not None:
        _log.info(train_dataset.exceed_ele)
    if train_dataset.fail_mole is not None and val_dataset.fail_mole is not None:
        _log.info('Fail molecules in Training set: {}, Fail molecules in Valid set size:{}'.format(
            train_dataset.fail_mole, val_dataset.fail_mole))
    _log.info('Training set size: {}, Valid set size:{}'.format(
        len(train_dataset), len(val_dataset)))
    
    # Compute normalization factors (mean, std) for regression target
    norm_factor = [train_dataset.mean(), train_dataset.std()]
    _log.info('Training set mean: {}, std:{}'.format(
        norm_factor[0], norm_factor[1]))
    
    # Reset random seeds after dataset loading
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # ========================================================================
    # Stage 3: Model Setup
    # ========================================================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model with pre-trained and fine-tune components
    from models.finetune_model import standard_finetune
    model = standard_finetune(class_flag=False, class_num=2)
    model = model.to(device)

    # Load pre-trained model weights if provided
    if args.checkpoint_pretrain is not None:
        _log.info('Start loading pretrain model')
        checkpoint = torch.load(args.checkpoint_pretrain, map_location=torch.device('cpu'))
        model.pretrain_model.load_state_dict(checkpoint)
        _log.info('Load pretrain model successfully')
    
    # Resume from checkpoint if provided (for continuing training)
    if args.resume is not None:
        _log.info('Start loading checkpoint')
        checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
        _log.info('Load checkpoint successfully')
    
    # Freeze pre-trained model parameters to preserve learned representations
    _log.info('Freezing the pretrain model')
    frozen_modules = [model.pretrain_model]
    for module in frozen_modules:
        for _, param in module.named_parameters():
            param.requires_grad = False
    
    # Initialize Exponential Moving Average (EMA) model for improved generalization
    model_ema = None
    if args.model_ema:
        model_ema = ModelEmaV2(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else None)

    # Wrap model for distributed training
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank])

    # Log model parameters
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _log.info('Number of trainable params: {}'.format(n_parameters))
    
    # ========================================================================
    # Stage 4: Optimizer and Learning Rate Scheduler
    # ========================================================================
    optimizer = create_optimizer(args, model)
    lr_scheduler, _ = create_scheduler(args, optimizer)
    
    # Create loss function based on task type
    if args.loss == 'l1':
        criterion = torch.nn.L1Loss()
    elif args.loss == 'l2':
        criterion = torch.nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss type: {args.loss}")

    # ========================================================================
    # Stage 5: Automatic Mixed Precision (AMP) Setup
    # ========================================================================
    # Setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = False  # Disabled by default
    loss_scaler = None
    if args.amp:
        amp_autocast = True
        loss_scaler = NativeScaler()

    # ========================================================================
    # Stage 6: Data Loader Setup
    # ========================================================================
    # ========================================================================
    # Stage 6: Data Loader Setup
    # ========================================================================
    if args.distributed:
        # Use DistributedSampler for multi-GPU training
        sampler_train = torch.utils.data.DistributedSampler(
            train_dataset,
            num_replicas=utils.get_world_size(),
            rank=utils.get_rank(),
            shuffle=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler_train,
            num_workers=args.workers,
            pin_memory=args.pin_mem,
            drop_last=True
        )
    else:
        # Single GPU training
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=args.pin_mem,
            drop_last=True
        )
    
    # Validation and test loaders (no shuffling needed)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    if args.tvt:
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # ========================================================================
    # Stage 7: Compute Dataset Statistics (Optional)
    # ========================================================================
    if args.compute_stats:
        compute_stats(train_loader, max_radius=args.radius, logger=_log, print_freq=args.print_freq)
        return
    
    # ========================================================================
    # Stage 8: Training Loop
    # ========================================================================
    # Initialize best metrics tracking
    best_epoch = 0
    best_train_r2, best_train_err = 0, float('inf')
    best_val_r2, best_val_err = 0, float('inf')
    best_test_r2, best_test_err = 0, float('inf')
    best_ema_epoch = 0
    best_ema_val_r2, best_ema_val_err = 0, float('inf')
    best_ema_test_r2, best_ema_test_err = 0, float('inf')
    
    _log.info('Start training')
    for epoch in range(args.epochs):
        _log.info(f"Training property: {args.name}")
        epoch_start_time = time.perf_counter()
        
        # Update learning rate scheduler
        lr_scheduler.step(epoch)

        # Set epoch for distributed sampler (ensures different shuffle each epoch)
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        
        # ====================================================================
        # Training Phase
        # ====================================================================
        train_err, train_r2 = train_one_epoch(
            model=model,
            criterion=criterion,
            norm_factor=norm_factor,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            model_ema=model_ema,
            amp_autocast=amp_autocast,
            loss_scaler=loss_scaler,
            print_freq=args.print_freq,
            logger=_log
        )
        
        # ====================================================================
        # Validation Phase
        # ====================================================================
        val_err, val_r2, val_loss = evaluate(
            model,
            norm_factor,
            val_loader,
            device,
            amp_autocast=amp_autocast,
            print_freq=args.print_freq,
            logger=_log
        )
        
        # ====================================================================
        # Checkpoint Management
        # ====================================================================
        checkpoints_dir = f'checkpoints/{args.name}/{exp_time}'
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        # Determine if this is the best model based on selected metric
        if args.main_metric == 'MAE':
            best_flag = (val_err < best_val_err)
        elif args.main_metric == 'R2':
            best_flag = (val_r2 > best_val_r2)
        else:
            best_flag = (val_err < best_val_err)

        # Save best model
        if best_flag:
            best_val_err = val_err
            best_train_err = train_err
            best_train_r2 = train_r2
            best_val_r2 = val_r2
            best_epoch = epoch

            # Evaluate on test set if available
            if args.tvt:
                test_err, test_r2, test_loss = evaluate(
                    model,
                    norm_factor,
                    test_loader,
                    device,
                    amp_autocast=amp_autocast,
                    print_freq=args.print_freq,
                    logger=_log
                )
                info_str = (f'Best Test -- Epoch: [{epoch}], '
                           f'MAE: {test_err:.5f}, R2: {test_r2:.5f}\n')
                _log.info(info_str)
                best_test_err = test_err
                best_test_r2 = test_r2

            # Save checkpoint (only on main process in distributed training)
            if is_main_process:
                checkpoint_data = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_err": best_val_err,
                    "best_test_err": best_test_err,
                    "norm_factor": norm_factor
                }
                torch.save(checkpoint_data, f'{checkpoints_dir}/{args.name}_2d.pt')

        # ====================================================================
        # Logging
        # ====================================================================
        train_log = (f'Epoch: [{epoch}], '
                    f'Train MAE: {train_err:.5f}, Train R2: {train_r2:.5f}, '
                    f'Val MAE: {val_err:.5f}, Val R2: {val_r2:.5f}, '
                    f'Time: {time.perf_counter() - epoch_start_time:.2f}s')
        _log.info(train_log)
        
        best_log = (f'Best: Epoch={best_epoch}, '
                   f'Train MAE: {best_train_err:.5f}, Train R2: {best_train_r2:.5f}, '
                   f'Val MAE: {best_val_err:.5f}, Val R2: {best_val_r2:.5f}, '
                   f'Test MAE: {best_test_err:.5f}, Test R2: {best_test_r2:.5f}\n')
        _log.info(best_log)
        
        # ====================================================================
        # EMA Model Evaluation (if enabled)
        # ====================================================================
        if model_ema is not None:
            ema_val_err, ema_val_r2, _ = evaluate(
                model_ema.module,
                norm_factor,
                val_loader,
                device,
                amp_autocast=amp_autocast,
                print_freq=args.print_freq,
                logger=_log
            )
            
            # Track best EMA model
            if ema_val_err < best_ema_val_err:
                best_ema_val_err = ema_val_err
                best_ema_val_r2 = ema_val_r2
                best_ema_epoch = epoch

                # Evaluate EMA model on test set
                if args.tvt:
                    test_ema_err, test_ema_r2, _ = evaluate(
                        model_ema.module,
                        norm_factor,
                        test_loader,
                        device,
                        amp_autocast=amp_autocast,
                        print_freq=args.print_freq,
                        logger=_log
                    )
                    info_str = (f'Best EMA Test -- Epoch: [{epoch}], '
                               f'MAE: {test_ema_err:.5f}, R2: {test_ema_r2:.5f}\n')
                    _log.info(info_str)
                    best_ema_test_err = test_ema_err
                    best_ema_test_r2 = test_ema_r2

                # Save EMA checkpoint
                if is_main_process:
                    checkpoint_data = {
                        "state_dict": model_ema.module.state_dict(),
                        "epoch": epoch,
                        "best_val_ema_err": best_ema_val_err,
                        "best_test_ema_err": best_ema_test_err,
                        "norm_factor": norm_factor
                    }
                    torch.save(checkpoint_data, f'{checkpoints_dir}/{args.name}_ema.pt')
    
            ema_log = (f'EMA -- Epoch: [{epoch}], '
                      f'Val MAE: {ema_val_err:.5f}, Val R2: {ema_val_r2:.5f}, '
                      f'Time: {time.perf_counter() - epoch_start_time:.2f}s')
            _log.info(ema_log)
            
            best_ema_log = (f'Best EMA: Epoch={best_ema_epoch}, '
                           f'Val MAE: {best_ema_val_err:.5f}, Val R2: {best_ema_val_r2:.5f}\n')
            _log.info(best_ema_log)


def train_classification(args):
    """
    Training pipeline for classification tasks (categorical property prediction).
    
    This function implements the complete training workflow for classification including:
    - Dataset loading with classification labels
    - Model initialization with appropriate output heads
    - Training with cross-entropy loss
    - Evaluation using accuracy, AUROC, and AUPRC metrics
    - EMA model tracking for improved generalization
    
    Args:
        args (Namespace): Parsed command line arguments containing all training configurations
    
    Returns:
        None: Results are saved to checkpoints and logged to files
    """

    # ========================================================================
    # Stage 1: Initialization
    # ========================================================================
    # Generate timestamp for experiment tracking
    exp_time = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    
    # Initialize distributed training environment
    utils.init_distributed_mode(args)
    is_main_process = (args.rank == 0)
    
    # Create logging directory and logger
    if is_main_process:
        if not os.path.exists('logs/' + args.name):
            os.makedirs('logs/' + args.name)
    
    _log = FileLogger(
        is_master=is_main_process,
        is_rank0=is_main_process,
        output_dir='logs/' + args.name + '/',
        time_name=exp_time
    )
    _log.info(args)
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
   
    # ========================================================================
    # Stage 2: Dataset Loading and Preprocessing
    # ========================================================================
    if args.data_mode == 'smiles_random':
        _log.info(f'Random spliting dataset to train dataset and valid dataset with ratio {args.ratio}')
        data_path = 'data/' + args.name
        train_dataset = PP_smiles_2d(data_path, 'train', args.name, args.ratio, classification=True)
        val_dataset   = PP_smiles_2d(data_path, 'valid', args.name, args.ratio, classification=True)
    elif args.data_mode == 'smiles_defined':
        if args.tvt:
            _log.info('Using defined train, valid, test dataset')
            data_path = 'data/' + args.name
            train_dataset = PP_smiles_2d(data_path, 'train', args.name, args.ratio, classification=True)
            val_dataset    = PP_smiles_2d(data_path, 'valid', args.name, args.ratio, classification=True)
            test_dataset   = PP_smiles_2d(data_path, 'test', args.name, defined=True, classification=True)
        else:
            _log.info('Using defined train, valid dataset')
            data_path = 'data/' + args.name
            train_dataset = PP_smiles_2d(data_path, 'train', args.name, defined=True, classification=True)
            val_dataset   = PP_smiles_2d(data_path, 'valid', args.name, defined=True, classification=True)
    else:
        raise ValueError('Unseen data file.')
    
    # Log dataset warnings and statistics
    if train_dataset.exceed_ele is not None:
        _log.info(train_dataset.exceed_ele)
    if train_dataset.fail_mole is not None and val_dataset.fail_mole is not None:
        _log.info('Fail molecules in Training set: {}, Fail molecules in Valid set size:{}'.format(
            train_dataset.fail_mole, val_dataset.fail_mole))
    _log.info('Training set size: {}, Valid set size:{}'.format(
        len(train_dataset), len(val_dataset)))
    
    # Get class number from dataset
    class_num = train_dataset.class_num
    _log.info('Number of classes: {}'.format(class_num))
    
    # Reset random seeds after dataset loading
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # ========================================================================
    # Stage 3: Model Setup
    # ========================================================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model with pre-trained and fine-tune components for classification
    from models.finetune_model import standard_finetune
    model = standard_finetune(class_flag=True, class_num=class_num)
    model = model.to(device)

    # Load pre-trained model weights if provided
    if args.checkpoint_pretrain is not None:
        _log.info('Start loading pretrain model')
        checkpoint = torch.load(args.checkpoint_pretrain, map_location=torch.device('cpu'))
        model.pretrain_model.load_state_dict(checkpoint)
        _log.info('Load pretrain model successfully')
    
    # Resume from checkpoint if provided (for continuing training)
    if args.resume is not None:
        _log.info('Start loading checkpoint')
        checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
        _log.info('Load checkpoint successfully')
    
    # Freeze pre-trained model parameters to preserve learned representations
    _log.info('Freezing the pretrain model')
    frozen_modules = [model.pretrain_model]
    for module in frozen_modules:
        for _, param in module.named_parameters():
            param.requires_grad = False
    
    # Move model to device
    model = model.to(device)
    
    # Initialize Exponential Moving Average (EMA) model for improved generalization
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEmaV2(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else None)

    # Wrap model for distributed training
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank])

    # Log model parameters
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _log.info('Number of trainable params: {}'.format(n_parameters))
    
    # ========================================================================
    # Stage 4: Optimizer and Learning Rate Scheduler
    # ========================================================================
    optimizer = create_optimizer(args, model)
    lr_scheduler, _ = create_scheduler(args, optimizer)
    
    # Cross-entropy loss for multi-class classification
    criterion = torch.nn.CrossEntropyLoss()
    
    # ========================================================================
    # Stage 5: Automatic Mixed Precision (AMP) Setup
    # ========================================================================
    # Setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = False  # Disabled by default
    loss_scaler = None
    if args.amp:
        amp_autocast = True
        loss_scaler = NativeScaler()
    
    # ========================================================================
    # Stage 6: Data Loader Setup
    # ========================================================================
    # ========================================================================
    # Stage 6: Data Loader Setup
    # ========================================================================
    if args.distributed:
        # Use DistributedSampler for multi-GPU training
        sampler_train = torch.utils.data.DistributedSampler(
            train_dataset,
            num_replicas=utils.get_world_size(),
            rank=utils.get_rank(),
            shuffle=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler_train,
            num_workers=args.workers,
            pin_memory=args.pin_mem,
            drop_last=True
        )
    else:
        # Single GPU training
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=args.pin_mem,
            drop_last=True
        )
    
    # Validation and test loaders (no shuffling needed)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    if args.tvt:
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # ========================================================================
    # Stage 7: Compute Dataset Statistics (Optional)
    # ========================================================================
    if args.compute_stats:
        compute_stats(train_loader, max_radius=args.radius, logger=_log, print_freq=args.print_freq)
        return
    
    # ========================================================================
    # Stage 8: Training Loop
    # ========================================================================
    # Initialize best metrics tracking
    best_epoch = 0
    best_train_acc = 0
    best_train_auprc, best_train_auroc = 0, 0
    best_val_acc = 0
    best_val_auprc, best_val_auroc = 0, 0
    best_test_acc = 0
    best_train_err, best_val_err = float('inf'), float('inf')
    best_ema_epoch = 0
    best_ema_val_acc = 0
    best_ema_val_err = float('inf')
    
    _log.info('Start training')
    for epoch in range(args.epochs):
        _log.info(f"Training property: {args.name}")
        epoch_start_time = time.perf_counter()
        
        # Update learning rate scheduler
        lr_scheduler.step(epoch)

        # Set epoch for distributed sampler (ensures different shuffle each epoch)
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        
        # ====================================================================
        # Training Phase
        # ====================================================================
        train_loss, train_acc, train_auprc, train_auroc = train_cls_one_epoch(
            model=model,
            criterion=criterion,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            model_ema=model_ema,
            amp_autocast=amp_autocast,
            loss_scaler=loss_scaler,
            print_freq=args.print_freq,
            logger=_log
        )
        
        # ====================================================================
        # Validation Phase
        # ====================================================================
        val_loss, val_acc, val_auprc, val_auroc = evaluate_cls(
            model,
            val_loader,
            device,
            criterion,
            amp_autocast=amp_autocast,
            print_freq=args.print_freq,
            logger=_log
        )
        
        # ====================================================================
        # Checkpoint Management
        # ====================================================================
        checkpoints_dir = f'checkpoints/{args.name}/{exp_time}'
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)

        # Determine if this is the best model based on selected metric
        best_flag = False
        if args.main_metric == 'ACC':
            best_flag = (val_acc > best_val_acc)
        elif args.main_metric == 'AUPRC':
            best_flag = (val_auprc > best_val_auprc)
        elif args.main_metric == 'AUROC':
            best_flag = (val_auroc > best_val_auroc)
        else:
            # Default to accuracy
            best_flag = (val_acc > best_val_acc)

        # Save best model
        if best_flag:
            best_val_err = val_loss
            best_train_err = train_loss
            best_train_acc = train_acc
            best_val_acc = val_acc
            best_epoch = epoch
            best_val_auprc = val_auprc
            best_val_auroc = val_auroc

            # Evaluate on test set if available
            if args.tvt:
                test_loss, test_acc, test_auprc, test_auroc= evaluate_cls(
                    model,
                    test_loader,
                    device,
                    criterion,
                    amp_autocast=amp_autocast,
                    print_freq=args.print_freq,
                    logger=_log
                )
                info_str = (f'Best Test -- Epoch: [{epoch}], '
                           f'Accuracy: {test_acc:.5f}\n')
                _log.info(info_str)
                best_test_acc = test_acc

            # Save checkpoint (only on main process in distributed training)
            if is_main_process:
                checkpoint_data = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_acc": best_val_acc,
                    "best_test_acc": best_test_acc
                }
                torch.save(checkpoint_data, f'{checkpoints_dir}/{args.name}_classification.pt')

        # ====================================================================
        # Logging
        # ====================================================================
        train_log = (f'Epoch: [{epoch}], '
                    f'Train Acc: {train_acc:.5f}, Train AUROC: {train_auroc:.5f}, Train AUPRC: {train_auprc:.5f}, '
                    f'Val Acc: {val_acc:.5f}, Val AUROC: {val_auroc:.5f}, Val AUPRC: {val_auprc:.5f}, '
                    f'Time: {time.perf_counter() - epoch_start_time:.2f}s')
        _log.info(train_log)
        
        best_log = (f'Best: Epoch={best_epoch}, '
                   f'Train Loss: {best_train_err:.5f}, Train Acc: {best_train_acc:.5f}, '
                   f'Val Loss: {best_val_err:.5f}, Val Acc: {best_val_acc:.5f}, '
                   f'Val AUROC: {best_val_auroc:.5f}, Val AUPRC: {best_val_auprc:.5f}, '
                   f'Test Acc: {best_test_acc:.5f}\n')
        _log.info(best_log)
        
        # ====================================================================
        # EMA Model Evaluation (if enabled)
        # ====================================================================
        if model_ema is not None:
            ema_val_loss, ema_val_acc = evaluate_cls(
                model_ema.module,
                val_loader,
                device,
                criterion,
                amp_autocast=amp_autocast,
                print_freq=args.print_freq,
                logger=_log
            )
            
            # Track best EMA model
            if ema_val_acc > best_ema_val_acc:
                best_ema_val_acc = ema_val_acc
                best_ema_epoch = epoch
                
                # Save EMA checkpoint
                if is_main_process:
                    checkpoint_data = {
                        "state_dict": model.state_dict(),
                        "epoch": epoch,
                        "best_val_acc": best_val_acc
                    }
                    torch.save(checkpoint_data, f'{checkpoints_dir}/{args.name}_ema.pt')
    
            ema_log = (f'EMA -- Epoch: [{epoch}], '
                      f'Val Loss: {ema_val_loss:.5f}, Val Acc: {ema_val_acc:.5f}, '
                      f'Time: {time.perf_counter() - epoch_start_time:.2f}s')
            _log.info(ema_log)
            
            best_ema_log = (f'Best EMA: Epoch={best_ema_epoch}, '
                           f'Val Loss: {best_ema_val_err:.5f}, Val Acc: {best_ema_val_acc:.5f}\n')
            _log.info(best_ema_log)
            

if __name__ == "__main__":
    """
    Main entry point for the fine-tuning training script.
    
    This script supports two training modes:
    - regression: For continuous property prediction
    - classification: For categorical property prediction
    
    The mode is selected via the --mode argument when running the script.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        'Fine tuning for various molecular properties',
        parents=[get_args_parser()]
    )
    args = parser.parse_args()
    
    # Route to appropriate training function based on mode
    if args.mode == 'regression':
        train_regression(args)
    elif args.mode == 'classification':
        train_classification(args)
    else:
        raise ValueError(f'Unknown training mode: {args.mode}. Must be "regression" or "classification"')