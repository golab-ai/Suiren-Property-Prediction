import torch
from typing import Iterable, Optional
from timm.utils import ModelEmaV2, dispatch_clip_grad
import time
from torch_cluster import radius_graph
import torch_geometric
from sklearn.metrics import r2_score

from torcheval.metrics.functional import binary_auroc, binary_auprc

ModelEma = ModelEmaV2


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        """Initialize the meter with zero values."""
        self.reset()

    def reset(self):
        """Reset all metrics to zero."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Update the meter with a new value.
        
        Args:
            val: The value to add
            n: The count of items (default: 1)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    norm_factor: list,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    model_ema: Optional[ModelEma] = None,
    amp_autocast: bool = False,
    loss_scaler=None,
    clip_grad=None,
    print_freq: int = 100,
    logger=None,
):
    """
    Train the model for one epoch.
    
    Args:
        model: The neural network model
        criterion: The loss function
        norm_factor: List containing [mean, std] for normalization
        data_loader: DataLoader for training data
        optimizer: The optimizer for updating weights
        device: The device to run on (CPU/GPU)
        epoch: Current epoch number
        model_ema: Optional exponential moving average model
        amp_autocast: Whether to use automatic mixed precision
        loss_scaler: Optional loss scaler for gradient scaling
        clip_grad: Gradient clipping threshold
        print_freq: Frequency of logging (every N steps)
        logger: Logger instance for recording metrics
        
    Returns:
        Tuple of (average_mae, r2_score)
    """
    model.train()
    criterion.train()

    loss_metric = AverageMeter()
    mae_metric = AverageMeter()

    start_time = time.perf_counter()

    task_mean = norm_factor[0]
    task_std = norm_factor[1]

    all_targets = []
    all_preds = []

    for step, data in enumerate(data_loader):
        data = data.to(device)
        
        # Forward pass with automatic mixed precision
        with torch.autocast(
            device_type=device.type,
            enabled=amp_autocast,
            dtype=torch.bfloat16,
        ):
            pred = model(data)
            pred = pred.view(-1)
            # Normalize targets using mean and std
            loss = criterion(pred, (data.y - task_mean) / task_std)

        # Backward pass and optimization
        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(loss, optimizer, parameters=model.parameters())
        else:
            loss.backward()
            # for names, param in model.named_parameters():
            #     if param.grad is None:
            #         print(names)
            if clip_grad is not None:
                dispatch_clip_grad(model.parameters(), value=clip_grad, mode="norm")
            optimizer.step()

        # Update metrics
        loss_metric.update(loss.item(), n=pred.shape[0])
        err = pred.detach() * task_std + task_mean - data.y
        mae_metric.update(torch.mean(torch.abs(err)).item(), n=pred.shape[0])

        all_targets.append(data.y.cpu())
        all_preds.append((pred.detach() * task_std + task_mean).cpu())

        # Update EMA model if provided
        if model_ema is not None:
            model_ema.update(model)

        torch.cuda.synchronize()

        # Logging
        if step % print_freq == 0 or step == len(data_loader) - 1:
            elapsed_time = time.perf_counter() - start_time
            progress = (step + 1) / len(data_loader)
            time_per_step = 1e3 * elapsed_time / progress / len(data_loader)
            
            info_str = (
                f"Epoch: [{epoch}][{step}/{len(data_loader)}] "
                f"loss: {loss_metric.avg:.5f}, "
                f"MAE: {mae_metric.avg:.5f}, "
                f"time/step={time_per_step:.0f}ms, "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )
            logger.info(info_str)

    # Compute final metrics
    all_targets = torch.cat(all_targets, dim=0).numpy().flatten()
    all_preds = torch.cat(all_preds, dim=0).to(dtype=torch.float).numpy().flatten()
    r2 = r2_score(all_targets, all_preds)

    return mae_metric.avg, r2


def evaluate(
    model: torch.nn.Module,
    norm_factor: list,
    data_loader: Iterable,
    device: torch.device,
    amp_autocast: bool = False,
    print_freq: int = 100,
    logger=None,
    epoch: int = 0,
    debug_bad_example: bool = False,
    threshold: float = 0.35,
):
    """
    Evaluate the model on validation/test data.
    
    Args:
        model: The neural network model
        norm_factor: List containing [mean, std] for normalization
        data_loader: DataLoader for evaluation data
        device: The device to run on
        amp_autocast: Whether to use automatic mixed precision
        print_freq: Frequency of logging
        logger: Logger instance
        epoch: Current epoch number
        debug_bad_example: Whether to track examples with high errors
        threshold: Error threshold for identifying bad examples
        
    Returns:
        Tuple of (average_mae, r2_score, average_loss)
    """
    model.eval()

    loss_metric = AverageMeter()
    mae_metric = AverageMeter()
    criterion = torch.nn.L1Loss()
    criterion.eval()

    task_mean = norm_factor[0]
    task_std = norm_factor[1]

    all_targets = []
    all_preds = []
    # worse_example = [] if debug_bad_example else None
    # worse_err = [] if debug_bad_example else None

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)

            with torch.autocast(
                device_type=device.type,
                enabled=amp_autocast,
                dtype=torch.bfloat16,
            ):
                pred = model(data)
                pred = pred.view(-1)

            loss = criterion(pred, (data.y - task_mean) / task_std)
            loss_metric.update(loss.item(), n=pred.shape[0])
            err = pred.detach() * task_std + task_mean - data.y
            mae_metric.update(torch.mean(torch.abs(err)).item(), n=pred.shape[0])

            # # Track examples with prediction errors exceeding threshold
            # if debug_bad_example:
            #     indices = torch.where(torch.abs(err) > threshold)[0]
            #     for i in indices:
            #         worse_example.append(data.smiles[i])
            #         worse_err.append(torch.abs(err)[i].item())

            all_targets.append(data.y.cpu())
            all_preds.append((pred.detach() * task_std + task_mean).cpu())

        # Compute final metrics
        all_targets = torch.cat(all_targets, dim=0).numpy().flatten()
        all_preds = (
            torch.cat(all_preds, dim=0).to(dtype=torch.float).numpy().flatten()
        )
        r2 = r2_score(all_targets, all_preds)

    return mae_metric.avg, r2, loss_metric.avg


def train_cls_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    model_ema: Optional[ModelEma] = None,
    amp_autocast: bool = False,
    loss_scaler=None,
    clip_grad=None,
    print_freq: int = 100,
    logger=None,
):
    """
    Train a classification model for one epoch.
    
    Args:
        model: The neural network model
        criterion: The loss function
        data_loader: DataLoader for training data
        optimizer: The optimizer for updating weights
        device: The device to run on
        epoch: Current epoch number
        model_ema: Optional exponential moving average model
        amp_autocast: Whether to use automatic mixed precision
        loss_scaler: Optional loss scaler for gradient scaling
        clip_grad: Gradient clipping threshold
        print_freq: Frequency of logging
        logger: Logger instance
        
    Returns:
        Tuple of (average_loss, accuracy, auprc, auroc)
    """
    model.train()
    criterion.train()

    loss_metric = AverageMeter()

    acc_count = 0
    total = 0

    start_time = time.perf_counter()
    all_targets = []
    all_preds = []

    for step, data in enumerate(data_loader):
        data = data.to(device)
        
        with torch.autocast(
            device_type=device.type,
            enabled=amp_autocast,
            dtype=torch.bfloat16,
        ):
            pred = model(data)
            loss = criterion(pred, data.y)

        # Backward pass and optimization
        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(loss, optimizer, parameters=model.parameters())
        else:
            loss.backward()
            if clip_grad is not None:
                dispatch_clip_grad(model.parameters(), value=clip_grad, mode="norm")
            optimizer.step()

        # Update metrics
        loss_metric.update(loss.item(), n=pred.shape[0])
        acc_count += (pred.detach().cpu().argmax(dim=1) == data.y.cpu()).sum()
        total += pred.shape[0]

        all_targets.append(data.y.cpu())
        all_preds.append(torch.softmax(pred, dim=-1).detach().cpu())

        # Update EMA model if provided
        if model_ema is not None:
            model_ema.update(model)

        torch.cuda.synchronize()

        # Logging
        if step % print_freq == 0 or step == len(data_loader) - 1:
            elapsed_time = time.perf_counter() - start_time
            progress = (step + 1) / len(data_loader)
            time_per_step = 1e3 * elapsed_time / progress / len(data_loader)
            
            info_str = (
                f"Epoch: [{epoch}][{step}/{len(data_loader)}] "
                f"loss: {loss_metric.avg:.5f}, "
                f"Acc: {acc_count / total:.5f}, "
                f"time/step={time_per_step:.0f}ms, "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )
            logger.info(info_str)

    # Compute final metrics
    all_targets = torch.cat(all_targets, dim=0)
    all_preds = torch.cat(all_preds, dim=0).to(dtype=torch.float)[:, 1]
    auprc = binary_auprc(all_preds, all_targets)
    auroc = binary_auroc(all_preds, all_targets)
    epoch_acc = acc_count / total

    return loss_metric.avg, epoch_acc, auprc, auroc


def evaluate_cls(
    model: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    criterion: torch.nn.Module,
    amp_autocast: bool = False,
    print_freq: int = 100,
    logger=None,
):
    """
    Evaluate the classification model.
    
    Args:
        model: The neural network model
        data_loader: DataLoader for evaluation data
        device: The device to run on
        criterion: The loss function
        amp_autocast: Whether to use automatic mixed precision
        print_freq: Frequency of logging
        logger: Logger instance
        
    Returns:
        Tuple of (average_loss, accuracy, auprc, auroc)
    """
    model.eval()

    loss_metric = AverageMeter()
    total = 0
    acc_count = 0

    criterion.eval()
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)

            with torch.autocast(
                device_type=device.type,
                enabled=amp_autocast,
                dtype=torch.bfloat16,
            ):
                pred = model(data)

            loss = criterion(pred, data.y)
            loss_metric.update(loss.item(), n=pred.shape[0])

            total += pred.shape[0]
            acc_count += (
                (pred.detach().cpu().argmax(dim=-1) == data.y.cpu()).sum()
            )

            all_targets.append(data.y.cpu())
            all_preds.append(torch.softmax(pred, dim=-1).detach().cpu())

        # Compute final metrics
        all_targets = torch.cat(all_targets, dim=0)
        all_preds = torch.cat(all_preds, dim=0).to(dtype=torch.float)[:, 1]
        auprc = binary_auprc(all_preds, all_targets)
        auroc = binary_auroc(all_preds, all_targets)
        acc = acc_count / total

    return loss_metric.avg, acc, auprc, auroc


def compute_stats(
    data_loader: Iterable,
    max_radius: float,
    logger,
    print_freq: int = 1000,
):
    """
    Compute statistics (mean nodes, edges, degrees) for the dataset.
    
    Args:
        data_loader: DataLoader for the dataset
        max_radius: Maximum radius for radius graph construction
        logger: Logger instance
        print_freq: Frequency of logging statistics
    """
    log_str = f"\nCalculating statistics with max_radius={max_radius}\n"
    logger.info(log_str)

    avg_node = AverageMeter()
    avg_edge = AverageMeter()
    avg_degree = AverageMeter()

    for step, data in enumerate(data_loader):
        pos = data.pos
        batch = data.batch
        
        # Construct radius graph
        edge_src, edge_dst = radius_graph(
            pos, r=max_radius, batch=batch, max_num_neighbors=1000
        )
        
        batch_size = float(batch.max() + 1)
        num_nodes = pos.shape[0]
        num_edges = edge_src.shape[0]
        num_degree = torch_geometric.utils.degree(edge_src, num_nodes)
        num_degree = torch.sum(num_degree)

        avg_node.update(num_nodes / batch_size, batch_size)
        avg_edge.update(num_edges / batch_size, batch_size)
        avg_degree.update(num_degree / num_nodes, num_nodes)

        if step % print_freq == 0 or step == len(data_loader) - 1:
            log_str = (
                f"[{step}/{len(data_loader)}]\t"
                f"avg node: {avg_node.avg:.2f}, "
                f"avg edge: {avg_edge.avg:.2f}, "
                f"avg degree: {avg_degree.avg:.2f}"
            )
            logger.info(log_str)