import random
import torch
import numpy as np
import os
import logging
from typing import Dict, Any, List, TYPE_CHECKING
from torch_geometric.loader import DataLoader

if TYPE_CHECKING:
    from .model import EdgeVGAE

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def save_checkpoint(model: torch.nn.Module, base_output_path: str, folder_name: str, cycle: int, epoch: int, metrics: Dict[str, Any]):
    checkpoints_dir = os.path.join(base_output_path, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    filename = os.path.join(checkpoints_dir, f"model_{folder_name}_cycle{cycle}_epoch{epoch}.pth")
    state_to_save = {
        'model_state_dict': model.state_dict(),
        'metrics': metrics,
        'cycle': cycle,
        'epoch': epoch
    }
    torch.save(state_to_save, filename)
    logging.info(f"Saved checkpoint (via utils.save_checkpoint): {filename}")

def warm_up_lr(epoch, num_epoch_warm_up, init_lr, optimizer):
    if num_epoch_warm_up <= 0: return
    lr = (epoch + 1)**3 * init_lr / num_epoch_warm_up**3
    for params in optimizer.param_groups:
        params['lr'] = lr

def predict(model: 'EdgeVGAE', device: str, loader: DataLoader) -> List[int]:
    model.eval()
    y_pred = []
    with torch.no_grad():
        for data_batch in loader:
            data_batch = data_batch.to(device)
            _, _, _, class_logits = model(data_batch.x, data_batch.edge_index, data_batch.edge_attr, data_batch.batch, eps=0.0)
            pred = class_logits.argmax(dim=1)
            y_pred.extend(pred.tolist())
    return y_pred