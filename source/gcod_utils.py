import logging
import sys # For sys.stdout
import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import ModelConfig
    from .model import EdgeVGAE
    from torch_geometric.loader import DataLoader

def setup_gcod_logger(log_file_path: str):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] GCOD_FT: %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='a'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"GCOD Fine-tuning logging re-initialized. Log file: {log_file_path}")


def run_gcod_finetune_epoch(
    model: 'EdgeVGAE',
    train_loader_indexed: 'DataLoader',
    u_params: torch.nn.Parameter,
    optimizer_model: torch.optim.Optimizer,
    optimizer_u: torch.optim.Optimizer,
    gcod_config: 'ModelConfig',
    device: str
):
    model.train()
    total_loss_gcod_combined = 0.0
    total_loss_l1_val_weighted, total_loss_l2_val_weighted, total_loss_l3_val_weighted = 0.0, 0.0, 0.0
    num_samples_processed = 0

    criterion_ce = torch.nn.CrossEntropyLoss()
    atrain = gcod_config.gcod_fixed_atrain 

    for data_batch, indices_batch in train_loader_indexed:
        data_batch = data_batch.to(device)
        indices_batch = indices_batch.to(device)

        u_batch = u_params[indices_batch]

        z_nodes, mu_nodes, logvar_nodes, class_logits = model(
            data_batch.x, data_batch.edge_index, data_batch.edge_attr, data_batch.batch, eps=1.0
        )

        yB_one_hot = F.one_hot(data_batch.y, num_classes=gcod_config.num_classes).float()
        u_expanded = u_batch.unsqueeze(1) 

        modified_logits_L1 = class_logits + atrain * u_expanded * yB_one_hot
        loss_L1_val = criterion_ce(modified_logits_L1, data_batch.y)

        
        u_batch_clipped_L3 = torch.clamp(u_batch, 1e-6, 1.0 - 1e-6) 
        with torch.no_grad(): 
             log_probs_fo_ZB_L3 = F.log_softmax(class_logits.detach(), dim=1)
             true_class_log_probs_L3 = log_probs_fo_ZB_L3.gather(1, data_batch.y.unsqueeze(1)).squeeze(1)
        p_model_L3 = torch.exp(true_class_log_probs_L3)

        q_target_L3 = torch.sigmoid(-torch.log(u_batch_clipped_L3)) 
        
        p_m_cl_L3 = torch.clamp(p_model_L3, 1e-6, 1.0 - 1e-6) 
        q_t_cl_L3 = torch.clamp(q_target_L3, 1e-6, 1.0 - 1e-6)

        term1_L3 = p_m_cl_L3 * (torch.log(p_m_cl_L3) - torch.log(q_t_cl_L3))
        term2_L3 = (1.0 - p_m_cl_L3) * (torch.log(1.0 - p_m_cl_L3) - torch.log(1.0 - q_t_cl_L3))
        kl_div_L3_persample = term1_L3 + term2_L3
        loss_L3_val = (1.0 - atrain) * kl_div_L3_persample.mean() 

        loss_model_params = (gcod_config.gcod_l1_weight * loss_L1_val +
                             gcod_config.gcod_l3_weight * loss_L3_val)
        
        optimizer_model.zero_grad()
        loss_model_params.backward(retain_graph=True) 
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer_model.step()

        with torch.no_grad(): 
            y_hat_B_idx = torch.argmax(class_logits.detach(), dim=1)
            y_hat_B_one_hot = F.one_hot(y_hat_B_idx, num_classes=gcod_config.num_classes).float()
        
        error_vec_L2 = y_hat_B_one_hot - yB_one_hot 
        term_L2_combined = error_vec_L2 + u_expanded * yB_one_hot
        loss_L2_val = term_L2_combined.pow(2).sum(dim=1).mean() / gcod_config.num_classes 

        loss_u_params = gcod_config.gcod_l2_weight * loss_L2_val
        optimizer_u.zero_grad()
        loss_u_params.backward() 
        optimizer_u.step()

        with torch.no_grad(): 
            u_params.data.clamp_(0.0, 1.0)

        current_batch_size = data_batch.y.size(0)
        total_loss_l1_val_weighted += loss_L1_val.item() * gcod_config.gcod_l1_weight * current_batch_size
        total_loss_l2_val_weighted += loss_L2_val.item() * gcod_config.gcod_l2_weight * current_batch_size
        total_loss_l3_val_weighted += loss_L3_val.item() * gcod_config.gcod_l3_weight * current_batch_size
        total_loss_gcod_combined += (loss_model_params.item() + loss_u_params.item()) * current_batch_size
        num_samples_processed += current_batch_size

    avg_loss_gcod_combined = total_loss_gcod_combined / num_samples_processed if num_samples_processed > 0 else 0
    avg_l1_weighted = total_loss_l1_val_weighted / num_samples_processed if num_samples_processed > 0 else 0
    avg_l2_weighted = total_loss_l2_val_weighted / num_samples_processed if num_samples_processed > 0 else 0
    avg_l3_weighted = total_loss_l3_val_weighted / num_samples_processed if num_samples_processed > 0 else 0

    logging.info(f"GCOD Epoch Avg Losses: Combined={avg_loss_gcod_combined:.4f} | "
                 f"Weighted (L1_model: {avg_l1_weighted:.4f}, L2_u: {avg_l2_weighted:.4f}, L3_model: {avg_l3_weighted:.4f})")
    
    return avg_loss_gcod_combined