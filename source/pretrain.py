import os
import logging
import torch
import dataclasses
from typing import Optional, Dict, Any
from torch_geometric.loader import DataLoader

from .config import ModelConfig
from .model import EdgeVGAE
from .trainer import ModelTrainer
from .utils import warm_up_lr
from .data_utils import load_dataset

def execute_pretrain_phase(
    dataset_char: str,
    input_model_path: Optional[str],
    output_model_path_suffix: str,
    global_hyperparams: Dict[str, Any],
    dataset_specific_seeds: Dict[str, int],
    base_data_path: str,
    base_output_path: str,
    device: str
) -> Optional[str]:
    
    phase_output_tag = f"pretrain_phase_{dataset_char}" 

    checkpoints_dir = os.path.join(base_output_path, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    output_model_filename = f"pretrained{output_model_path_suffix}.pth" 
    full_output_model_path = os.path.join(checkpoints_dir, output_model_filename)

    logging.info(f"\n--- Starting Pretrain Phase for Dataset: {dataset_char} (Output Tag: {phase_output_tag}) ---")
    logging.info(f"Input model for this phase: {input_model_path if input_model_path else 'None (starting fresh)'}")
    logging.info(f"Target output model for this phase: {full_output_model_path}")

    model = EdgeVGAE(
        input_dim=1, edge_dim=7,
        hidden_dim=global_hyperparams["hidden_dim"],
        latent_dim=global_hyperparams["latent_dim"],
        num_classes=global_hyperparams["num_classes"]
    ).to(device)

    if input_model_path and os.path.exists(input_model_path):
        try:
            checkpoint = torch.load(input_model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            logging.info(f"Successfully loaded model weights from {input_model_path} for pretrain phase {dataset_char}")
        except Exception as e:
            logging.error(f"Error loading model from {input_model_path}: {e}. Starting with fresh weights for pretrain phase {dataset_char}.")
            model.init_weights()
    else:
        if input_model_path:
             logging.warning(f"Input model path '{input_model_path}' not found. Starting with fresh weights for pretrain phase {dataset_char}.")

    optimizer = torch.optim.Adam(model.parameters(), lr=global_hyperparams["learning_rate"])
    
    current_train_file_path = os.path.join(base_data_path, dataset_char, "train.json.gz")

    if not os.path.exists(current_train_file_path):
        logging.error(f"ERROR: Train path not found for pretraining on {dataset_char}: {current_train_file_path}. Skipping this pretrain phase.")
        return None

    df_current_dataset = load_dataset(current_train_file_path)
    if df_current_dataset.empty:
        logging.warning(f"WARNING: Data for pretraining on dataset {dataset_char} is empty. Skipping this pretrain phase.")
        return None

    pretrain_phase_specific_config = ModelConfig(
        train_path=current_train_file_path,
        output_tag=phase_output_tag,
        batch_size=global_hyperparams["batch_size"],
        hidden_dim=global_hyperparams["hidden_dim"],
        latent_dim=global_hyperparams["latent_dim"],
        num_classes=global_hyperparams["num_classes"],
        epochs=global_hyperparams["epochs"],
        learning_rate=global_hyperparams["learning_rate"],
        warmup=global_hyperparams["warmup"],
        early_stopping_patience=global_hyperparams["early_stopping_patience"],
        label_smoothing_epsilon=global_hyperparams.get("label_smoothing_epsilon", 0.0)
    )

    trainer_for_pretrain_utils = ModelTrainer(pretrain_phase_specific_config, device, base_output_path)
    
    split_seed = dataset_specific_seeds.get(dataset_char, dataset_specific_seeds['A'])
    logging.info(f"Preparing data split for pretraining on {dataset_char} using seed: {split_seed}")
    pretrain_train_data, pretrain_val_data = trainer_for_pretrain_utils.prepare_data_split(df_current_dataset, seed=split_seed)

    if not pretrain_train_data:
        logging.error(f"Pretrain training data for {dataset_char} is empty after split. Skipping this phase.")
        return None

    train_loader = DataLoader(pretrain_train_data, batch_size=pretrain_phase_specific_config.batch_size, shuffle=True)
    val_loader = DataLoader(pretrain_val_data, batch_size=pretrain_phase_specific_config.batch_size, shuffle=False) if pretrain_val_data else None

    best_val_f1_this_phase = 0.0
    epoch_best_this_phase = 0
    last_avg_train_loss = float('inf')
    last_val_f1_epoch = -1.0
    last_val_loss_epoch = float('inf')

    if not val_loader:
        logging.warning(f"No validation data for pretraining on {dataset_char}. Training for all {pretrain_phase_specific_config.epochs} epochs.")
        for epoch in range(pretrain_phase_specific_config.epochs):
            if epoch < pretrain_phase_specific_config.warmup:
                warm_up_lr(epoch, pretrain_phase_specific_config.warmup, pretrain_phase_specific_config.learning_rate, optimizer)
            
            last_avg_train_loss = trainer_for_pretrain_utils.train_epoch(model, train_loader, optimizer)
            
            if (epoch + 1) % 10 == 0 or epoch == pretrain_phase_specific_config.epochs - 1:
                current_lr = optimizer.param_groups[0]['lr']
                logging.info(f'Pretrain Phase {dataset_char}, Epoch {epoch+1}/{pretrain_phase_specific_config.epochs}, '
                             f'LR {current_lr:.2e}, TrainLoss: {last_avg_train_loss:.4f} (No Validation)')
        torch.save({
            'model_state_dict': model.state_dict(), 'epoch': pretrain_phase_specific_config.epochs,
            'train_loss_final': last_avg_train_loss,
            'val_f1': -1.0,
            'config_params': dataclasses.asdict(pretrain_phase_specific_config)
        }, full_output_model_path)
        logging.info(f"Pretrained model for {dataset_char} (trained for all epochs, no validation) saved to {os.path.basename(full_output_model_path)}")
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.7, patience=10, min_lr=1e-6, verbose=False)
        for epoch in range(pretrain_phase_specific_config.epochs):
            if epoch < pretrain_phase_specific_config.warmup:
                warm_up_lr(epoch, pretrain_phase_specific_config.warmup, pretrain_phase_specific_config.learning_rate, optimizer)
            
            last_avg_train_loss = trainer_for_pretrain_utils.train_epoch(model, train_loader, optimizer)
            val_metrics = trainer_for_pretrain_utils.evaluate_model(model, val_loader)
            last_val_loss_epoch, last_val_f1_epoch = val_metrics['cross_entropy_loss'], val_metrics['f1_score']
            
            if (epoch + 1) % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                logging.info(f'Pretrain Phase {dataset_char}, Epoch {epoch+1}, LR {current_lr:.2e}, '
                             f'TrainLoss: {last_avg_train_loss:.4f}, ValLoss: {last_val_loss_epoch:.4f}, ValF1: {last_val_f1_epoch:.4f}')

            if epoch >= pretrain_phase_specific_config.warmup: scheduler.step(last_val_f1_epoch)
            
            if last_val_f1_epoch > best_val_f1_this_phase:
                best_val_f1_this_phase = last_val_f1_epoch
                epoch_best_this_phase = epoch
                torch.save({
                    'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch + 1, 'val_f1': last_val_f1_epoch, 'val_loss': last_val_loss_epoch,
                    'train_loss': last_avg_train_loss, 
                    'config_params': dataclasses.asdict(pretrain_phase_specific_config)
                }, full_output_model_path)
                logging.info(f"New best model for pretrain phase {dataset_char} saved to {os.path.basename(full_output_model_path)} "
                             f"(Epoch: {epoch+1}, ValF1: {last_val_f1_epoch:.4f}, ValLoss: {last_val_loss_epoch:.4f})")
            
            patience_counter = epoch - epoch_best_this_phase
            if patience_counter > pretrain_phase_specific_config.early_stopping_patience:
                logging.info(f"Early stopping at epoch {epoch+1} for pretrain phase {dataset_char}. "
                             f"Best Val F1: {best_val_f1_this_phase:.4f} at epoch {epoch_best_this_phase+1}.")
                break
        
        if not os.path.exists(full_output_model_path):
             logging.warning(f"No best model was saved to {full_output_model_path} during pretraining for {dataset_char} "
                             f"(e.g., F1 never improved or ES before save). Saving current model state at epoch {epoch+1} as a fallback.")
             torch.save({
                'model_state_dict': model.state_dict(), 'epoch': epoch + 1, 
                'final_train_loss': last_avg_train_loss,
                'val_f1': last_val_f1_epoch,
                'val_loss': last_val_loss_epoch,
                'config_params': dataclasses.asdict(pretrain_phase_specific_config)
                }, full_output_model_path)

    logging.info(f"--- Finished Pretrain Phase for Dataset: {dataset_char} ---")
    return full_output_model_path if os.path.exists(full_output_model_path) else None