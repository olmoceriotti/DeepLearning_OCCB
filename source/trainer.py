import dataclasses
import os
import logging
from datetime import datetime
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from typing import List, Dict, Any, Tuple, Optional

from .config import ModelConfig, DATASET_SPECIFIC_SEEDS
from .model import EdgeVGAE
from .utils import warm_up_lr, set_seed, predict as util_predict
from .data_utils import create_dataset_from_dataframe

class ModelTrainer:
    def __init__(self, config: ModelConfig, device: str, base_output_path: str):
        self.config = config
        self.device = device
        self.base_output_path = base_output_path
        self.models: List[str] = []
        self.pretrain_models: List[str] = []
        
        self.setup_directories()
        self.setup_logging()

        self.criterion = torch.nn.CrossEntropyLoss()
        if self.config.label_smoothing_epsilon > 0.0:
            logging.info(f"Using CrossEntropyLoss with label smoothing: {self.config.label_smoothing_epsilon}")
            self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing_epsilon)

    def setup_directories(self):
        os.makedirs(self.base_output_path, exist_ok=True)
        directories = ['checkpoints', 'submission', 'logs']
        for directory in directories:
            dir_path = os.path.join(self.base_output_path, directory)
            os.makedirs(dir_path, exist_ok=True)

    def setup_logging(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.join(self.base_output_path, 'logs')
        log_filename = os.path.join(log_dir, f'training_{self.config.folder_name}_{timestamp}.log')

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        logging.info(f"Logging initialized for trainer of '{self.config.folder_name}'. Log file: {log_filename}")

    def evaluate_model(self, model: EdgeVGAE, data_loader: DataLoader) -> Dict[str, float]:
        model.eval()
        total_loss, total_samples = 0.0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device)
                _, _, _, class_logits = model(data.x, data.edge_index, data.edge_attr, data.batch, eps=0.0)
                loss = self.criterion(class_logits, data.y)
                
                all_preds.extend(class_logits.argmax(dim=1).cpu().numpy())
                all_labels.extend(data.y.cpu().numpy())
                
                batch_size = data.y.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        return {'cross_entropy_loss': avg_loss, 'f1_score': f1, 'num_samples': total_samples}

    def load_pretrained(self):
        self.pretrain_models = []
        if self.config.pretrain_paths:
            if os.path.isfile(self.config.pretrain_paths) and self.config.pretrain_paths.endswith('.pth'):
                self.pretrain_models = [self.config.pretrain_paths]
                logging.info(f"Loaded single pretrained model path: {self.config.pretrain_paths}")
            elif os.path.isfile(self.config.pretrain_paths) and self.config.pretrain_paths.endswith('.txt'):
                try:
                    with open(self.config.pretrain_paths, 'r') as f:
                        model_paths = [line.strip() for line in f.readlines() if line.strip() and os.path.exists(line.strip())]
                    self.pretrain_models = model_paths
                    logging.info(f"Loaded {len(self.pretrain_models)} pretrained model paths from {self.config.pretrain_paths}")
                except FileNotFoundError:
                    logging.warning(f"Pretrain model list file '{self.config.pretrain_paths}' not found.")
            else:
                logging.warning(f"Pretrain path '{self.config.pretrain_paths}' is not a valid .pth or .txt file. No models loaded.")
        else:
            logging.info("No pretrain_paths configured. Starting from scratch or using model from previous iterative step if applicable.")


    def train_single_cycle(self, cycle_num: int, train_data: List[Data], val_data: Optional[List[Data]]):
        logging.info(f"\nStarting fine-tuning cycle {cycle_num} for {self.config.folder_name}")
        model = EdgeVGAE(1, 7, self.config.hidden_dim, self.config.latent_dim, self.config.num_classes).to(self.device)

        if self.pretrain_models:
            num_pt_models = len(self.pretrain_models)
            model_file_to_load = self.pretrain_models[(cycle_num - 1) % num_pt_models] 
            try:
                logging.info(f"Attempting to load pretrained model: {model_file_to_load} for cycle {cycle_num}")
                model_data = torch.load(model_file_to_load, map_location=self.device, weights_only=False)
                model.load_state_dict(model_data['model_state_dict'])
                logging.info(f"Successfully loaded pretrained model: {model_file_to_load}")
            except Exception as e:
                logging.error(f"Error loading pretrained model {model_file_to_load}: {e}. Starting fresh for this cycle.")
                model.init_weights()
        else:
            logging.info(f"No pretrained models specified or loaded for cycle {cycle_num}. Initializing model from scratch.")

        train_loader = DataLoader(train_data, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.config.batch_size, shuffle=False) if val_data else None
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        
        best_val_f1, best_val_loss_at_best_f1, epoch_best, best_model_path = 0.0, float('inf'), 0, None
        checkpoints_dir = os.path.join(self.base_output_path, "checkpoints")

        scheduler = None
        if val_loader:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.7, patience=10, min_lr=1e-6, verbose=False)
        else:
            logging.warning(f"No validation data for cycle {cycle_num} of {self.config.folder_name}. Training for all configured epochs.")

        for epoch in range(self.config.epochs):
            if epoch < self.config.warmup:
                warm_up_lr(epoch, self.config.warmup, self.config.learning_rate, optimizer)

            model.train()
            train_loss = self.train_epoch(model, train_loader, optimizer)

            current_lr = optimizer.param_groups[0]['lr']
            if val_loader:
                val_metrics = self.evaluate_model(model, val_loader)
                val_loss_epoch, val_f1_epoch = val_metrics['cross_entropy_loss'], val_metrics['f1_score']

                if (epoch + 1) % 10 == 0:
                    logging.info(f'Cyc {cycle_num}, Ep {epoch+1}, LR {current_lr:.2e}, TrLoss: {train_loss:.4f}, ValLoss: {val_loss_epoch:.4f}, ValF1: {val_f1_epoch:.4f}')

                if epoch >= self.config.warmup and scheduler: scheduler.step(val_f1_epoch)

                if val_f1_epoch > best_val_f1:
                    best_val_f1 = val_f1_epoch
                    best_val_loss_at_best_f1 = val_loss_epoch
                    epoch_best = epoch
                    
                    if best_model_path and os.path.exists(best_model_path):
                        try: os.remove(best_model_path)
                        except OSError as e: logging.warning(f"Could not remove old best model {best_model_path}: {e}")

                    best_model_filename = f"model_{self.config.output_tag}_cycle_{cycle_num}_epoch_{epoch+1}_loss_{val_loss_epoch:.3f}_f1_{val_f1_epoch:.3f}.pth"
                    best_model_path = os.path.join(checkpoints_dir, best_model_filename)
                    
                    torch.save({
                        'model_state_dict': model.state_dict(), 
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch + 1, 
                        'val_loss': val_loss_epoch, 'val_f1': val_f1_epoch, 
                        'train_loss': train_loss,
                        'config': dataclasses.asdict(self.config)
                    }, best_model_path)
                    logging.info(f"New best model saved: {os.path.basename(best_model_path)} (F1: {val_f1_epoch:.4f}, Loss: {val_loss_epoch:.4f})")
                
                patience_counter = epoch - epoch_best
                if patience_counter > self.config.early_stopping_patience // 2 and epoch % 10 == 0 and best_model_path and os.path.exists(best_model_path):
                    logging.info(f"Patience {patience_counter}/{self.config.early_stopping_patience}. Reloading best model: {os.path.basename(best_model_path)}")
                    model.load_state_dict(torch.load(best_model_path, map_location=self.device)['model_state_dict'])

                if patience_counter > self.config.early_stopping_patience:
                    logging.info(f"Early stopping at epoch {epoch+1}. Best F1: {best_val_f1:.4f} at epoch {epoch_best+1}.")
                    break
            else:
                if (epoch + 1) % 10 == 0 or epoch == self.config.epochs -1 :
                    logging.info(f'Cyc {cycle_num}, Ep {epoch+1}/{self.config.epochs}, LR {current_lr:.2e}, TrLoss: {train_loss:.4f} (No validation set)')
        
        if not val_loader:
            final_model_filename = f"model_{self.config.output_tag}_cycle_{cycle_num}_epoch_{self.config.epochs}_NOVAL.pth"
            best_model_path = os.path.join(checkpoints_dir, final_model_filename)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': self.config.epochs,
                'train_loss': train_loss,
                'config': dataclasses.asdict(self.config)
            }, best_model_path)
            logging.info(f"Model after full epochs (no validation) saved to: {os.path.basename(best_model_path)}")
            best_val_f1 = -1.0 
            best_val_loss_at_best_f1 = train_loss 

        if best_model_path and os.path.exists(best_model_path):
            self.models.append(best_model_path)
        else:
            logging.warning(f"No model was saved for cycle {cycle_num} of {self.config.folder_name}. This might happen if validation F1 never improved.")
        
        return best_val_loss_at_best_f1, best_val_f1, best_model_path


    def train_epoch(self, model: EdgeVGAE, train_loader: DataLoader, optimizer: torch.optim.Optimizer):
        model.train()
        total_loss, total_samples = 0.0, 0
        for data in train_loader:
            data = data.to(self.device)
            optimizer.zero_grad()
            
            z_nodes, mu_nodes, logvar_nodes, class_logits = model(data.x, data.edge_index, data.edge_attr, data.batch, eps=1.0)
            
            loss = (0.15 * model.recon_loss(z_nodes, data.edge_index, data.edge_attr) +
                    0.1 * model.kl_loss(mu_nodes, logvar_nodes) +
                    self.criterion(class_logits, data.y) + 
                    0.1 * model.denoise_recon_loss(z_nodes, data.edge_index))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            batch_size = data.y.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
        return total_loss / total_samples if total_samples > 0 else 0

    def train_multiple_cycles(self, df_train_val: pd.DataFrame, num_cycles: int):
        self.models = []
        self.load_pretrained()

        all_cycle_results = []
        
        dataset_char_for_split = self.config.folder_name 
        if dataset_char_for_split.startswith("finetune_") and len(dataset_char_for_split.split("_")[-1]) == 1:
            dataset_char_for_split = dataset_char_for_split.split("_")[-1] 
        elif len(dataset_char_for_split) > 1 and not dataset_char_for_split.startswith("pretrain_phase_"):
             logging.warning(f"Cannot extract a single dataset character (A,B,C,D) from ModelConfig.folder_name ('{self.config.folder_name}') for data splitting seed. Defaulting to seed for 'A'.")
             dataset_char_for_split = "A"

        data_split_seed = DATASET_SPECIFIC_SEEDS.get(dataset_char_for_split, DATASET_SPECIFIC_SEEDS['A'])
        if dataset_char_for_split not in DATASET_SPECIFIC_SEEDS:
            logging.error(f"Dataset character '{dataset_char_for_split}' (from folder_name '{self.config.folder_name}') not in DATASET_SPECIFIC_SEEDS. Using default seed {data_split_seed}.")

        logging.info(f"Preparing data split for fine-tuning '{self.config.folder_name}' (dataset char '{dataset_char_for_split}') using seed: {data_split_seed}")
        train_list_data, val_list_data = self.prepare_data_split(df_train_val, seed=data_split_seed)

        if not train_list_data:
            logging.error(f"Training data for {self.config.folder_name} is empty after split. Skipping fine-tuning cycles.")
            return []

        for i_cycle in range(num_cycles):
            current_cycle_num = i_cycle + 1
            cycle_specific_training_seed = data_split_seed + current_cycle_num
            set_seed(cycle_specific_training_seed) 
            
            logging.info(f"\nStarting Fine-Tuning Cycle {current_cycle_num}/{num_cycles}, training_seed {cycle_specific_training_seed} for {self.config.folder_name} (Data split seed was {data_split_seed})")
            
            val_loss, val_f1, model_path = self.train_single_cycle(current_cycle_num, train_list_data, val_list_data)
            
            all_cycle_results.append({'cycle': current_cycle_num, 'seed': cycle_specific_training_seed, 'val_loss': val_loss, 'val_f1': val_f1, 'model_path': model_path})
            if model_path:
                logging.info(f"Cycle {current_cycle_num} ended. ValLoss: {val_loss:.4f}, ValF1: {val_f1:.4f}. Best Model: {os.path.basename(model_path)}")
            else:
                 logging.info(f"Cycle {current_cycle_num} ended. ValLoss: {val_loss:.4f}, ValF1: {val_f1:.4f}. No model path returned (e.g. F1 never improved).")

        model_paths_filename = f"model_paths_{self.config.folder_name}.txt" 
        model_paths_output_file = os.path.join(self.base_output_path, model_paths_filename) 
        
        os.makedirs(os.path.dirname(model_paths_output_file), exist_ok=True)
        with open(model_paths_output_file, 'w') as f:
            for path in self.models:
                if path and os.path.exists(path):
                    f.write(f"{path}\n")
        logging.info(f"Saved {len(self.models)} model paths to {model_paths_output_file}")
        return all_cycle_results

    def prepare_data_split(self, df: pd.DataFrame, seed: int = 1, test_size: float = 0.2) -> Tuple[List[Data], List[Data]]:
        unique_dbs = df['db'].unique() if 'db' in df.columns else [None]
        df_train_parts, df_val_parts = [], []
        
        has_y_for_stratify = 'y' in df.columns and \
                             df['y'].apply(lambda r: isinstance(r, list) and len(r) > 0 and \
                                                     isinstance(r[0], list) and len(r[0]) > 0).all()

        current_test_size_for_split = test_size

        if len(unique_dbs) > 1 and unique_dbs[0] is not None :
            logging.info(f"Multiple 'db' tags found: {unique_dbs}. Splitting and stratifying per 'db'.")
            for db_tag in unique_dbs:
                df_s = df[df['db'] == db_tag]
                slen = len(df_s)
                if slen < 2 :
                    df_train_parts.append(df_s)
                    logging.warning(f"Stratum '{db_tag}' has {slen} samples, too few to split. Added all to train.")
                    continue

                current_test_size_for_split = test_size if slen * test_size >= 1 else (1 / slen if slen >=2 else 0)
                if current_test_size_for_split == 0 and slen == 1:
                     df_train_parts.append(df_s); continue

                stratify_array = None
                if has_y_for_stratify:
                    try:
                        s_col = df_s['y'].apply(lambda x: x[0][0])
                        if s_col.nunique() > 1: stratify_array = s_col
                        else: logging.info(f"Stratum '{db_tag}': Not enough unique classes ({s_col.nunique()}) for stratification.")
                    except Exception as e: logging.warning(f"Stratification error for db '{db_tag}': {e}. Proceeding without stratification for this db.")
                
                if current_test_size_for_split > 0 :
                    t, v = train_test_split(df_s, test_size=current_test_size_for_split, shuffle=True, random_state=seed, stratify=stratify_array)
                    df_train_parts.append(t)
                    df_val_parts.append(v)
                else:
                    df_train_parts.append(df_s)

        else:
            logging.info("Single 'db' tag or no 'db' column. Splitting the entire dataset.")
            dflen = len(df)
            if dflen < 2:
                df_train, df_val = df, pd.DataFrame(columns=df.columns)
                logging.warning(f"Dataset has {dflen} samples, too few to split. Using all for train.")
            else:
                current_test_size_for_split = test_size if dflen * test_size >= 1 else (1 / dflen if dflen >= 2 else 0)
                stratify_array = None
                if has_y_for_stratify:
                    try:
                        s_col = df['y'].apply(lambda x: x[0][0])
                        if s_col.nunique() > 1: stratify_array = s_col
                        else: logging.info(f"Entire dataset: Not enough unique classes ({s_col.nunique()}) for stratification.")
                    except Exception as e: logging.warning(f"Stratification error for entire dataset: {e}. Proceeding without stratification.")
                
                if current_test_size_for_split > 0 :
                    df_train, df_val = train_test_split(df, test_size=current_test_size_for_split, shuffle=True, random_state=seed, stratify=stratify_array)
                else: 
                    df_train, df_val = df, pd.DataFrame(columns=df.columns)


        if df_train_parts:
            df_train_final = pd.concat(df_train_parts, ignore_index=True) if df_train_parts else pd.DataFrame()
            df_val_final = pd.concat(df_val_parts, ignore_index=True) if df_val_parts else pd.DataFrame()
        elif 'df_train' in locals():
            df_train_final = df_train
            df_val_final = df_val
        else:
            df_train_final, df_val_final = df, pd.DataFrame(columns=df.columns)


        train_pyg_list = create_dataset_from_dataframe(df_train_final, True) if not df_train_final.empty else []
        val_pyg_list = create_dataset_from_dataframe(df_val_final, True) if not df_val_final.empty else []
        
        logging.info(f"Data split with seed {seed}: Train size: {len(train_pyg_list)}, Validation size: {len(val_pyg_list)}")
        if not val_pyg_list and len(df) >=2 and current_test_size_for_split > 0:
            logging.warning(f"Validation set is empty after split (seed {seed}, original df size {len(df)}, effective test_size {current_test_size_for_split}). This might happen with small datasets or strata.")
        return train_pyg_list, val_pyg_list


    def predict_with_ensemble_score(self, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if not self.models:
            logging.error("No models available in self.models for ensemble prediction.")
            model_list_filename_stem = self.config.folder_name 
            model_list_filename = f"model_paths_{model_list_filename_stem}.txt"
            model_list_file = os.path.join(self.base_output_path, model_list_filename) 

            if os.path.exists(model_list_file):
                logging.info(f"Attempting to load models for ensemble from {model_list_file}")
                with open(model_list_file, 'r') as f:
                    loaded_paths = [line.strip() for line in f.readlines() if line.strip()]
                    self.models = [p for p in loaded_paths if os.path.exists(p)]
                if self.models:
                    logging.info(f"Loaded {len(self.models)} model paths from {model_list_file}.")
                else:
                    logging.error(f"No valid model paths found in {model_list_file} or file was empty.")
                    return np.full(len(test_df), -1, dtype=int), np.zeros(len(test_df))
            else:
                logging.error(f"Model list file {model_list_file} not found. Cannot load models for ensemble.")
                return np.full(len(test_df), -1, dtype=int), np.zeros(len(test_df))

        test_loader = DataLoader(create_dataset_from_dataframe(test_df, result=False), 
                                 batch_size=self.config.batch_size, shuffle=False)
        
        all_model_predictions_list = [] 
        model_f1_scores_for_weighting = []
        valid_models_used = [] 

        for model_path in self.models:
            if not os.path.exists(model_path):
                logging.warning(f"Model path {model_path} not found. Skipping for ensemble.")
                continue
            
            model_instance = EdgeVGAE(1, 7, self.config.hidden_dim, self.config.latent_dim, self.config.num_classes).to(self.device)
            try:
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                model_instance.load_state_dict(checkpoint['model_state_dict'])
                
                f1_for_weighting = checkpoint.get('val_f1', 0.0) 
                if f1_for_weighting == -1.0: f1_for_weighting = 0.5 

                model_f1_scores_for_weighting.append(f1_for_weighting)
                all_model_predictions_list.append(util_predict(model_instance, self.device, test_loader))
                valid_models_used.append(model_path)
                logging.info(f"Loaded and got predictions from {os.path.basename(model_path)}, F1 for weighting: {f1_for_weighting:.4f}")

            except Exception as e:
                logging.error(f"Failed to load or predict with model {model_path}: {e}")
        
        self.models = valid_models_used 

        if not all_model_predictions_list:
            logging.error("No models successfully contributed to predictions for the ensemble.")
            return np.full(len(test_df), -1, dtype=int), np.zeros(len(test_df))

        predictions_np_array = np.array(all_model_predictions_list) 
        f1_scores_np_array = np.array(model_f1_scores_for_weighting)

        weights = np.ones_like(f1_scores_np_array) / len(f1_scores_np_array) if len(f1_scores_np_array) > 0 else []
        if len(f1_scores_np_array) > 0 and np.sum(f1_scores_np_array) > 0 and len(np.unique(f1_scores_np_array)) > 1 : 
            exp_scores = np.exp((f1_scores_np_array + 1e-6) * 10) 
            weights = exp_scores / np.sum(exp_scores)
        else:
            logging.info("Using uniform weights for ensemble (F1s were uniform, zero, or only one/zero models).")
        
        if len(weights) > 0:
            logging.info("Ensemble model weights:")
            for i, model_p in enumerate(self.models):
                if i < len(weights):
                    logging.info(f"- {os.path.basename(model_p)}: F1={f1_scores_np_array[i]:.4f}, Weight={weights[i]:.4f}")
        else:
            logging.warning("No weights calculated, likely no models to ensemble.")
            return np.full(len(test_df), -1, dtype=int), np.zeros(len(test_df))


        num_samples = predictions_np_array.shape[1]
        num_classes = self.config.num_classes
        
        weighted_votes = np.zeros((num_samples, num_classes))
        for i, class_predictions_for_model_i in enumerate(predictions_np_array): 
            if i < len(weights): 
                for sample_idx, predicted_class in enumerate(class_predictions_for_model_i):
                    weighted_votes[sample_idx, predicted_class] += weights[i]
        
        ensemble_predictions = np.argmax(weighted_votes, axis=1)
        confidence_scores = np.max(weighted_votes, axis=1) 

        logging.info("\nEnsemble prediction class distribution:")
        unique_preds, counts = np.unique(ensemble_predictions, return_counts=True)
        for pred_class, count in zip(unique_preds, counts):
            logging.info(f"Class {pred_class}: {count} predictions")
        if len(confidence_scores) > 0:
             logging.info(f"Average confidence score of ensemble: {np.mean(confidence_scores):.4f}")
             
        return ensemble_predictions, confidence_scores