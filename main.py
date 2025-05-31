import argparse
import os
import logging
import sys
import re
import pandas as pd
import numpy as np
import torch
import dataclasses
from datetime import datetime
from typing import Optional, List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from source.config import ModelConfig, GLOBAL_HYPERPARAMS, DATASET_SPECIFIC_SEEDS, ALL_DATASET_CHARS
from source.data_utils import load_dataset, create_dataset_from_dataframe, IndexedDataset
from torch_geometric.loader import DataLoader
from source.model import EdgeVGAE
from source.trainer import ModelTrainer
from source.pretrain import execute_pretrain_phase
from source.gcod_utils import setup_gcod_logger, run_gcod_finetune_epoch

BASE_OUTPUT_PATH = SCRIPT_DIR 
GCOD_FINETUNE_EPOCHS_PER_MODEL = 50

def derive_dataset_char_and_base_data_path(file_path: str) -> tuple[Optional[str], Optional[str]]:
    if not file_path:
        return None, None
    try:
        abs_file_path = os.path.abspath(file_path)
        dataset_char = os.path.basename(os.path.dirname(abs_file_path))
        if dataset_char not in ALL_DATASET_CHARS:
            logging.error(f"Could not derive dataset_char (A,B,C,D) from path {file_path}. "
                          f"Detected folder name '{dataset_char}' is not one of A, B, C, D.")
            return None, None

        base_data_path = os.path.dirname(os.path.dirname(abs_file_path))
        return dataset_char, base_data_path
    except Exception as e:
        logging.error(f"Error deriving dataset_char/base_data_path from {file_path}: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description="Graph Classification Model Training and Prediction")
    parser.add_argument('--test_path', type=str, required=True,
                        help='Path to the test.json.gz file (e.g., /path/to/data/A/test.json.gz)')
    parser.add_argument('--train_path', type=str,
                        help='Optional path to the train.json.gz file for training (e.g., /path/to/data/A/train.json.gz)')
    parser.add_argument('--external_pretrained_model', type=str,
                        help='Optional path to an external .pth model file to use as base for fine-tuning.')
    parser.add_argument('--base_output_dir', type=str, default=SCRIPT_DIR,
                        help="Base directory for script outputs (logs, checkpoints, submission). "
                             "If your 'checkpoints' dir is elsewhere, set this.")


    args = parser.parse_args()

    global BASE_OUTPUT_PATH
    BASE_OUTPUT_PATH = os.path.abspath(args.base_output_dir)


    dataset_char, base_data_path_derived = derive_dataset_char_and_base_data_path(args.test_path)
    if not dataset_char or not base_data_path_derived:
        print("Critical error: Could not determine dataset character or base data path from --test_path. Exiting.")
        sys.exit(1)

    BASE_DATA_PATH = base_data_path_derived
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    for subdir in ["checkpoints", "logs", "submission", "checkpoints_gcod"]:
        os.makedirs(os.path.join(BASE_OUTPUT_PATH, subdir), exist_ok=True)

    main_log_file = os.path.join(BASE_OUTPUT_PATH, "logs", f"main_run_orchestration_{dataset_char}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] MAIN_ORCHESTRATOR: %(message)s',
        handlers=[logging.FileHandler(main_log_file), logging.StreamHandler(sys.stdout)]
    )
    logging.info(f"Main script started. Target Dataset: {dataset_char}")
    logging.info(f"Device: {DEVICE}")
    logging.info(f"Base Output Path (for logs, checkpoints, submission): {BASE_OUTPUT_PATH}")
    logging.info(f"Derived Base Data Path (for A,B,C,D train/test files): {BASE_DATA_PATH}")
    logging.info(f"Test Path: {args.test_path}")
    if args.train_path: logging.info(f"Train Path: {args.train_path}")
    if args.external_pretrained_model: logging.info(f"External Pretrained Model: {args.external_pretrained_model}")
    logging.info(f"Number of GCOD fine-tuning epochs per model: {GCOD_FINETUNE_EPOCHS_PER_MODEL}")


    base_model_for_finetuning = args.external_pretrained_model
    models_for_final_ensemble: List[str] = []


    # --- TRAINING MODE ---
    if args.train_path:
        logging.info("--- Training Mode Activated ---")

        logging.info(f"\n{'='*60}\nSTARTING ITERATIVE PRETRAINING (Sequential A->B->C->D)\n{'='*60}\n")
        previous_phase_model_path = None
        previous_phase_suffix_for_filename = ""
        final_iterative_pretrained_model_path = None

        for char_idx, pretrain_char_iter in enumerate(ALL_DATASET_CHARS):
            logging.info(f"\n--- Iterative Pretrain Sub-Phase for Dataset: {pretrain_char_iter} ---")
            current_output_suffix = f"{previous_phase_suffix_for_filename}_{pretrain_char_iter}" if previous_phase_suffix_for_filename else f"_{pretrain_char_iter}"

            _pretrain_train_file_check = os.path.join(BASE_DATA_PATH, pretrain_char_iter, "train.json.gz")
            if not os.path.exists(_pretrain_train_file_check):
                logging.warning(f"Train data for iterative pretraining on '{pretrain_char_iter}' not found at '{_pretrain_train_file_check}'. Skipping sub-phase.")
                if char_idx == len(ALL_DATASET_CHARS) - 1:
                    final_iterative_pretrained_model_path = previous_phase_model_path
                continue

            output_model_path_iter_phase = execute_pretrain_phase(
                dataset_char=pretrain_char_iter,
                input_model_path=previous_phase_model_path,
                output_model_path_suffix=current_output_suffix,
                global_hyperparams=GLOBAL_HYPERPARAMS,
                dataset_specific_seeds=DATASET_SPECIFIC_SEEDS,
                base_data_path=BASE_DATA_PATH,
                base_output_path=BASE_OUTPUT_PATH,
                device=DEVICE
            )
            if output_model_path_iter_phase and os.path.exists(output_model_path_iter_phase):
                previous_phase_model_path = output_model_path_iter_phase
                previous_phase_suffix_for_filename = current_output_suffix
                final_iterative_pretrained_model_path = output_model_path_iter_phase
                logging.info(f"Iterative Pretrain Sub-Phase {pretrain_char_iter} complete. Output model: {os.path.basename(output_model_path_iter_phase)}")
            else:
                logging.warning(f"Iterative Pretrain Sub-Phase {pretrain_char_iter} did not produce a model. Using previous phase's model if available for next.")
                if char_idx == len(ALL_DATASET_CHARS) - 1:
                    final_iterative_pretrained_model_path = previous_phase_model_path

        logging.info(f"Iterative pretraining complete. Final model: {final_iterative_pretrained_model_path if final_iterative_pretrained_model_path else 'None (or error in last phase)'}")

        if not base_model_for_finetuning:
            if final_iterative_pretrained_model_path and os.path.exists(final_iterative_pretrained_model_path):
                base_model_for_finetuning = final_iterative_pretrained_model_path
                logging.info(f"Using this script's iteratively pretrained model for fine-tuning: {base_model_for_finetuning}")
            else:
                logging.warning("No --external_pretrained_model and no iterative pretrain model. Fine-tuning will start from random weights.")
        else:
             logging.info(f"Using EXTERNALLY PROVIDED pretrained model for fine-tuning: {base_model_for_finetuning}")

        if base_model_for_finetuning and not os.path.exists(base_model_for_finetuning):
            logging.error(f"Base model for fine-tuning ('{base_model_for_finetuning}') not found. Proceeding with random weights.")
            base_model_for_finetuning = None

        logging.info(f"\n{'='*60}\nSTARTING STANDARD FINE-TUNING STAGE: DATASET {dataset_char}\n{'='*60}\n")
        finetuning_cycles_per_dataset = 3

        if not os.path.exists(args.train_path):
            logging.error(f"Train path for fine-tuning {dataset_char} ({args.train_path}) not found. Skipping standard fine-tuning.")
        else:
            ft_hyperparams_copy = GLOBAL_HYPERPARAMS.copy()
            config_standard_finetune = ModelConfig(
                train_path=args.train_path,
                test_path=args.test_path,
                pretrain_paths=base_model_for_finetuning,
                output_tag=f"finetune_{dataset_char}",
                num_cycles=finetuning_cycles_per_dataset,
                **ft_hyperparams_copy
            )
            trainer_standard_finetune = ModelTrainer(config_standard_finetune, DEVICE, BASE_OUTPUT_PATH)
            logging.info(f"Standard Fine-tuning for dataset {dataset_char}. Base model: {base_model_for_finetuning if base_model_for_finetuning else 'Random Init'}")

            try:
                df_train_val_data_for_ft = load_dataset(config_standard_finetune.train_path)
                if df_train_val_data_for_ft.empty:
                    logging.error(f"Training data for standard fine-tuning of {dataset_char} is empty. Skipping.")
                else:
                    trainer_standard_finetune.train_multiple_cycles(df_train_val_data_for_ft, finetuning_cycles_per_dataset)
            except Exception as e:
                logging.error(f"Error during standard fine-tuning for {dataset_char}: {e}", exc_info=True)

            if os.path.exists(args.test_path) and trainer_standard_finetune.models:
                logging.info(f"--- Prediction after standard fine-tuning for {dataset_char} ---")
                try:
                    df_test_data_for_ft_pred = load_dataset(args.test_path)
                    if not df_test_data_for_ft_pred.empty:
                        predictions_ft_stage, _ = trainer_standard_finetune.predict_with_ensemble_score(df_test_data_for_ft_pred)
                        if not (len(predictions_ft_stage) == len(df_test_data_for_ft_pred) and np.all(predictions_ft_stage == -1)):
                            submission_dir_ft_pred = os.path.join(BASE_OUTPUT_PATH, "submission")
                            output_csv_path_ft_pred = os.path.join(submission_dir_ft_pred, f"testset_{dataset_char}_standardFT.csv")
                            pd.DataFrame({"id": range(len(predictions_ft_stage)), "pred": predictions_ft_stage}).to_csv(output_csv_path_ft_pred, index=False)
                            logging.info(f"Saved standard fine-tuning predictions for {dataset_char} to {output_csv_path_ft_pred}")
                        else: logging.error(f"Standard FT prediction invalid for {dataset_char}.")
                    else: logging.error(f"Test data for standard FT pred for {dataset_char} empty.")
                except Exception as e: logging.error(f"Error in standard FT pred for {dataset_char}: {e}", exc_info=True)
            elif not trainer_standard_finetune.models:
                 logging.warning(f"No models from standard fine-tuning for {dataset_char}. No intermediate prediction.")


        # 3. GCOD Fine-tuning for the target dataset_char
        logging.info(f"\n{'='*60}\nSTARTING GCOD FINE-TUNING STAGE: DATASET {dataset_char}\n{'='*60}\n")

        gcod_tuned_checkpoints_dir = os.path.join(BASE_OUTPUT_PATH, "checkpoints")
        source_models_for_gcod_dir = os.path.join(BASE_OUTPUT_PATH, "checkpoints")

        gcod_run_log_file = os.path.join(BASE_OUTPUT_PATH, "logs", f"gcod_finetuning_run_{dataset_char}.log")
        setup_gcod_logger(gcod_run_log_file)

        gcod_hyperparams_dict = {
            "gcod_u_lr": 0.01, "gcod_fixed_atrain": 0.7, "gcod_l1_weight": 1.0,
            "gcod_l2_weight": 1.0, "gcod_l3_weight": 1.0, "gcod_u_init_value": 0.0,
            "learning_rate": 5e-5,
            "batch_size": GLOBAL_HYPERPARAMS.get("batch_size", 24),
            "num_classes": GLOBAL_HYPERPARAMS.get("num_classes", 6),
            "hidden_dim": GLOBAL_HYPERPARAMS.get("hidden_dim", 128),
            "latent_dim": GLOBAL_HYPERPARAMS.get("latent_dim", 8),
        }

        if not os.path.exists(args.train_path):
            logging.error(f"GCOD: Train data for {dataset_char} ({args.train_path}) not found. Skipping GCOD fine-tuning.")
        else:
            df_train_for_gcod = load_dataset(args.train_path)
            if df_train_for_gcod.empty:
                logging.error(f"GCOD: Training data for {dataset_char} loaded empty. Skipping GCOD fine-tuning.")
            else:
                pyg_train_dataset_for_gcod = create_dataset_from_dataframe(df_train_for_gcod, result=True)
                if not pyg_train_dataset_for_gcod:
                    logging.error(f"GCOD: Failed to create PyG dataset from DataFrame for {dataset_char}. Skipping GCOD fine-tuning.")
                else:
                    indexed_train_dataset_gcod = IndexedDataset(pyg_train_dataset_for_gcod)
                    train_loader_indexed_gcod = DataLoader(
                        indexed_train_dataset_gcod,
                        batch_size=gcod_hyperparams_dict["batch_size"],
                        shuffle=True
                    )
                    num_train_samples_for_gcod = len(indexed_train_dataset_gcod)

                    models_to_gcod_finetune: List[str] = []
                    if os.path.exists(source_models_for_gcod_dir):
                        pattern_for_gcod_source_models = re.compile(f"^model_finetune_{dataset_char}_.*\\.pth$", re.IGNORECASE)
                        for f_name in os.listdir(source_models_for_gcod_dir):
                            if pattern_for_gcod_source_models.search(f_name):
                                models_to_gcod_finetune.append(os.path.join(source_models_for_gcod_dir, f_name))
                        models_to_gcod_finetune = sorted(list(set(models_to_gcod_finetune)))

                    if not models_to_gcod_finetune:
                        logging.warning(f"GCOD: No standard fine-tuned models found in {source_models_for_gcod_dir} "
                                        f"matching pattern 'model_finetune_{dataset_char}_*.pth'. Skipping GCOD fine-tuning.")
                    else:
                        logging.info(f"GCOD: Found {len(models_to_gcod_finetune)} models from standard fine-tuning of {dataset_char} to process with GCOD.")

                        current_run_gcod_tuned_models: List[str] = []

                        for model_path_for_gcod_ft_input in models_to_gcod_finetune:
                            logging.info(f"\n--- GCOD Processing model: {os.path.basename(model_path_for_gcod_ft_input)} for dataset {dataset_char} ---")

                            gcod_model_instance = EdgeVGAE(
                                input_dim=1, edge_dim=7,
                                hidden_dim=gcod_hyperparams_dict["hidden_dim"],
                                latent_dim=gcod_hyperparams_dict["latent_dim"],
                                num_classes=gcod_hyperparams_dict["num_classes"]
                            ).to(DEVICE)

                            original_val_f1_from_ckpt = 0.0
                            try:
                                checkpoint_data_gcod_load = torch.load(model_path_for_gcod_ft_input, map_location=DEVICE, weights_only=False)
                                gcod_model_instance.load_state_dict(checkpoint_data_gcod_load['model_state_dict'])
                                original_val_f1_from_ckpt = checkpoint_data_gcod_load.get('val_f1', 0.0)
                                logging.info(f"GCOD: Successfully loaded weights from {os.path.basename(model_path_for_gcod_ft_input)}. Original Val F1: {original_val_f1_from_ckpt:.4f}")
                            except Exception as e:
                                logging.error(f"GCOD: Error loading model {os.path.basename(model_path_for_gcod_ft_input)}: {e}. Skipping this model.")
                                continue

                            u_params_for_gcod = torch.nn.Parameter(
                                torch.full((num_train_samples_for_gcod,), gcod_hyperparams_dict["gcod_u_init_value"], device=DEVICE, dtype=torch.float32)
                            )
                            optimizer_gcod_model = torch.optim.Adam(gcod_model_instance.parameters(), lr=gcod_hyperparams_dict["learning_rate"])
                            optimizer_gcod_u = torch.optim.Adam([u_params_for_gcod], lr=gcod_hyperparams_dict["gcod_u_lr"])

                            gcod_epoch_runner_config = ModelConfig(
                                num_classes=gcod_hyperparams_dict["num_classes"],
                                batch_size=gcod_hyperparams_dict["batch_size"],
                                hidden_dim=gcod_hyperparams_dict["hidden_dim"],
                                latent_dim=gcod_hyperparams_dict["latent_dim"],
                                use_gcod_loss=True,
                                gcod_u_lr = gcod_hyperparams_dict["gcod_u_lr"],
                                gcod_fixed_atrain = gcod_hyperparams_dict["gcod_fixed_atrain"],
                                gcod_l1_weight = gcod_hyperparams_dict["gcod_l1_weight"],
                                gcod_l2_weight = gcod_hyperparams_dict["gcod_l2_weight"],
                                gcod_l3_weight = gcod_hyperparams_dict["gcod_l3_weight"],
                                gcod_u_init_value = gcod_hyperparams_dict["gcod_u_init_value"]
                            )

                            logging.info(f"GCOD: Starting {GCOD_FINETUNE_EPOCHS_PER_MODEL}-epoch GCOD fine-tuning for {os.path.basename(model_path_for_gcod_ft_input)}...")
                            avg_gcod_epoch_loss_final = 0
                            for gcod_epoch_num in range(GCOD_FINETUNE_EPOCHS_PER_MODEL):
                                avg_gcod_epoch_loss_final = run_gcod_finetune_epoch(
                                    model=gcod_model_instance,
                                    train_loader_indexed=train_loader_indexed_gcod,
                                    u_params=u_params_for_gcod,
                                    optimizer_model=optimizer_gcod_model,
                                    optimizer_u=optimizer_gcod_u,
                                    gcod_config=gcod_epoch_runner_config,
                                    device=DEVICE
                                )
                                logging.info(f"GCOD: Fine-tuning epoch {gcod_epoch_num+1}/{GCOD_FINETUNE_EPOCHS_PER_MODEL} completed for {os.path.basename(model_path_for_gcod_ft_input)}. Avg Combined Loss: {avg_gcod_epoch_loss_final:.4f}")

                            original_model_name_no_ext = os.path.splitext(os.path.basename(model_path_for_gcod_ft_input))[0]
                            gcod_finetuned_output_name = f"{original_model_name_no_ext}_GCODft_ep{GCOD_FINETUNE_EPOCHS_PER_MODEL}.pth"
                            gcod_final_save_path = os.path.join(gcod_tuned_checkpoints_dir, gcod_finetuned_output_name)
                            try:
                                torch.save({
                                    'model_state_dict': gcod_model_instance.state_dict(),
                                    'val_f1': original_val_f1_from_ckpt,
                                    'gcod_u_params_final': u_params_for_gcod.detach().cpu(),
                                    'original_model_path': model_path_for_gcod_ft_input,
                                    'gcod_finetune_epochs_run': GCOD_FINETUNE_EPOCHS_PER_MODEL,
                                    'gcod_finetune_last_epoch_loss': avg_gcod_epoch_loss_final,
                                    'gcod_config_params_used': dataclasses.asdict(gcod_epoch_runner_config)
                                }, gcod_final_save_path)
                                logging.info(f"GCOD: Saved GCOD fine-tuned model to: {gcod_final_save_path} (Original Val F1 from source model: {original_val_f1_from_ckpt:.4f})")
                                current_run_gcod_tuned_models.append(gcod_final_save_path)
                            except Exception as e:
                                logging.error(f"GCOD: Error saving GCOD fine-tuned model {gcod_final_save_path}: {e}")

                        if current_run_gcod_tuned_models or models_to_gcod_finetune:
                            combined_from_training = list(set(current_run_gcod_tuned_models + models_to_gcod_finetune))
                            models_for_final_ensemble = sorted(combined_from_training)
                            logging.info(f"GCOD stage complete. {len(current_run_gcod_tuned_models)} GCOD models (from checkpoints_gcod) and "
                                         f"{len(models_to_gcod_finetune)} source standard FT models (from checkpoints) "
                                         f"({len(models_for_final_ensemble)} unique total) selected from this training run for potential final ensemble.")
                        else:
                            logging.info("GCOD stage complete. No new GCOD models or source standard FT models identified from this run.")


        logging.info(f"--- GCOD Fine-tuning for Dataset {dataset_char} Complete ---")


    # --- PREDICTION MODE (FINAL ENSEMBLE) ---
    logging.info(f"\n{'='*60}\nSTARTING FINAL ENSEMBLE PREDICTION: DATASET {dataset_char}\n{'='*60}\n")

    if not models_for_final_ensemble:
        logging.info("Prediction-only mode or no models populated from current training run. Discovering models for final ensemble...")

        discovered_ensemble_models_list: List[str] = []
        model_discovery_dir = os.path.join(BASE_OUTPUT_PATH, "checkpoints")

        if os.path.exists(model_discovery_dir):
            pattern_for_discovery = re.compile(
                f"^model_{dataset_char}_.*\\.pth$",
                re.IGNORECASE
            )
            logging.info(f"Searching for models in '{model_discovery_dir}' with pattern: '{pattern_for_discovery.pattern}'")

            for f_name in os.listdir(model_discovery_dir):
                if pattern_for_discovery.search(f_name):
                    discovered_ensemble_models_list.append(os.path.join(model_discovery_dir, f_name))

            if discovered_ensemble_models_list:
                 logging.info(f"Discovered {len(discovered_ensemble_models_list)} models in '{model_discovery_dir}' matching the pattern.")
            else:
                 logging.info(f"No models found in '{model_discovery_dir}' matching the pattern.")
        else:
            logging.warning(f"Model discovery directory '{model_discovery_dir}' not found. Cannot discover models.")

        script_gcod_dir = os.path.join(BASE_OUTPUT_PATH, "checkpoints_gcod")
        if os.path.exists(script_gcod_dir):
            pattern_script_gcod = re.compile(
                f"^model_finetune_{dataset_char}_.*_GCODft_ep\\d+\\.pth$",
                re.IGNORECASE
            )
            logging.info(f"Also searching for GCOD models (script's convention) in '{script_gcod_dir}' with pattern: '{pattern_script_gcod.pattern}'")
            found_script_gcod = 0
            for f_name in os.listdir(script_gcod_dir):
                if pattern_script_gcod.search(f_name):
                    discovered_ensemble_models_list.append(os.path.join(script_gcod_dir, f_name))
                    found_script_gcod +=1
            if found_script_gcod > 0:
                logging.info(f"Discovered an additional {found_script_gcod} GCOD models from '{script_gcod_dir}'.")


        models_for_final_ensemble = sorted(list(set(discovered_ensemble_models_list)))

        if models_for_final_ensemble:
            logging.info(f"Final Ensemble Pred: Using a total of {len(models_for_final_ensemble)} unique discovered models for dataset '{dataset_char}'.")
        else:
            logging.warning(f"Final Ensemble Pred: No models discovered for dataset '{dataset_char}'.")
    else:
        logging.info(f"Using {len(models_for_final_ensemble)} models selected from the current training run for final prediction.")


    if not models_for_final_ensemble:
        logging.error(f"Final Ensemble Pred: No models available for Dataset {dataset_char}. Cannot perform final prediction.")
    else:
        df_test_data_for_final_pred = None
        if os.path.exists(args.test_path):
            df_test_data_for_final_pred = load_dataset(args.test_path)
            if df_test_data_for_final_pred.empty:
                logging.error(f"Final Ensemble Pred: Test data from {args.test_path} loaded empty. Cannot proceed.")
                df_test_data_for_final_pred = None
        else:
            logging.error(f"Final Ensemble Pred: Test data file not found at {args.test_path}. Cannot predict.")

        if df_test_data_for_final_pred is not None:
            prediction_config_final_ens = ModelConfig(
                output_tag=f"prediction_final_ensemble_{dataset_char}",
                batch_size=GLOBAL_HYPERPARAMS["batch_size"],
                hidden_dim=GLOBAL_HYPERPARAMS["hidden_dim"],
                latent_dim=GLOBAL_HYPERPARAMS["latent_dim"],
                num_classes=GLOBAL_HYPERPARAMS["num_classes"],
                train_path="dummy_train.json.gz", test_path="dummy_test.json.gz"
            )

            final_ensemble_predictor_trainer = ModelTrainer(prediction_config_final_ens, DEVICE, BASE_OUTPUT_PATH)
            final_ensemble_predictor_trainer.models = models_for_final_ensemble

            logging.info(f"--- Starting Final Ensemble Prediction for dataset {dataset_char} with {len(final_ensemble_predictor_trainer.models)} models ---")
            for i, model_p in enumerate(final_ensemble_predictor_trainer.models):
                 logging.info(f"  Using Model {i+1}: {model_p}")

            try:
                final_predictions_array, _ = final_ensemble_predictor_trainer.predict_with_ensemble_score(df_test_data_for_final_pred)

                if not (len(final_predictions_array) == len(df_test_data_for_final_pred) and np.all(final_predictions_array == -1)):
                    submission_output_dir = os.path.join(BASE_OUTPUT_PATH, "submission")
                    final_output_csv_path = os.path.join(submission_output_dir, f"testset_{dataset_char}.csv")

                    pd.DataFrame({
                        "id": range(len(final_predictions_array)),
                        "pred": final_predictions_array
                    }).to_csv(final_output_csv_path, index=False)

                    logging.info(f"Final Ensemble Pred: Final predictions for {dataset_char} saved to {final_output_csv_path}")
                    print(f"SUCCESS: Final ensemble submission file for Dataset {dataset_char} saved to: {final_output_csv_path}")
                else:
                    logging.error(f"Final Ensemble Pred: Prediction failed or returned all invalid results for {dataset_char}.")
            except Exception as e:
                logging.error(f"Final Ensemble Pred: An error occurred during final ensemble prediction for {dataset_char}: {e}", exc_info=True)

    logging.info(f"--- Script execution finished for dataset {dataset_char} ---")

if __name__ == '__main__':
    main()