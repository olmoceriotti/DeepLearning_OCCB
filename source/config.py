import dataclasses
from typing import Optional, List, Dict, Any
import os

@dataclasses.dataclass
class ModelConfig:
    test_path: Optional[str] = None
    train_path: Optional[str] = None
    pretrain_paths: Optional[str] = None
    output_tag: Optional[str] = None
    batch_size: int = 24
    hidden_dim: int = 128
    latent_dim: int = 8
    num_classes: int = 6
    epochs: int = 1000
    learning_rate: float = 0.0005
    num_cycles: int = 3
    warmup: int = 5
    early_stopping_patience: int = 100
    label_smoothing_epsilon: float = 0.0

    use_gcod_loss: bool = False
    gcod_u_lr: float = 0.01
    gcod_fixed_atrain: float = 0.7
    gcod_l1_weight: float = 1.0
    gcod_l2_weight: float = 1.0
    gcod_l3_weight: float = 1.0
    gcod_u_init_value: float = 0.0

    @property
    def folder_name(self) -> str:
        if self.output_tag:
            if self.output_tag.startswith("pretrain_phase_") and len(self.output_tag.split("_")[-1]) == 1:
                 return self.output_tag.split("_")[-1] 
            if self.output_tag.startswith("gcod_finetune_") and len(self.output_tag.split("_")[-1]) == 1:
                return self.output_tag.split("_")[-1]
            return self.output_tag 
        files_str = self.train_path if self.train_path is not None else self.test_path
        if files_str is None:
            return "unknown_folder"
        first_file_path = files_str.split(' ')[0]
        return os.path.basename(os.path.dirname(first_file_path))


ALL_DATASET_CHARS = ['A', 'B', 'C', 'D']

DATASET_SPECIFIC_SEEDS = {
    'A': 1001, 'B': 1002, 'C': 1003, 'D': 1004
}

GLOBAL_HYPERPARAMS = {
    "batch_size": 24,
    "hidden_dim": 128,
    "latent_dim": 8,
    "num_classes": 6,
    "epochs": 200,
    "learning_rate": 0.0005,
    "warmup": 5,
    "early_stopping_patience": 30,
    "label_smoothing_epsilon": 0.1
}