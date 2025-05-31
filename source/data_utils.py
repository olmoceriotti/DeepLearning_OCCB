import os
import gzip
import json
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from typing import List, Tuple

def load_dataset(file_path: str) -> pd.DataFrame:
    data = []
    db_tags = []
    for file in file_path.split(' '):
        folder_tag = os.path.basename(os.path.dirname(file))
        with gzip.open(file, 'rt', encoding='utf-8') as f:
            tmp = json.load(f)
            data.extend(tmp)
            db_tags.extend([folder_tag] * len(tmp))
    df = pd.DataFrame(data)
    if not df.empty:
        df = df.assign(db=db_tags)
    return df

def create_dataset_from_dataframe(df: pd.DataFrame, result: bool = True) -> List[Data]:
    dataset = []
    for _, row in df.iterrows():
        edge_index = torch.tensor(row['edge_index'], dtype=torch.long)
        edge_attr = torch.tensor(row['edge_attr'], dtype=torch.float)
        num_nodes = row['num_nodes']
        y_val = 0
        if result and 'y' in row and isinstance(row['y'], list) and len(row['y']) > 0 and \
           isinstance(row['y'][0], list) and len(row['y'][0]) > 0:
            y_val = row['y'][0][0]
        
        y = torch.tensor([y_val], dtype=torch.long) if result else torch.tensor([0], dtype=torch.long)

        data_obj = Data(x=torch.ones((num_nodes, 1)),
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        y=y,
                        num_nodes=num_nodes)
        
        data_obj.x = torch.nan_to_num(data_obj.x, nan=0.0)
        data_obj.edge_attr = torch.nan_to_num(data_obj.edge_attr, nan=0.0)
        dataset.append(data_obj)
    return dataset

def create_data_loader(df: pd.DataFrame, batch_size: int, train: bool = True, use_indexed_dataset: bool = False) -> DataLoader:
    pyg_dataset = create_dataset_from_dataframe(df, result=train)
    if use_indexed_dataset:
        final_dataset = IndexedDataset(pyg_dataset)
    else:
        final_dataset = pyg_dataset
    return DataLoader(final_dataset, batch_size=batch_size, shuffle=train)


class IndexedDataset(Dataset):
    def __init__(self, original_dataset: List[Data]):
        super().__init__()
        self.original_dataset = original_dataset

    def get(self, idx: int) -> Tuple[Data, int]:
        return self.original_dataset[idx], idx

    def len(self) -> int:
        return len(self.original_dataset)

    def __getitem__(self, idx):
        return self.get(idx)

    def __len__(self):
        return self.len()