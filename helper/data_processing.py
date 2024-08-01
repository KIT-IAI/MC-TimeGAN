"""data_processing.py"""

import os
import json
import datetime
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

# Function to save model state and metadata
def save_model(model: nn.Module, data: Dict, name: Optional[str] = None) -> None:
    """
    Save the model state and metadata.

    Args:
    - model (nn.Module): The neural network model to be saved.
    - data (dict): Metadata associated with the training.
    - name (str): Optional custom name for the saved files.

    Returns:
    - None
    """
    model_dir = os.path.join('helper', 'models')
    date_str = str(datetime.date.today())
    name = f"{date_str}_{name}" if name is not None else f"{date_str}_MC-TimeGAN"
    # Ensure the directory exists
    os.makedirs(model_dir, exist_ok=True)
    # Save metadata as JSON file
    metadata_path = os.path.join(model_dir, name + '.json')
    with open(metadata_path, 'w', encoding='utf-8') as metadata_file:
        json.dump(data, metadata_file, ensure_ascii=False, indent=4)
    # Save model state as .pth file
    model_path = os.path.join(model_dir, name + '.pth')
    torch.save(model.state_dict(), model_path)

# Function to load CSV files into pandas DataFrames
def loading(*files: str) -> pd.DataFrame:
    """
    Load CSV files into pandas DataFrames.

    Args:
    - files (str): Names of the CSV files to be loaded (without extension).

    Returns:
    - pd.DataFrame or tuple of pd.DataFrame: Loaded data.
    """
    print("""
    ___  ________     _____ _                _____   ___   _   _ 
    |  \/  /  __ \   |_   _(_)              |  __ \ / _ \ | \ | |
    | .  . | /  \/_____| |  _ _ __ ___   ___| |  \// /_\ \|  \| |
    | |\/| | |  |______| | | | '_ ` _ \ / _ \ | __ |  _  || . ` |
    | |  | | \__/\     | | | | | | | | |  __/ |_\ \| | | || |\  |
    \_|  |_/\____/     \_/ |_|_| |_| |_|\___|\____/\_| |_/\_| \_/""")
    return_list = list()
    for file_name in files:
        data = pd.read_csv(file_name)
        print('Shape of "' + file_name + '":', data.shape)
        return_list.append(data)

    return return_list.pop() if len(return_list) == 1 else tuple(return_list)

# Function to prepare data with scaling and sliding window
def preparing(*inputs: Tuple, horizon: int, shuffle_stack: bool = True,
              random_state: Optional[int] = None):
    """
    Prepare data by scaling and creating sequences with a sliding window.
    Args:
    - inputs (tuple): Tuples of data and boolean indicating if scaling is needed.
    - horizon (int): The width of the sliding window.
    - shuffle_stack (bool): Whether to shuffle the data stack.
    - random_state (int): Seed for random number generator.
    Returns:
    - np.ndarray: Prepared data stack.
    """
    if len(inputs) > 2:
        raise ValueError('Only one input (data) or two inputs (data and labels) are allowed')

    return_list = []
    for data, bool_scale in inputs:
        # Scaling
        if bool_scale:
            scaler = MinMaxScaler().fit(data)
            data = scaler.transform(data)
            max_val = scaler.data_max_
            min_val = scaler.data_min_

        # Create sequences with sliding window
        data_stack = [data[i:i+horizon] for i in range(len(data) - horizon)]
        data_stack = np.stack(data_stack)

        # Shuffle data stack if required
        if shuffle_stack:
            data_stack = shuffle(data_stack, random_state=random_state)

        if bool_scale:
            return_list.extend([data_stack, max_val, min_val])
        else:
            return_list.append(data_stack)

    return return_list
