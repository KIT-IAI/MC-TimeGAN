""" __main__.py """
import os
import argparse
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import torch
import numpy as np
import pandas as pd

from helper.mctimegan import MCTimeGAN
from helper.data_processing import loading, preparing
from helper.metrics import visualization

def parse_arguments():
    """
    Parse command line arguments.
    Returns:
        args: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="MC-TimeGAN Training Script")
    parser.add_argument(
        '--data', default=r"helper\data\raw\feeder_sgens_4w_data.csv", type=str,
        help="Path to the data file"
    )
    parser.add_argument(
        '--labels', default=r"helper\data\raw_labels\feeder_sgens_4w_labels_ordinal.csv", type=str,
        help="Path to the labels file"
    )
    parser.add_argument(
        '--horizon', default=96, type=int,
        help="Horizon for sequence slicing"
    )
    parser.add_argument(
        '--hidden_dim', default=8, type=int,
        help="Hidden dimension size for the model"
    )
    parser.add_argument(
        '--num_layers', default=3, type=int,
        help="Number of layers in the model"
    )
    parser.add_argument(
        '--epochs', default=1000, type=int,
        help="Number of training epochs"
    )
    parser.add_argument(
        '--batch_size', default=128, type=int,
        help="Batch size for training"
    )
    parser.add_argument(
        '--learning_rate', default=1e-3, type=float,
        help="Learning rate for training"
    )
    parser.add_argument(
        '--csv_filename', default=r'helper\synthetic_data\main_mctimegan_synthetic_sgen_data.csv', type=str,
        help="Filename for the exported CSV of synthetic data"
    )
    return parser.parse_args()

def train_model(args):
    """
    Train the MC-TimeGAN model and generate synthetic sequences.
    Args:
        args: Command line arguments.
    Returns:
        None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using {device} device')

    # Load data and labels
    data, labels = loading(args.data, args.labels)
    print(f'Shape of data: {data.shape}')
    print(f'Shape of labels: {labels.shape}')

    # Preprocessing Pt. 1: Scale data and slice data/labels into sequences
    data_train, max_val, min_val, labels_train = preparing(
        (data, True), (labels, False), horizon=args.horizon, shuffle_stack=False
    )

    # Preprocessing Pt. 2: Shuffle data and labels consistently
    data_train, labels_train = shuffle(data_train, labels_train)

    # Initialize MC-TimeGAN model
    model = MCTimeGAN(
        input_features=data_train.shape[-1],
        input_conditions=labels_train.shape[-1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    ).to(device)

    # Train the model
    model.fit(data_train, cond_labels=labels_train)

    # Synthesize sequences
    data_gen = model.transform(data_train.shape, cond=labels_train)

    # Rescale generated data from range (0,1) back to original feature range
    data_gen = data_gen * (max_val - min_val) + min_val

    # Rescale training data
    data_ori = data_train * (max_val - min_val) + min_val
    # Print the shape of the generated data
    print(data_gen.shape)
    # Reshape data_gen to 2D
    data_gen_reshaped = data_gen.reshape(data_gen.shape[0], -1)
    # Print the reshape of the generated data
    print(data_gen.shape)

    # Convert the generated data to a pandas DataFrame
    data_export = pd.DataFrame(data_gen_reshaped, columns=['fake_' + str(i+1) for i in range(data_gen_reshaped.shape[1])])
    # Export the DataFrame to a CSV file
    data_export.to_csv(args.csv_filename, index=False)
    # Visualize generated sequences
    visualize_sequences(data_gen, data_ori, labels_train, args.horizon)

    # Visualize data using PCA and t-SNE
    visualize_data(data_train, data_gen)

def visualize_sequences(data_gen, data_ori, labels_train, horizon):
    """
    Visualize the generated and original sequences.
    Args:
        data_gen: Generated data.
        data_ori: Original data.
        labels_train: Training labels.
        horizon: Horizon for sequence slicing.
    Returns:
        None
    """
    _, ax = plt.subplots()
    ax_label = ax.twinx()
    for i in range(2):
        ax.plot(data_gen[i, :, :], label='fake_' + str(i))
    ax.plot(data_ori[0, :, :], label='real_0')
    for i in range(2):
        ax_label.plot(labels_train[i, :, :], '.', alpha=0.5)
    ax.legend()
    ax.set_xlim(-0.1, horizon + 0.1)
    ax.set_xlabel('Horizon')
    ax.set_ylabel('Magnitude')
    ax_label.set_ylabel('Label')
    ax.grid(True)
    # Save the plot
    if not os.path.exists('helper/synthetic_data'):
        os.makedirs('helper/synthetic_data')
    plt.savefig('helper/synthetic_data/main_generated_and_original_sequences.png')
    plt.show()

def visualize_data(data_train, data_gen):
    """
    Visualize the data using PCA and t-SNE.
    Args:
        data_train: Training data.
        data_gen: Generated data.
    Returns:
        None
    """
    visualization(data_train, data_gen, 'pca')
    visualization(data_train, data_gen, 'tsne')

def main():
    """
    Main function to parse arguments and train the MC-TimeGAN model.
    Args:
        None
    Returns:
        None
    """
    args = parse_arguments()
    train_model(args)

if __name__ == '__main__':
    main()
