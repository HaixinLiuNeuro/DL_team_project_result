import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def read_csv_data(file_path):
    """Read CSV data and extract epoch info and loss values"""
    data = pd.read_csv(file_path)
    
    # Check if 'Loss' column exists, if not, try to find the right column
    if 'Loss' not in data.columns and len(data.columns) >= 3:
        loss_column = data.columns[2]  # Assume third column is loss
    else:
        loss_column = 'Loss'
    
    return data[loss_column].values

def plot_loss_curves(file1, file2, output_path=None, labels=None):
    """
    Plot loss curves from two CSV files side by side
    
    Args:
        file1: Path to first CSV file
        file2: Path to second CSV file
        output_path: Path to save the output figure
        labels: List of two strings to use as labels for the curves
    """
    
    # Read loss data
    loss_data1 = read_csv_data(file1)
    loss_data2 = read_csv_data(file2)
    
    # Create x-axis values (steps)
    x1 = np.arange(len(loss_data1))
    x2 = np.arange(len(loss_data2))
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Use custom labels if provided, otherwise use filenames
    label1 = labels[0] if labels and len(labels) > 0 else os.path.basename(file1).split('.')[0]
    label2 = labels[1] if labels and len(labels) > 1 else os.path.basename(file2).split('.')[0]
    
    # Plot both loss curves with enhanced visibility
    plt.plot(x1, loss_data1, label=label1, color='blue', linewidth=2, alpha=0.8)
    plt.plot(x2, loss_data2, label=label2, color='red', linewidth=2, alpha=0.8)
    
    # Add grid and styling
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Comparison', fontsize=14)
    plt.legend(fontsize=10, framealpha=0.8, loc='best')
    
    # Improve appearance
    plt.tight_layout()
    
    # Save the plot if output path is specified
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Plot and compare loss curves from two CSV files.')
    parser.add_argument('--file1', type=str, required=True, help='Path to first CSV file')
    parser.add_argument('--file2', type=str, required=True, help='Path to second CSV file')
    parser.add_argument('--labels', nargs=2, type=str, help='Custom labels for the two curves')
    parser.add_argument('--output', type=str, help='Path to save the output figure')
    
    args = parser.parse_args()
    
    # Use command line arguments
    plot_loss_curves(args.file1, args.file2, args.output, args.labels)
    
    # Print information about the created plot
    print(f"Plotted loss curves from:")
    print(f"  - {args.file1}")
    print(f"  - {args.file2}")
    if args.output:
        print(f"Plot saved to: {args.output}")