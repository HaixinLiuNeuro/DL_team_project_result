import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_training_loss(csv_file):
    """
    Plot the training loss from the CSV file with Index on x-axis and Loss on y-axis
    
    Args:
        csv_file (str): Path to the CSV file containing training loss data
    """
    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} not found.")
        return
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(df['Index'], df['Loss'].astype(float), 'b-', alpha=0.7)
    
    # Add a smoother trend line
    window_size = min(100, len(df) // 10) if len(df) > 100 else 10
    if window_size > 1:
        df['Smoothed_Loss'] = df['Loss'].astype(float).rolling(window=window_size).mean()
        plt.plot(df['Index'], df['Smoothed_Loss'], 'r-', linewidth=2)
    
    # Customize the plot
    plt.title('Training Loss vs. Index')
    plt.xlabel('Index')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    # Add legend if we have a smoothed line
    if window_size > 1:
        plt.legend(['Raw Loss', f'Moving Average (window={window_size})'])
    
    # Save the figure
    output_file = os.path.splitext(csv_file)[0] + '_plot.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Plot saved as {output_file}")

if __name__ == "__main__":
    # Default file name or take from command line
    import sys
    csv_file = "Yan_small_model_4M_finetune_loss.csv"
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    
    plot_training_loss(csv_file)