import re
import csv
import sys
import os

def extract_training_loss(input_file, output_file="training_loss.csv"):
    """
    Extract training loss information from a log file and save to CSV.
    
    Args:
        input_file (str): Path to the input log file
        output_file (str): Path to the output CSV file (default: "training_loss.csv")
    
    Returns:
        list: The extracted data as a list of [index, epoch_info, loss] entries
    """
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return None

    # Open the input file and extract relevant data
    data = []
    with open(input_file, "r") as file:
        for idx, line in enumerate(file):
            # Match the desired pattern
            match = re.search(r"Train Epoch: \[(\d+)\]  \[\s*(\d+)/(\d+)\].*?loss: ([\d\.]+)", line)
            if match:
                epoch = match.group(1)
                batch = match.group(2)
                total = match.group(3)
                epoch_info = f"{epoch}-{batch}/{total}"
                loss = match.group(4)
                data.append([idx, epoch_info, loss])

    # Write the extracted data to a CSV file
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Index", "Epoch Info", "Loss"])  # Header
        writer.writerows(data)

    print(f"Data extracted and saved to {output_file}")
    return data

# Example usage when run as a script
if __name__ == "__main__":
    # Default values
    input_file = "Yan_small_model_4M_finetune.txt"
    output_file = "Yan_small_model_4M_finetune_loss.csv"
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    print(f"Processing file: {input_file}")
    print(f"Output will be saved to: {output_file}")
    
    extract_training_loss(input_file, output_file)