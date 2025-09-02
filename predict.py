import torch
import matplotlib.pyplot as plt
from dataset import WeatherDataset # Assuming dataset.py is in the same directory
from model import BiGRUModel       # Assuming model.py is in the same directory
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# --- Setup and Configuration ---

def main():
    """
    Main function to load the model and run predictions.
    """
    # 1. Device Configuration
    # Set the device to a GPU (cuda) if available, otherwise use the CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Parameters
    csv_path = "./weather.csv"
    model_path = "weather_bigru_fixed.pth"
    window_size = 24
    input_dim = 4  # Should match the input_dim used during training
    num_samples_to_test = 100

    # 3. Load Dataset
    try:
        dataset = WeatherDataset(csv_path, window_size=window_size)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {csv_path}")
        return

    # 4. Load Model
    # Initialize the model and move it to the configured device.
    model = BiGRUModel(input_dim=input_dim).to(device)
    
    try:
        # Load the saved weights from the trained model
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return
        
    # Set the model to evaluation mode. This disables layers like Dropout.
    model.eval()

    # --- Prediction Loop ---

    print(f"Running predictions on the first {num_samples_to_test} samples...")
    true_vals = []
    pred_vals = []

    for i in range(num_samples_to_test):
        # Get a single sample from the dataset
        x, y = dataset[i]
        
        # Prepare the input tensor:
        # 1. Add a batch dimension with .unsqueeze(0) -> shape becomes [1, seq_len, features]
        # 2. Convert to float type
        # 3. Move to the selected device
        input_tensor = x.unsqueeze(0).float().to(device)

        # Perform inference within a no_grad() context to save memory and computations
        with torch.no_grad():
            pred = model(input_tensor)
        
        # Store the actual and predicted values
        # .item() extracts the scalar value from a tensor
        true_vals.append(y.item())
        pred_vals.append(pred.squeeze().item())

    print("Prediction complete.")

    # --- Plotting the Results ---

    plt.figure(figsize=(12, 6))
    plt.plot(true_vals, label="Actual Temperature", color='blue', marker='o', linestyle='-', markersize=4)
    plt.plot(pred_vals, label="Predicted Temperature", color='red', marker='x', linestyle='--', markersize=4)
    plt.legend()
    plt.title("Temperature Forecast vs. Actual Values")
    plt.xlabel("Time Step (Sample Number)")
    plt.ylabel("Normalized Temperature")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()