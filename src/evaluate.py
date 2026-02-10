import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# Import model
from src.model import ProcessLSTM

# --- CONFIGURATION ---
# These MUST match the hyperparameters used in training
NUM_CLASSES = 25
EMBED_DIM = 16
HIDDEN_DIM = 64
BATCH_SIZE = 64

def load_tensors():
    """
    Loads the pre-saved test tensors from the 'data/processed' folder.
    """
    print("Loading pre-computed test tensors...")
    
    # Adjust path if you saved them elsewhere (e.g. 'data/processed/X_test.pt')
    try:
        X_test = torch.load('data/X_test.pt')
        y_test = torch.load('data/y_test.pt')
        print(f"Loaded X_test: {X_test.shape}")
        print(f"Loaded y_test: {y_test.shape}")
        return X_test, y_test
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit()

def evaluate(model, loader, criterion, device, int_to_act):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            
            # Get the index of the max logit
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            
    avg_loss = total_loss / len(loader)
    
    # --- REPORTING ---
    print(f"\nTest Loss: {avg_loss:.4f}")
    
    # Classification Report
    # We filter target_names to only include classes present in the data/predictions
    # to avoid warnings about missing classes.
    unique_labels = sorted(list(set(all_labels) | set(all_preds)))
    target_names = [int_to_act.get(i, f"Unknown-{i}") for i in unique_labels]
    
    print("\n--- Classification Report ---")
    print(classification_report(all_labels, all_preds, labels=unique_labels, target_names=target_names, zero_division=0))
    
    return all_labels, all_preds

def main():
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device}")

    # 1. Load Data
    X_test, y_test = load_tensors()
    
    # Load Mapping (Critical for readable report)
    with open("data/activity_map.json", "r") as f:
        activity_to_int = json.load(f)
        # Create Reverse Mapping: 0 -> 'PAD', 1 -> 'A_SUBMITTED'
        int_to_act = {int(v): k for k, v in activity_to_int.items()}
        # Add Padding explicitly if missing
        if 0 not in int_to_act:
            int_to_act[0] = "PAD"

    # Create DataLoader
    test_ds = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Load Model Structure
    model = ProcessLSTM(NUM_CLASSES, EMBED_DIM, HIDDEN_DIM).to(device)
    
    # 3. Load Model Weights
    weights_path = "models/process_lstm.pth"
    if not os.path.exists(weights_path):
        print(f"Error: Weights not found at {weights_path}")
        return

    model.load_state_dict(torch.load(weights_path, map_location=device))
    print("Model weights loaded successfully.")

    # 4. Run Evaluation
    criterion = nn.CrossEntropyLoss()
    y_true, y_pred = evaluate(model, test_loader, criterion, device, int_to_act)
    
    # 5. Plot Confusion Matrix
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                    xticklabels=[int_to_act.get(i, str(i)) for i in sorted(list(set(y_pred)))],
                    yticklabels=[int_to_act.get(i, str(i)) for i in sorted(list(set(y_true)))])
        plt.title("Confusion Matrix")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig("models/confusion_matrix.png")
        print("\nConfusion Matrix saved to 'models/confusion_matrix.png'")
    except Exception as e:
        print(f"\nCould not plot confusion matrix: {e}")

if __name__ == "__main__":
    main()