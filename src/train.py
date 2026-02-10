import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os

# Import your model
from src.model import ProcessLSTM

# --- 1. Helper Function: The Training Loop ---
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train() # Set model to training mode (enables Dropout, etc.)
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_X, batch_y in loader:
        # Move data to GPU/CPU
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        # A. Zero Gradients (Crucial!)
        optimizer.zero_grad()
        
        # B. Forward Pass
        predictions = model(batch_X) # Shape: [Batch, Num_Classes]
        
        # C. Calculate Loss
        loss = criterion(predictions, batch_y)
        
        # D. Backward Pass (Backprop)
        loss.backward()
        
        # E. Update Weights
        optimizer.step()
        
        total_loss += loss.item()

        # --- Accuracy Calculation ---
        # argmax gives the index of the highest probability (the prediction)
        _, predicted_classes = torch.max(predictions, 1)
        correct += (predicted_classes == batch_y).sum().item()
        total += batch_y.size(0)
        
    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy

# --- 2. Main Execution ---
def main():
    # A. Setup
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {DEVICE}")
    
    # HYPERPARAMETERS
    NUM_CLASSES = 25       
    EMBED_DIM = 16
    HIDDEN_DIM = 64
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 10

    # B. Data Loading 
    # ---------------------------------------------------------
    print("Loading data...")
    X_train = torch.load('./data/X_train.pt')
    y_train = torch.load('./data/y_train.pt')
    # ---------------------------------------------------------

    # Create TensorDatasets
    train_ds = TensorDataset(torch.LongTensor(X_train), torch.LongTensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # C. Model, Loss, Optimizer
    model = ProcessLSTM(NUM_CLASSES, EMBED_DIM, HIDDEN_DIM).to(DEVICE)
    criterion = nn.CrossEntropyLoss() # The standard for classification
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # D. The Training Loop
    print("\nStarting Training...")
    for epoch in range(EPOCHS):
        avg_loss, accuracy = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {avg_loss:.4f} | Acc: {accuracy:.2%}")

    # Create the directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # E. Save the Model (The "Artifact")
    torch.save(model.state_dict(), "models/process_lstm.pth")
    print("\nModel saved to models/process_lstm.pth")

if __name__ == "__main__":
    main()