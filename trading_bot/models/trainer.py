import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import RobustScaler
from config import MODEL_DIR, SEQUENCE_LENGTH, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, LEARNING_RATE, EPOCHS, BATCH_SIZE
import joblib

def train_model(X, y, symbol: str, model_path: str = None) -> nn.Module:
    """
    Trains a new LSTM model.
    Saves the PyTorch `.pth` and an `sklearn` RobustScaler.
    """
    if model_path is None:
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, f"lstm_{symbol.replace('/', '_')}.pth")
        
    scaler_path = model_path.replace('.pth', '_scaler.pkl')
    
    # 1. Scale Features
    # LSTM networks are sensitive to unscaled inputs
    # Shape of X is (samples, sequence_length, features)
    samples, seq_len, num_features = X.shape
    X_reshaped = X.reshape(-1, num_features)
    
    # RobustScaler handles stock price outliers better than MinMaxScaler
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_reshaped)
    X_scaled = X_scaled.reshape(samples, seq_len, num_features)
    
    # Save the fitted scaler so the Predictor can use the exact same normalization!
    joblib.dump(scaler, scaler_path)

    # 2. Convert to PyTorch tensors
    tensor_X = torch.Tensor(X_scaled)
    tensor_y = torch.Tensor(y).unsqueeze(1) # shape (samples, 1)

    # 3. Create Dataset and DataLoader for batching
    dataset = TensorDataset(tensor_X, tensor_y)
    
    # We don't shuffle time series data if we want to preserve ordering,
    # but since sequences are completely built, shuffling batches is typically fine for LSTM
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 4. Initialize Network
    # We dynamically pass the shape of the features the user generated
    model = LSTMTradingModel(
        input_size=num_features, 
        hidden_size=HIDDEN_SIZE, 
        num_layers=NUM_LAYERS, 
        dropout=DROPOUT
    )
    
    # Binary Cross Entropy Loss since target is 0 or 1
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # 5. Training Loop
    model.train()
    print(f"[Trainer] Training {symbol} on {device}...")
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_loss = total_loss / len(loader)
            print(f"[Trainer] Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

    # 6. Save trained model weights
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': num_features,
        'hidden_size': HIDDEN_SIZE,
        'num_layers': NUM_LAYERS,
        'dropout': DROPOUT
    }, model_path)
    print(f"[Trainer] Model saved to {model_path}")
    
    return model

from trading_bot.models.lstm_model import LSTMTradingModel
