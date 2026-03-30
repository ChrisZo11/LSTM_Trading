import os
import torch
import joblib
import numpy as np
from config import MODEL_DIR
from trading_bot.models.lstm_model import LSTMTradingModel

CONFIDENCE_THRESHOLD = 0.60  # minimum probability to act; else HOLD


def predict_signal(live_sequence, symbol: str, model_path: str = None) -> tuple[str, float]:
    """
    Given a single 3D sequence of features (1, sequence_length, features),
    return a (signal, confidence) tuple where signal is 'BUY', 'SELL', or 'HOLD'.
    """
    if model_path is None:
        model_path = os.path.join(MODEL_DIR, f"lstm_{symbol.replace('/', '_')}.pth")
        
    scaler_path = model_path.replace('.pth', '_scaler.pkl')

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return "HOLD", 0.0

    # 1. Load the exact Scaler used during training
    scaler = joblib.load(scaler_path)
    
    # live_sequence is shape (1, seq_len, num_features)
    samples, seq_len, num_features = live_sequence.shape
    live_reshaped = live_sequence.reshape(-1, num_features)
    scaled_live = scaler.transform(live_reshaped)
    scaled_live = scaled_live.reshape(1, seq_len, num_features)

    # 2. Convert to PyTorch Tensor
    tensor_live = torch.Tensor(scaled_live)

    # 3. Load the Model checkpoint
    checkpoint = torch.load(model_path)
    model = LSTMTradingModel(
        input_size=checkpoint['input_size'],
        hidden_size=checkpoint['hidden_size'],
        num_layers=checkpoint['num_layers'],
        dropout=checkpoint['dropout']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # 4. Predict probability vector
    with torch.no_grad():
        tensor_live = tensor_live.to(device)
        output = model(tensor_live)
        prob_up = output.item()
    
    # 5. Threshold prediction into a categorical Signal
    prob_down = 1.0 - prob_up

    if prob_up >= CONFIDENCE_THRESHOLD:
        return "BUY", float(prob_up)
    elif prob_down >= CONFIDENCE_THRESHOLD:
        return "SELL", float(prob_down)
    else:
        return "HOLD", float(max(prob_up, prob_down))
