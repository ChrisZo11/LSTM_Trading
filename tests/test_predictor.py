import os
import torch
import numpy as np
import pytest
from trading_bot.models.lstm_model import LSTMTradingModel
from trading_bot.models.trainer import train_model
from trading_bot.models.predictor import predict_signal

def test_lstm_model_instantiation():
    model = LSTMTradingModel(input_size=10, hidden_size=32, num_layers=2)
    assert isinstance(model, torch.nn.Module)
    assert model.hidden_size == 32

def test_lstm_forward_pass():
    # Model configuration
    batch_size = 4
    seq_len = 60
    input_size = 15
    model = LSTMTradingModel(input_size, hidden_size=64, num_layers=1)
    
    # Dummy tensor
    dummy_input = torch.randn(batch_size, seq_len, input_size)
    output = model(dummy_input)
    
    # Should be (batch_size, 1) and between 0 and 1 because of sigmoid
    assert output.shape == (batch_size, 1)
    assert torch.all((output >= 0) & (output <= 1))

def test_trainer_and_predictor_integration(tmp_path, monkeypatch):
    from config import PROJECT_ROOT

    # Configure mock saving path
    temp_model_dir = str(tmp_path / "saved_models")
    os.makedirs(temp_model_dir, exist_ok=True)
    temp_model_file = os.path.join(temp_model_dir, "lstm_MOCK.pth")
    temp_scaler_file = os.path.join(temp_model_dir, "lstm_MOCK_scaler.pkl")

    # Override MODEL_DIR internally in modules using it
    import trading_bot.models.trainer
    import trading_bot.models.predictor
    monkeypatch.setattr(trading_bot.models.trainer, "MODEL_DIR", temp_model_dir)
    monkeypatch.setattr(trading_bot.models.predictor, "MODEL_DIR", temp_model_dir)
    
    # 1. Provide X and y
    samples = 100
    seq_len = 20
    features = 5
    
    dummy_X = np.random.randn(samples, seq_len, features)
    dummy_y = np.random.choice([0.0, 1.0], size=samples)
    
    # 2. Run Train
    model = train_model(dummy_X, dummy_y, symbol="MOCK", model_path=temp_model_file)
    
    # Verify outputs
    assert os.path.exists(temp_model_file)
    assert os.path.exists(temp_scaler_file)
    assert isinstance(model, torch.nn.Module)
    
    # 3. Predict Single Test Frame
    live_frame = np.random.randn(1, seq_len, features) # Extract 1 frame as 3D array
    signal, conf = predict_signal(live_frame, symbol="MOCK", model_path=temp_model_file)
    
    assert signal in ["BUY", "SELL", "HOLD"]
    assert 0.0 <= conf <= 1.0
