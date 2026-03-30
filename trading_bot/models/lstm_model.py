import torch
import torch.nn as nn

class LSTMTradingModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.2):
        super(LSTMTradingModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # The LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layer for the final output
        # Here we output 1 value: the probability of the price going UP
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x is expected to have shape: (batch_size, sequence_length, input_size)
        
        # Initialize hidden and cell states with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the LAST time step only
        out = out[:, -1, :]
        
        # Pass through fully connected layer
        out = self.fc(out)
        
        # Apply sigmoid to squash output to [0, 1] probability
        out = torch.sigmoid(out)
        
        return out
