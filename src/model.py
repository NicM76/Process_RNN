import torch
import torch.nn as nn

class ProcessLSTM(nn.Module):
    def __init__(self, num_classes, embedding_dim, hidden_size, num_layers=1):
        super(ProcessLSTM, self).__init__()
        
        # 1. Embedding Layer
        self.embedding = nn.Embedding(num_embeddings=num_classes, 
                                      embedding_dim=embedding_dim, 
                                      padding_idx=0)
        
        # 2. LSTM Layer
        self.lstm = nn.LSTM(input_size=embedding_dim, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True)
        
        # 3. Output Layer
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        out, _ = self.lstm(embedded)
        # Take the output of the LAST time step in the sequence
        last_out = out[:, -1, :]
        logits = self.fc(last_out)
        return logits