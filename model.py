import torch
import torch.nn as nn



class BiGRUModel(nn.Module):
    def __init__(self,input_dim, hidden_dim=64, num_layers=1):
        super(BiGRUModel,self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers,
                          batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)  # ×2 برای bidirectional

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]  # آخرین تایم‌استپ
        return self.fc(out).squeeze()