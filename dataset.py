import pandas as pd    
import torch
from torch.utils.data import Dataset


class WeatherDataset(Dataset):
    def __init__(self,csv_path,window_size=24):
        self.data = pd.read_csv(csv_path)
        self.data = self.data[['temperature', 'relative_humidity', 'wind_speed_10m (km/h)', 'surface_pressure (hPa)']]
        
        
        self.min = self.data.min()
        self.max = self.data.max()
        self.data = (self.data - self.min) / (self.max - self.min )

        
        self.window_size = window_size
        self.X,self.y = [] , []
        
        
        for i in range(len(self.data) - window_size):
            seq = self.data.iloc[i:i+window_size].values
            target = self.data.iloc[i+window_size]["temperature"]
            
            
            self.X.append(seq)
            self.y.append(target)
            
            
    def __len__(self):
        return len(self.X)
    
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx],dtype=torch.float32),
            torch.tensor(self.y[idx],dtype=torch.float32)
        )

