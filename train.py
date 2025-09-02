import torch
from torch.utils.data import DataLoader
from dataset import WeatherDataset
from model import BiGRUModel
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"




# Dataset 

dataset = WeatherDataset("./weather.csv",window_size=24)
loader = DataLoader(dataset,batch_size=32,shuffle=True)


# Model

model = BiGRUModel(input_dim=4)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)


# Training

losses = []
for epoch in range(2):
    epoch_loss = 0
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        preds = model(x_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    losses.append(epoch_loss / len(loader))
    print(f"Epoch {epoch+1}: Loss = {losses[-1]:.4f}")
    
    
# Plot

plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# Save Model

torch.save(model.state_dict(), "weather_bigru.pth")
