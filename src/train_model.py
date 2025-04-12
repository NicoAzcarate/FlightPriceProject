import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import pandas as pd
from nn_dataset import FlightDataset

class FlightPriceNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

def run_training():
    # Cargar dataset
    df = pd.read_parquet("../data_subsets/final_dataset.parquet")
    df = df.dropna()  # Limpiar nulos

    # Crear dataset
    dataset = FlightDataset(df)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1024)

    model = FlightPriceNN(input_size=df.shape[1] - 1)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):  # Ajusta según lo que necesites
        model.train()
        total_loss = 0
        for X, y in train_loader:
            pred = model(X)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "../models/nn_model.pt")
    print("✅ Modelo guardado en '../models/nn_model.pt'")
