import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

class PairDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

train_ds = PairDataset(X_train, y_train)
val_ds = PairDataset(X_val, y_val)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=16)

class WeightPredictor(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=16, n_weights=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_weights)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        return self.softmax(self.fc2(torch.relu(self.fc1(x))))

model = WeightPredictor()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# ### Training loop
for epoch in range(20):
    model.train()
    for X_batch, y_batch in train_loader:
        weights = model(X_batch)  # → [w_audio, w_meta]
        loss = criterion(weights, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, loss={loss.item():.4f}")


# ### Inference
model.eval()
with torch.no_grad():
    test_input = torch.tensor([[0.75, 0.30]], dtype=torch.float32)  # example features
    predicted_weights = model(test_input)
    print("Predicted weights:", predicted_weights.numpy())
