#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Step 4: Model Optimiaztion 


# In[3]:


# Model 2: Bidirectional LSTM
import torch.nn as nn

class ImprovedLSTMTrader(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True  # ← Bidirectional
        )
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size * 2, 2)  # The output is still binary

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Only take the output of the last time step
        out = self.norm(out)
        out = self.dropout(out)
        return self.fc(out)
        
model = ImprovedLSTMTrader(input_size=X.shape[2])
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# In[ ]:


# Train Model and Test Accuracy
train_losses = []

for epoch in range(10):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)  #record avg_loss
    print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for xb, yb in test_loader:
        preds = model(xb).argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)

print(f"Test Accuracy: {correct / total:.2%}")


# In[ ]:


# Loss Curve Visualization
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.plot(train_losses, marker='o')
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:


# Model 3: Regularized LSTM
class RegularizedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Only take the output of the last time step
        out = self.norm(out)
        out = self.dropout(out)
        return self.fc(out)

# Standardize model
model = RegularizedLSTM(input_size=X.shape[2])
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



# In[8]:


# Train Model with Early Stopping and Validation
train_losses, val_losses = [], []
best_val_loss = float('inf')
patience, patience_counter = 5, 0

for epoch in range(30):
    # Train
    model.train()
    total_train_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    total_val_loss = 0
    correct = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            total_val_loss += loss.item()
            correct += (pred.argmax(dim=1) == yb).sum().item()
    avg_val_loss = total_val_loss / len(test_loader)
    val_losses.append(avg_val_loss)
    val_acc = correct / len(test_ds)

    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Val Acc = {val_acc:.2%}")

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break



# In[ ]:


# Loss Curve Standarize
plt.figure(figsize=(8, 4))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:


# Exact model forcasting results
import torch

model.eval()
with torch.no_grad():
    test_X_tensor = torch.tensor(X_test)
    preds = model(test_X_tensor)
    predicted_labels = preds.argmax(dim=1).numpy()  #Predict Result: 0 or 1
close_series = df['close'].values
offset = len(close_series) - len(X)
close_test = close_series[offset + split:]

