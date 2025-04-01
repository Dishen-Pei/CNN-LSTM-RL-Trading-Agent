#!/usr/bin/env python
# coding: utf-8

# In[35]:


get_ipython().system('pip install pandas numpy scikit-learn torch matplotlib')
get_ipython().system('pip install ta')


# In[24]:


# Step 1: Load Data
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator  
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands
df = pd.read_csv(r"C:\Users\24716\Downloads\cleaned_crypto_data.csv")
features_cols = ['open', 'high', 'low', 'close', 'Volume BTC', 'Volume USDT', 'price_change', 'price_range', 'vwap']


# In[26]:


# Step 2: Create tech indicator
df["rsi_14"] = RSIIndicator(close=df["close"], window=14).rsi()
macd = MACD(close=df["close"])
df["macd"] = macd.macd()
df["macd_signal"] = macd.macd_signal()
df["ema_5"] = EMAIndicator(close=df["close"], window=5).ema_indicator()
df["ema_20"] = EMAIndicator(close=df["close"], window=20).ema_indicator()
bb = BollingerBands(close=df["close"], window=20)
df["bb_upper"] = bb.bollinger_hband()
df["bb_middle"] = bb.bollinger_mavg()
df["bb_lower"] = bb.bollinger_lband()
df["momentum_10"] = df["close"] - df["close"].shift(10)


# In[28]:


# Step 3: Preprocess
future_close = df['close'].shift(-1)
df['label'] = (future_close > df['close']).astype(int)
df.dropna(inplace=True)

features_cols = [
    'open', 'high', 'low', 'close', 'Volume BTC', 'Volume USDT',
    'price_change', 'price_range', 'vwap',
    'rsi_14', 'macd', 'macd_signal',
    'ema_5', 'ema_20',
    'bb_upper', 'bb_middle', 'bb_lower',
    'momentum_10'
]

features = df[features_cols].values
labels = df['label'].values

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)


# In[30]:


#Step 4: Create Sequences
def create_sequences(features, labels, window=24):
    X, y = [], []
    for i in range(len(features) - window):
        X.append(features[i:i+window].astype(np.float32))
        y.append(labels[i+window])
    return np.array(X), np.array(y)

X, y = create_sequences(features_scaled[-100000:], labels[-100000:], window=24)


# In[32]:


#Step 5: Dataset Split 
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long))
test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test, dtype=torch.long))

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64)


# In[34]:


#Step 6: Define Model 
class LSTMTrader(nn.Module):
    def __init__(self, input_size=9, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = LSTMTrader(input_size=X.shape[2])
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# In[36]:


# --- Step 6: Train Model ---
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
    train_losses.append(avg_loss)  # ← 记录
    print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")


# In[38]:


import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.plot(train_losses, marker='o')
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.show()


# In[42]:


#Step 7: Evaluate
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for xb, yb in test_loader:
        preds = model(xb).argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)

print(f"Test Accuracy: {correct / total:.2%}")


# In[44]:


#Step 8: Bidirectional LSTM
import torch.nn as nn

class ImprovedLSTMTrader(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True  # ← 双向 LSTM
        )
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size * 2, 2)  # 输出仍然是二分类

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 只取最后一层时间步的输出
        out = self.norm(out)
        out = self.dropout(out)
        return self.fc(out)
        
model = ImprovedLSTMTrader(input_size=X.shape[2])
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# In[46]:


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
    train_losses.append(avg_loss)  # ← 记录
    print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for xb, yb in test_loader:
        preds = model(xb).argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)

print(f"Test Accuracy: {correct / total:.2%}")


# In[48]:


import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.plot(train_losses, marker='o')
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.show()


# In[50]:


#Regularized LSTM
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
        out = out[:, -1, :]  # 取最后时刻
        out = self.norm(out)
        out = self.dropout(out)
        return self.fc(out)

# 初始化模型
model = RegularizedLSTM(input_size=X.shape[2])
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# === 训练流程 + Early Stopping ===
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

# === 可视化损失曲线 ===
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


#Exact model forcasting results
import torch

model.eval()
with torch.no_grad():
    test_X_tensor = torch.tensor(X_test)
    preds = model(test_X_tensor)
    predicted_labels = preds.argmax(dim=1).numpy()  #Predict Result: 0 or 1
close_series = df['close'].values
offset = len(close_series) - len(X)
close_test = close_series[offset + split:]

