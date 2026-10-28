# Predictive Maintenance and RUL Estimation using LSTM (NASA C-MAPSS FD001)
# Put train_FD001.txt, test_FD001.txt, RUL_FD001.txt in the SAME folder as this file.
# Install once:  pip install numpy pandas matplotlib scikit-learn tensorflow
# Run:           python rul_project.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

WINDOW = 30      # cycles per sample
RUL_CAP = 125    # clip RUL (standard practice)
EPOCHS = 30

# ---------- 1. Load data ----------
cols = ['unit', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]
train = pd.read_csv('train_FD001.txt', sep=r'\s+', header=None, names=cols)
test = pd.read_csv('test_FD001.txt', sep=r'\s+', header=None, names=cols)
true_rul = pd.read_csv('RUL_FD001.txt', header=None)[0].values
true_rul = np.minimum(true_rul, RUL_CAP)
print("Train:", train.shape, "| Test:", test.shape, "| Engines in test:", len(true_rul))

# ---------- 2. Add RUL label to training data ----------
max_cycle = train.groupby('unit')['cycle'].transform('max')
train['RUL'] = (max_cycle - train['cycle']).clip(upper=RUL_CAP)

# ---------- 3. Select useful sensors + scale ----------
# (other sensors are constant in FD001, so they carry no information)
features = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12',
            's13', 's14', 's15', 's17', 's20', 's21']
scaler = MinMaxScaler()
train[features] = scaler.fit_transform(train[features])
test[features] = scaler.transform(test[features])

# ---------- 4. Sliding windows ----------
def make_train_windows(df):
    X, y = [], []
    for _, g in df.groupby('unit'):
        data = g[features].values
        rul = g['RUL'].values
        for i in range(len(g) - WINDOW + 1):
            X.append(data[i:i + WINDOW])
            y.append(rul[i + WINDOW - 1])
    return np.array(X), np.array(y)

def make_test_windows(df):
    X = []
    for _, g in df.groupby('unit'):
        data = g[features].values
        if len(data) < WINDOW:  # pad short engines by repeating first row
            pad = np.repeat(data[:1], WINDOW - len(data), axis=0)
            data = np.vstack([pad, data])
        X.append(data[-WINDOW:])  # last 30 cycles of each test engine
    return np.array(X)

X_train, y_train = make_train_windows(train)
X_test = make_test_windows(test)
print("X_train:", X_train.shape, "| X_test:", X_test.shape)

# ---------- 5. Baseline: Random Forest (on last cycle only) ----------
rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
rf.fit(X_train[:, -1, :], y_train)
rf_pred = rf.predict(X_test[:, -1, :])
rf_rmse = np.sqrt(mean_squared_error(true_rul, rf_pred))

# ---------- 6. LSTM model ----------
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(WINDOW, len(features))),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.summary()

history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=128,
                    validation_split=0.1, verbose=2)

# ---------- 7. Evaluate ----------
lstm_pred = model.predict(X_test).flatten()
lstm_rmse = np.sqrt(mean_squared_error(true_rul, lstm_pred))
print("\n===== RESULTS (RMSE, lower is better) =====")
print(f"Random Forest (baseline): {rf_rmse:.2f}")
print(f"LSTM                    : {lstm_rmse:.2f}")

# ---------- 8. Plots (saved as PNG for the poster) ----------
plt.figure(figsize=(6, 4))
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.title('LSTM Training Loss')
plt.legend(); plt.tight_layout()
plt.savefig('loss_curve.png', dpi=200); plt.close()

order = np.argsort(true_rul)
plt.figure(figsize=(6, 4))
plt.plot(true_rul[order], label='Actual RUL', linewidth=2)
plt.plot(lstm_pred[order], label='Predicted RUL (LSTM)', linestyle='--')
plt.xlabel('Test engines (sorted by actual RUL)'); plt.ylabel('RUL (cycles)')
plt.title(f'Predicted vs Actual RUL (RMSE = {lstm_rmse:.2f})')
plt.legend(); plt.tight_layout()
plt.savefig('pred_vs_actual.png', dpi=200); plt.close()

plt.figure(figsize=(4, 4))
plt.bar(['Random Forest', 'LSTM'], [rf_rmse, lstm_rmse], color=['gray', 'tab:blue'])
plt.ylabel('RMSE'); plt.title('Model Comparison')
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=200); plt.close()

print("Saved: loss_curve.png, pred_vs_actual.png, model_comparison.png")
