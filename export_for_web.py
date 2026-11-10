# Run this in a NEW cell in Colab, right AFTER the main project cell has finished.
# It saves your trained LSTM + test engine data into model_data.js for the website.

import json

def r(a, d=6):
    return np.round(np.array(a, dtype=float), d).tolist()

engines = []
for i, (unit, g) in enumerate(test.groupby('unit')):
    engines.append({
        "unit": int(unit),
        "seq": r(g[features].values, 4),      # all cycles, scaled sensors
        "rul": int(true_rul[i]),              # true RUL at the last cycle
        "pred": float(lstm_pred[i]),          # Keras prediction at the last cycle
    })

data = {
    "window": WINDOW,
    "cap": RUL_CAP,
    "lstm_rmse": float(lstm_rmse),
    "rf_rmse": float(rf_rmse),
    "w": [r(w) for w in model.get_weights()],  # LSTM1, LSTM2, Dense, Dense weights
    "engines": engines,
}

with open('model_data.js', 'w') as f:
    f.write('const MODEL_DATA = ' + json.dumps(data) + ';')

print("Saved model_data.js | LSTM RMSE:", round(lstm_rmse, 2), "| RF RMSE:", round(rf_rmse, 2))

try:
    from google.colab import files
    files.download('model_data.js')
except ImportError:
    pass
