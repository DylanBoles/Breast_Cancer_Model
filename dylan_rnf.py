######### BEST MODEL ###############

## Final Project Model (Training on Combined Train+Dev) ##

# Machine Learning Spring

# Import arguments
import sys
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# --- Load Data ---
train_csv = f'../../FullData/train.csv'
dev_csv = f'../../FullData/dev.csv'
eval_csv = f'../../FullData/eval.csv'

print("Loading data files...")

def load_data(file_path, is_eval=False):
    """Load data with correct format."""
    try:
        df = pd.read_csv(file_path, skiprows=1, header=None)
        print(f"Loaded {os.path.basename(file_path)} with shape: {df.shape}")
        if not is_eval and not all(label in range(9) for label in df.iloc[:, 0].unique()):
            raise ValueError("Invalid labels found. Labels must be in range 0-8")
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        raise

# Load datasets
print("\nLoading training data...")
train_df = load_data(train_csv)
print("\nLoading dev data...")
dev_df = load_data(dev_csv)
print("\nLoading eval data...")
eval_df = load_data(eval_csv, is_eval=True)

def prepare_data(df):
    """Prepare features and labels."""
    X = df.iloc[:, 1:].values  # 3072 features
    y = df.iloc[:, 0].values.astype(int)  # Labels must be integers
    return X, y

print("\nPreparing datasets...")
X_train, y_train = prepare_data(train_df)
X_dev_eval, y_dev_eval = prepare_data(dev_df) # Keep the original dev set for evaluation
X_eval = prepare_data(eval_df)[0] # Only features for evaluation

# --- Combine Training and Development Data for Training ---
X_train_combined = np.vstack((X_train, X_dev_eval))
y_train_combined = np.hstack((y_train, y_dev_eval))

# Normalize features using only the combined training data stats
scaler = StandardScaler()
X_train_combined = scaler.fit_transform(X_train_combined)
X_dev_eval_scaled = scaler.transform(X_dev_eval) # Scale the original dev set
X_eval_scaled = scaler.transform(X_eval)

print("\nTraining Random Forest model on combined training and development data...")

# Using parameters that should help prevent overfitting
model = RandomForestClassifier(
    n_estimators=800,
    max_depth=12,
    min_samples_split=3,
    min_samples_leaf=6,
    random_state=14,
    n_jobs=-1,
    class_weight='balanced',
    max_features='sqrt'
)

model.fit(X_train_combined, y_train_combined)

print("\nMaking predictions on the development set...")
dev_pred = model.predict(X_dev_eval_scaled)

# Calculate accuracy on the original development set
dev_acc = accuracy_score(y_dev_eval, dev_pred)
print(f"\nDevelopment accuracy (trained on combined): {dev_acc*100:.2f}%")

print("\nMaking predictions on the evaluation set...")
eval_pred = model.predict(X_eval_scaled)

print("\nSaving predictions...")
output_dir = f'./prediction'
os.makedirs(output_dir, exist_ok=True)

# Save predictions with header 'label'
for name, preds in [('train', model.predict(X_train_combined)), ('dev', dev_pred), ('eval', eval_pred)]:
    output_file = f'{output_dir}/{name}_predictions.csv'
    preds = np.clip(preds, 0, 8).astype(int)
    pd.DataFrame({'Label': preds}).to_csv(output_file, index=False)
    print(f"Saved {name} predictions to: {output_file}")

print("\nPrediction distributions (Development Set):")
dist_dev = pd.Series(dev_pred).value_counts().sort_index()
print(dist_dev)

print("\nProcess completed!")