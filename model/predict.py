import sqlite3
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def train_and_predict(db_path="database/stocks.db", table_name="stocks"):
    """
    Trains a Random Forest Classifier to predict if tomorrow's stock price goes UP or DOWN.
    Returns model metrics, predictions for the next day, and feature importances.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at: {db_path}. Run ETL first.")
        
    print(f"Reading data from database table '{table_name}'...")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table_name} ORDER BY Date ASC", conn)
    conn.close()
    
    # 1. Feature Engineering (stationary/normalized metrics)
    df['Close_to_SMA_5'] = (df['Close'] / df['SMA_5']) - 1
    df['Close_to_SMA_20'] = (df['Close'] / df['SMA_20']) - 1
    
    # Target: 1 if Tomorrow's Close > Today's Close, else 0
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Define features
    feature_cols = [
        'Return_Lag_1', 'Return_Lag_2', 'Return_Lag_3', 'Return_Lag_4', 'Return_Lag_5',
        'Volume_Change', 'Close_to_SMA_5', 'Close_to_SMA_20'
    ]
    
    # The last row has features but Target is NaN (since we don't know tomorrow's close yet)
    # We will use the last row to predict tomorrow's movement
    last_row = df.iloc[-1]
    
    # Check if last row has valid features
    if last_row[feature_cols].isna().any():
        # Fallback if somehow last row has NaNs
        last_row_valid = df.dropna(subset=feature_cols).iloc[-1]
    else:
        last_row_valid = last_row
        
    # Dataset for training and testing (drop NaN targets & features)
    df_ml = df.dropna(subset=feature_cols + ['Target'])
    
    X = df_ml[feature_cols]
    y = df_ml['Target']
    
    # 2. Sequential Train/Test Split (80% train, 20% test to prevent time-leakage)
    split_idx = int(len(df_ml) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Training set size: {len(X_train)} rows")
    print(f"Testing set size: {len(X_test)} rows")
    
    # 3. Model Training
    # Limit max_depth to prevent overfitting on small financial dataset
    model = RandomForestClassifier(n_estimators=150, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Model Evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred)
    
    # 5. Predict Tomorrow (next day direction)
    X_tomorrow = pd.DataFrame([last_row_valid[feature_cols]], columns=feature_cols)
    tomorrow_pred = int(model.predict(X_tomorrow)[0])
    tomorrow_prob = model.predict_proba(X_tomorrow)[0] # [prob_down, prob_up]
    
    confidence = tomorrow_prob[tomorrow_pred] * 100
    
    # 6. Feature Importances
    importances = model.feature_importances_
    features_df = pd.DataFrame({
        'Feature': [f.replace('_', ' ') for f in feature_cols],
        'Importance': importances
    }).sort_values(by='Importance', ascending=True)
    
    # Format confusion matrix for display
    # Confusion matrix is [[TN, FP], [FN, TP]]
    # Let's turn it into a readable structure:
    conf_df = pd.DataFrame(
        conf_mat,
        index=['Actual DOWN', 'Actual UP'],
        columns=['Predicted DOWN', 'Predicted UP']
    )
    
    print("--- ML Run Summary ---")
    print(f"Test Accuracy: {accuracy:.2%}")
    print(f"Tomorrow's Prediction: {'UP' if tomorrow_pred == 1 else 'DOWN'} (Confidence: {confidence:.2f}%)")
    print("Confusion Matrix:")
    print(conf_df)
    
    return {
        'accuracy': accuracy,
        'prediction': 'UP' if tomorrow_pred == 1 else 'DOWN',
        'confidence': confidence,
        'confusion_matrix': conf_df,
        'feature_importance': features_df,
        'last_date': last_row_valid['Date']
    }

if __name__ == "__main__":
    train_and_predict()
