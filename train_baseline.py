import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb

def load_and_preprocess_data(data_dir="./ev_dataset"):
    print(" Loading dataset tables...")
    sessions_path = os.path.join(data_dir, "ev_charging_sessions.csv")
    vehicles_path = os.path.join(data_dir, "ev_vehicles.csv")
    
    if not os.path.exists(sessions_path) or not os.path.exists(vehicles_path):
        raise FileNotFoundError(f"Dataset files not found in '{data_dir}'. Run generate.py first.")
        
    df_sessions = pd.read_csv(sessions_path)
    df_vehicles = pd.read_csv(vehicles_path)
    
    print(f"  ✓ Loaded sessions: {len(df_sessions):,} rows")
    print(f"  ✓ Loaded vehicles: {len(df_vehicles):,} rows")
    
    # Merge tables on vehicle_id
    print(" Merging sessions and vehicle specifications...")
    df = pd.merge(df_sessions, df_vehicles, on="vehicle_id", how="inner")
    
    # Feature Engineering from Timestamps
    df['session_start_time'] = pd.to_datetime(df['session_start_time'])
    df['hour_of_day'] = df['session_start_time'].dt.hour
    df['day_of_week'] = df['session_start_time'].dt.dayofweek
    df['month'] = df['session_start_time'].dt.month
    
    # Sort chronologically to prevent temporal data leakage during split
    df = df.sort_values('session_start_time').reset_index(drop=True)
    
    return df

def train_energy_prediction_model():
    # 1. Load data
    df = load_and_preprocess_data()
    
    # 2. Define Features and Target
    # CRITICAL: Exclude post-session targets like soc_end_pct, session_duration_minutes, 
    # or energy_drawn_from_grid_kwh to prevent data leakage!
    features = [
        'soc_start_pct',
        'battery_capacity_kwh',
        'usable_battery_capacity_kwh',
        'rated_range_km',
        'vehicle_efficiency_kwh_per_100km',
        'maximum_ac_charging_power_kw',
        'maximum_dc_charging_power_kw',
        'initial_battery_soh_pct',
        'ambient_temperature_c',
        'electricity_price_per_kwh',
        'hour_of_day',
        'day_of_week',
        'month'
    ]
    
    target = 'energy_delivered_kwh'
    
    X = df[features]
    y = df[target]
    
    # Encode categorical columns if any exist in features
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = X[col].astype('category')

    # 3. Chronological Train-Test Split (70% Train, 15% Validation, 15% Test)
    print(" Splitting data chronologically...")
    train_idx = int(len(df) * 0.70)
    val_idx = int(len(df) * 0.85)
    
    X_train, y_train = X.iloc[:train_idx], y.iloc[:train_idx]
    X_val, y_val = X.iloc[train_idx:val_idx], y.iloc[train_idx:val_idx]
    X_test, y_test = X.iloc[val_idx:], y.iloc[val_idx:]
    
    print(f"  - Training samples:   {len(X_train):,}")
    print(f"  - Validation samples: {len(X_val):,}")
    print(f"  - Test samples:       {len(X_test):,}")

    # 4. Initialize and Train LightGBM Regressor
    print(" Training LightGBM Regressor model...")
    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )

    # 5. Evaluate on Test Set
    print(" Evaluating model on test set...")
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("\n" + "="*40)
    print("  MODEL PERFORMANCE RESULTS ")
    print("="*40)
    print(f"  - Mean Absolute Error (MAE): {mae:.4f} kWh")
    print(f"  - Root Mean Squared Error (RMSE): {rmse:.4f} kWh")
    print(f"  - R-Squared Score (R²): {r2:.4f}")
    print("="*40)
    
    # 6. Feature Importance
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n Top Predictive Features:")
    print(importance.head(5).to_string(index=False))

if __name__ == "__main__":
    train_energy_prediction_model()
