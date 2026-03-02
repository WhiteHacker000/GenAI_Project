import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def train():
    print("Loading data...")
    df = pd.read_csv('data/EVChargingStationUsage.csv')

    print("Cleaning data...")
    df['Start Date'] = pd.to_datetime(df['Start Date'])
    df['Hour'] = df['Start Date'].dt.hour
    df['Weekday'] = df['Start Date'].dt.weekday
    
    # Remove rows with zero energy
    df = df[df['Energy (kWh)'] > 0]
    
    # Keep only necessary columns based on notebook logic
    df = df[['Station Name', 'Energy (kWh)', 'Hour', 'Weekday']]
    
    print("Aggregating data...")
    df_grouped = df.groupby(['Station Name', 'Hour', 'Weekday'])['Energy (kWh)'].sum().reset_index()
    
    print("Engineering features...")
    df_grouped['Peak_Hour'] = df_grouped['Hour'].apply(
        lambda x: 1 if (6 <= x <= 10) or (17 <= x <= 21) else 0
    )
    
    df_grouped['Is_Weekend'] = df_grouped['Weekday'].apply(
        lambda x: 1 if x >= 5 else 0
    )
    
    # Calculate and save Station Average Load mapping
    station_avg = df_grouped.groupby('Station Name')['Energy (kWh)'].mean().to_dict()
    df_grouped['Station_Avg_Load'] = df_grouped['Station Name'].map(station_avg)
    
    df_grouped['Hour_Weekend_Interaction'] = df_grouped['Hour'] * df_grouped['Is_Weekend']
    
    # One-hot encode Station Name
    df_grouped = pd.get_dummies(df_grouped, columns=['Station Name'], drop_first=True)
    
    # Define features and target
    X = df_grouped.drop(columns=['Energy (kWh)'])
    y = df_grouped['Energy (kWh)']
    
    # Save the feature columns so the web app can recreate the exact same dataframe structure
    feature_columns = list(X.columns)
    
    print("Training model...")
    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    print("Saving artifacts...")
    artifacts = {
        'model': rf,
        'station_avg_map': station_avg,
        'feature_columns': feature_columns
    }
    
    joblib.dump(artifacts, 'model_artifacts.pkl')
    print("Done! Artifacts saved to model_artifacts.pkl")

if __name__ == "__main__":
    import os
    os.chdir(os.path.join(os.path.dirname(__file__), '..'))
    train()


