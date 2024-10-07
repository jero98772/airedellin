import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the PM2.5 data.
    """
    data = pd.read_csv(file_path)
    data['coord'] = data['lat'].astype(str) + ', ' + data['lon'].astype(str)
    
    # Filter coordinates within Colombia
    unique_coords = data[['lat', 'lon']].drop_duplicates()
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    colombia = world[world['name'] == 'Colombia']
    unique_coords_gdf = gpd.GeoDataFrame(unique_coords, geometry=gpd.points_from_xy(unique_coords.lon, unique_coords.lat))
    coords_in_colombia = gpd.sjoin(unique_coords_gdf, colombia)
    coords_in_colombia['coord'] = coords_in_colombia['lat'].astype(str) + ', ' + coords_in_colombia['lon'].astype(str)
    
    df = data[data['coord'].isin(coords_in_colombia['coord'])].drop('Unnamed: 0', axis=1)
    df['timestamp'] = pd.to_datetime(df['time'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    
    le = LabelEncoder()
    df['location_encoded'] = le.fit_transform(df['coord'])
    df.rename(columns={'MERRA2_CNN_Surface_PM25': 'pm25'}, inplace=True)
    
    return df, le

def create_lag_features(df):
    """
    Create lag features for the PM2.5 data.
    """
    df['pm25_lag1'] = df.groupby('location_encoded')['pm25'].shift(1)
    df['pm25_lag2'] = df.groupby('location_encoded')['pm25'].shift(2)
    df['pm25_lag3'] = df.groupby('location_encoded')['pm25'].shift(3)
    df['pm25_lagyear'] = df.groupby('location_encoded')['pm25'].shift(24*365)
    return df.dropna()

def prepare_model_data(df):
    """
    Prepare data for model training and testing.
    """
    features = ['hour', 'day_of_week', 'month', 'location_encoded', 'pm25_lagyear', 'pm25_lag3', 'pm25_lag2', 'pm25_lag1']
    X = df[features]
    y = df['pm25']
    
    split_index = int(len(df) * 0.8)
    train_mask = df.index < split_index
    test_mask = df.index >= split_index
    
    return X[train_mask], X[test_mask], y[train_mask], y[test_mask], features

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    """
    Train and evaluate the XGBoost model.
    """
    model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train, verbose=1)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    return model, rmse

def prepare_future_data(df, last_date, future_periods, location_encoded):
    """
    Prepare data for future predictions.
    """
    future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=4*4, days=15), periods=future_periods, freq='h')
    future_df = pd.DataFrame({'timestamp': future_dates})
    future_df['hour'] = future_df['timestamp'].dt.hour
    future_df['day_of_week'] = future_df['timestamp'].dt.dayofweek
    future_df['month'] = future_df['timestamp'].dt.month
    future_df['location_encoded'] = location_encoded
    future_df['pm25_lag1'] = df['pm25'].iloc[-1]
    future_df['pm25_lag2'] = df['pm25'].iloc[-2]
    future_df['pm25_lag3'] = df['pm25'].iloc[-3]
    future_df['pm25_lagyear'] = df['pm25'].iloc[-24*365:]
    
    return future_df

def make_future_predictions(model, future_df, features):
    """
    Make predictions for future dates.
    """
    future_predictions = []
    
    for i in range(len(future_df)):
        X_future = future_df.iloc[i:i+1][features]
        pred = model.predict(X_future)[0]
        future_predictions.append(pred)
        
        if i + 1 < len(future_df):
            future_df.loc[future_df.index[i+1], 'pm25_lag1'] = pred
            if i + 2 < len(future_df):
                future_df.loc[future_df.index[i+2], 'pm25_lag2'] = pred
            if i + 3 < len(future_df):
                future_df.loc[future_df.index[i+3], 'pm25_lag3'] = pred
    
    future_df['predicted_pm25'] = future_predictions
    return future_df

def plot_predictions(future_df):
    """
    Plot the predicted PM2.5 values.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=future_df.timestamp,
        y=future_df.predicted_pm25,
        mode='lines',
        name='Predicted',
        line=dict(color='blue'),
        marker=dict(symbol='circle', size=8, color='blue')
    ))
    fig.update_layout(title='PM2.5 Predictions', xaxis_title='Date', yaxis_title='PM2.5')
    return fig

def main():
    # Load and preprocess data
    df, location_encoder = load_and_preprocess_data('2017-2024 Col.csv')
    
    # Create lag features
    df = create_lag_features(df)
    
    # Prepare data for model
    X_train, X_test, y_train, y_test, features = prepare_model_data(df)
    
    # Train and evaluate model
    model, rmse = train_and_evaluate_model(X_train, X_test, y_train, y_test)
    print(f"RMSE: {rmse}")
    
    # Prepare future data
    last_date = df['timestamp'].max()
    future_df = prepare_future_data(df, last_date, future_periods=168, location_encoded=200)
    
    # Make future predictions
    future_df = make_future_predictions(model, future_df, features)
    
    # Plot predictions
    fig = plot_predictions(future_df)
    fig.show()
    
    # Display the first few predictions
    print(future_df[['timestamp', 'predicted_pm25']].head())

if __name__ == "__main__":
    main()