from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import os
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Predefined coefficients for multiLinReg (for PM2.5, PM10, NO2)
lrCoef = [25.142174108944573, 0.48236037, 0.0356627, -0.04208933]

def multiLinReg(pm25, pm10, no2):
    """
    Calculate the AQI using PM2.5, PM10, NO2 and coefficients.
    """
    return lrCoef[0] + pm25 * lrCoef[1] + pm10 * lrCoef[2] + no2 * lrCoef[3]

def load_data(csv_path='data.csv'):
    """
    Load CSV data, select PM2.5, PM10, NO2, compute AQI, and return DataFrame.
    """
    try:
        # Construct absolute path to data.csv
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_full_path = os.path.join(base_dir, csv_path)
        logger.info(f"Attempting to load CSV from: {csv_full_path}")
        
        if not os.path.exists(csv_full_path):
            logger.error(f"CSV file not found at: {csv_full_path}")
            return None
            
        # Load only the required columns
        df = pd.read_csv(csv_full_path, usecols=['PM2.5 (ug/m3)', 'PM10 (ug/m3)', 'NO2 (ug/m3)'])
        # Drop rows with any NaN values in the selected columns
        df = df.dropna()
        # Calculate AQI using the multiLinReg function
        df['AQI'] = df.apply(lambda row: multiLinReg(
            row['PM2.5 (ug/m3)'], row['PM10 (ug/m3)'], row['NO2 (ug/m3)']), axis=1)
        logger.info("CSV loaded successfully")
        return df
    except Exception as e:
        logger.error(f"Error loading CSV file: {str(e)}")
        return None

def train_model(df):
    """
    Train a linear regression model using PM2.5, PM10, NO2.
    """
    X = df[['PM2.5 (ug/m3)', 'PM10 (ug/m3)', 'NO2 (ug/m3)']]
    y = df['AQI']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Load data and train the model at startup
df = load_data()
if df is not None:
    model = train_model(df)
else:
    model = None
    logger.error("Model training skipped due to missing or invalid data")

@app.route('/')
def index():
    """
    Render a simple homepage for inputting PM2.5, PM10, NO2 values.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to predict AQI from PM2.5, PM10, NO2 inputs.
    Expects JSON payload with keys: pm25, pm10, no2.
    """
    if model is None:
        logger.error("Prediction failed: Model not loaded")
        return jsonify({'error': 'Model not loaded properly. Check if data.csv is available.'}), 500
    try:
        data = request.get_json()
        pm25 = float(data['pm25'])
        pm10 = float(data['pm10'])
        no2 = float(data['no2'])
        input_features = np.array([[pm25, pm10, no2]])
        prediction = model.predict(input_features)[0]
        logger.info(f"Prediction successful: AQI={prediction}")
        return jsonify({'predicted_aqi': prediction})
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)