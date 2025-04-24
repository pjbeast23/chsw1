from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import logging
import json

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# AQI Breakpoints as per EPA standards
# Format: [C_low, C_high, I_low, I_high]
# PM2.5 (μg/m³, 24-hour average)
PM25_BREAKPOINTS = [
    [0.0, 12.0, 0, 50],
    [12.1, 35.4, 51, 100],
    [35.5, 55.4, 101, 150],
    [55.5, 150.4, 151, 200],
    [150.5, 250.4, 201, 300],
    [250.5, 500.4, 301, 500]
]

# PM10 (μg/m³, 24-hour average)
PM10_BREAKPOINTS = [
    [0, 54, 0, 50],
    [55, 154, 51, 100],
    [155, 254, 101, 150],
    [255, 354, 151, 200],
    [355, 424, 201, 300],
    [425, 604, 301, 500]
]

# NO2 (ppb, 1-hour average)
NO2_BREAKPOINTS = [
    [0, 53, 0, 50],
    [54, 100, 51, 100],
    [101, 360, 101, 150],
    [361, 649, 151, 200],
    [650, 1249, 201, 300],
    [1250, 2049, 301, 500]
]

# AQI Categories and health recommendations
AQI_CATEGORIES = [
    {
        "name": "Good",
        "range": [0, 50],
        "description": "Air quality is satisfactory, and air pollution poses little or no risk."
    },
    {
        "name": "Moderate",
        "range": [51, 100],
        "description": "Air quality is acceptable. However, there may be a risk for some people, particularly those who are unusually sensitive to air pollution."
    },
    {
        "name": "Unhealthy for Sensitive Groups",
        "range": [101, 150],
        "description": "Members of sensitive groups may experience health effects. The general public is less likely to be affected."
    },
    {
        "name": "Unhealthy",
        "range": [151, 200],
        "description": "Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects."
    },
    {
        "name": "Very Unhealthy",
        "range": [201, 300],
        "description": "Health alert: The risk of health effects is increased for everyone."
    },
    {
        "name": "Hazardous",
        "range": [301, 500],
        "description": "Health warning of emergency conditions: everyone is more likely to be affected."
    }
]

def calculate_aqi(concentration, breakpoints):
    """
    Calculate AQI using EPA formula:
    AQI = ((I_high - I_low) / (C_high - C_low)) * (C - C_low) + I_low
    
    Where:
    C = input concentration
    C_low = concentration breakpoint ≤ C
    C_high = concentration breakpoint ≥ C
    I_low = index breakpoint corresponding to C_low
    I_high = index breakpoint corresponding to C_high
    """
    for bp in breakpoints:
        c_low, c_high, i_low, i_high = bp
        if c_low <= concentration <= c_high:
            aqi = ((i_high - i_low) / (c_high - c_low)) * (concentration - c_low) + i_low
            return round(aqi)
    
    # If concentration is higher than the highest breakpoint
    if concentration > breakpoints[-1][1]:
        c_low, c_high, i_low, i_high = breakpoints[-1]
        aqi = ((i_high - i_low) / (c_high - c_low)) * (concentration - c_low) + i_low
        return round(min(aqi, 500))  # Cap at 500
    
    # If concentration is lower than the lowest breakpoint
    return 0

def get_aqi_from_pollutants(pm25, pm10, no2):
    """
    Calculate individual AQI values for each pollutant and return the highest one
    """
    aqi_pm25 = calculate_aqi(pm25, PM25_BREAKPOINTS)
    aqi_pm10 = calculate_aqi(pm10, PM10_BREAKPOINTS)
    aqi_no2 = calculate_aqi(no2, NO2_BREAKPOINTS)
    
    # Return the highest AQI value (EPA uses the "dominant pollutant" approach)
    aqi = max(aqi_pm25, aqi_pm10, aqi_no2)
    
    # Get category and description
    category = None
    description = None
    for cat in AQI_CATEGORIES:
        if cat["range"][0] <= aqi <= cat["range"][1]:
            category = cat["name"]
            description = cat["description"]
            break
    
    return {
        "aqi": aqi,
        "category": category,
        "description": description,
        "pollutant_values": {
            "pm25": aqi_pm25,
            "pm10": aqi_pm10,
            "no2": aqi_no2
        }
    }

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
        
        # Calculate AQI for each row using EPA method
        aqi_data = []
        for _, row in df.iterrows():
            aqi_info = get_aqi_from_pollutants(
                row['PM2.5 (ug/m3)'], 
                row['PM10 (ug/m3)'], 
                row['NO2 (ug/m3)']
            )
            aqi_data.append(aqi_info['aqi'])
        
        df['AQI'] = aqi_data
        logger.info("CSV loaded successfully")
        return df
    except Exception as e:
        logger.error(f"Error loading CSV file: {str(e)}")
        return None

# No need for a trained model since we're using standard formulas
@app.route('/')
def index():
    """
    Render the homepage for inputting pollution parameters.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to predict AQI from PM2.5, PM10, NO2 inputs.
    Expects JSON payload with keys: pm25, pm10, no2.
    """
    try:
        data = request.get_json()
        pm25 = float(data['pm25'])
        pm10 = float(data['pm10'])
        no2 = float(data['no2'])
        
        # Input validation
        if pm25 < 0 or pm10 < 0 or no2 < 0:
            return jsonify({'error': 'Pollutant values cannot be negative'}), 400
            
        # Calculate AQI using EPA method
        aqi_result = get_aqi_from_pollutants(pm25, pm10, no2)
        
        logger.info(f"Prediction successful: AQI={aqi_result['aqi']}, Category={aqi_result['category']}")
        return jsonify({
            'predicted_aqi': aqi_result['aqi'],
            'category': aqi_result['category'],
            'description': aqi_result['description'],
            'pollutant_values': aqi_result['pollutant_values']
        })
    except KeyError as e:
        logger.error(f"Missing parameter in request: {str(e)}")
        return jsonify({'error': f'Missing parameter: {str(e)}'}), 400
    except ValueError as e:
        logger.error(f"Invalid value in request: {str(e)}")
        return jsonify({'error': f'Invalid value: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/aqi-info', methods=['GET'])
def aqi_info():
    """
    Return AQI categories and descriptions
    """
    return jsonify(AQI_CATEGORIES)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)

