from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import os
import logging
import json
import datetime
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from io import BytesIO
import base64

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

# Weather condition health impacts based on temperature and humidity
WEATHER_IMPACTS = {
    "hot_dry": {
        "condition": "Hot and Dry",
        "description": "Hot, dry conditions can cause pollutants to concentrate and increase irritation to respiratory systems.",
        "threshold": {"temp_min": 30, "humidity_max": 40}
    },
    "hot_humid": {
        "condition": "Hot and Humid",
        "description": "Hot, humid conditions can trap pollutants near the ground and exacerbate breathing difficulties.",
        "threshold": {"temp_min": 30, "humidity_min": 70}
    },
    "cold_dry": {
        "condition": "Cold and Dry",
        "description": "Cold, dry air can irritate airways and worsen effects of pollutants, especially for those with asthma.",
        "threshold": {"temp_max": 10, "humidity_max": 40}
    },
    "cold_humid": {
        "condition": "Cold and Humid",
        "description": "Cold, humid conditions can promote mold growth and increase risk for respiratory issues.",
        "threshold": {"temp_max": 10, "humidity_min": 70}
    },
    "moderate": {
        "condition": "Moderate",
        "description": "Moderate temperature and humidity conditions typically have minimal impact on air quality health effects.",
        "threshold": {}  # Default condition
    }
}

# Health recommendations based on combined AQI and weather factors
def get_health_recommendations(aqi, temperature, humidity):
    """Generate specific health recommendations based on AQI and weather conditions"""
    
    # Determine weather condition
    weather_condition = "moderate"  # Default
    
    if temperature >= 30:
        if humidity <= 40:
            weather_condition = "hot_dry"
        elif humidity >= 70:
            weather_condition = "hot_humid"
    elif temperature <= 10:
        if humidity <= 40:
            weather_condition = "cold_dry"
        elif humidity >= 70:
            weather_condition = "cold_humid"
    
    weather_impact = WEATHER_IMPACTS[weather_condition]
    
    # Base recommendations on AQI category
    if aqi <= 50:
        base_rec = "It's a great day for outdoor activities."
    elif aqi <= 100:
        base_rec = "Consider reducing prolonged outdoor exertion if you're sensitive to air pollution."
    elif aqi <= 150:
        base_rec = "People with respiratory or heart conditions, the elderly and children should limit prolonged outdoor exertion."
    elif aqi <= 200:
        base_rec = "Everyone should limit outdoor exertion, especially those with respiratory conditions."
    elif aqi <= 300:
        base_rec = "Everyone should avoid outdoor activities. If possible, remain indoors with air purification."
    else:
        base_rec = "Health emergency: Everyone should avoid all outdoor physical activities and remain indoors."
    
    # Add weather-specific recommendation
    weather_rec = weather_impact["description"]
    
    # Add PM2.5 specific advice if it's high (over 35.4 μg/m³)
    pm25_rec = ""
    if aqi > 100:  # When AQI is above 100, PM2.5 might be significant
        pm25_rec = " Fine particulate matter (PM2.5) may penetrate deep into lungs; consider using a mask if going outdoors."
    
    return {
        "general": base_rec,
        "weather_condition": weather_impact["condition"],
        "weather_impact": weather_rec,
        "combined": f"{base_rec} {weather_rec}{pm25_rec}"
    }

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
    Load CSV data, compute AQI, and return DataFrame with target columns.
    """
    try:
        # Construct absolute path to data.csv
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_full_path = os.path.join(base_dir, csv_path)
        logger.info(f"Attempting to load CSV from: {csv_full_path}")
        
        if not os.path.exists(csv_full_path):
            logger.error(f"CSV file not found at: {csv_full_path}")
            return None
            
        # Load columns we need for our enhanced analysis
        df = pd.read_csv(csv_full_path, parse_dates=['From Date', 'To Date'])
        
        # We'll focus on PM2.5, Temperature and Humidity
        # But need to keep PM10 and NO2 for AQI calculation
        cols = ['From Date', 'To Date', 'PM2.5 (ug/m3)', 'PM10 (ug/m3)', 'NO2 (ug/m3)', 
                'Temp (degree C)', 'RH (%)', 'WS (m/s)', 'RF (mm)']
        
        df_filtered = df[cols].copy()
        # Convert column names to more convenient formats
        df_filtered.columns = ['from_date', 'to_date', 'pm25', 'pm10', 'no2', 
                              'temperature', 'humidity', 'wind_speed', 'rainfall']
        
        # Drop rows with NaN in essential columns
        required_cols = ['pm25', 'pm10', 'no2', 'temperature', 'humidity']
        df_filtered = df_filtered.dropna(subset=required_cols)
        
        # Calculate AQI for each row
        aqi_data = []
        for _, row in df_filtered.iterrows():
            aqi_info = get_aqi_from_pollutants(
                row['pm25'], 
                row['pm10'], 
                row['no2']
            )
            aqi_data.append(aqi_info['aqi'])
        
        df_filtered['aqi'] = aqi_data
        
        # Add date only and hour columns for time analysis
        df_filtered['date'] = df_filtered['from_date'].dt.date
        df_filtered['hour'] = df_filtered['from_date'].dt.hour
        
        logger.info(f"CSV loaded successfully: {len(df_filtered)} valid rows")
        return df_filtered
    except Exception as e:
        logger.error(f"Error loading CSV file: {str(e)}")
        return None

# Load data once at startup
air_quality_data = load_data()
prediction_model = None
scaler = None

def train_prediction_model():
    """
    Train a simple prediction model for PM2.5 based on temperature and humidity
    """
    global prediction_model, scaler, air_quality_data
    
    if air_quality_data is None or len(air_quality_data) < 10:
        logger.warning("Not enough data to train model")
        return None
    
    try:
        # Features: temperature, humidity
        X = air_quality_data[['temperature', 'humidity']].values
        # Target: PM2.5
        y = air_quality_data['pm25'].values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        logger.info(f"Model trained - R² train: {train_score:.3f}, R² test: {test_score:.3f}")
        
        prediction_model = model
        return {
            "train_score": train_score,
            "test_score": test_score,
            "coefficients": model.coef_.tolist(),
            "intercept": model.intercept_
        }
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        return None

# Train model at startup if data is available
if air_quality_data is not None and len(air_quality_data) > 0:
    train_prediction_model()

def create_correlation_plot():
    """Create a correlation heatmap between PM2.5, temperature, and humidity"""
    global air_quality_data
    
    if air_quality_data is None or len(air_quality_data) == 0:
        return None
    
    try:
        # Calculate correlation matrix
        corr_data = air_quality_data[['pm25', 'temperature', 'humidity']].corr()
        
        # Create heatmap
        plt.figure(figsize=(8, 6))
        plt.imshow(corr_data, cmap='coolwarm', interpolation='none', aspect='auto')
        plt.colorbar(label='Correlation coefficient')
        
        # Add correlation values
        for i in range(len(corr_data)):
            for j in range(len(corr_data)):
                plt.text(j, i, f"{corr_data.iloc[i, j]:.2f}", 
                         ha="center", va="center", color="black")
        
        # Labels
        plt.xticks(range(len(corr_data)), corr_data.columns, rotation=45)
        plt.yticks(range(len(corr_data)), corr_data.columns)
        plt.title('Correlation Between PM2.5, Temperature and Humidity')
        
        # Save to bytes buffer
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        
        # Convert to base64 for embedding in HTML
        image_png = buffer.getvalue()
        buffer.close()
        encoded = base64.b64encode(image_png).decode('utf-8')
        
        plt.close()  # Close plot to prevent memory leak
        return encoded
    except Exception as e:
        logger.error(f"Error creating correlation plot: {str(e)}")
        return None

def create_time_series_plot():
    """Create a time series plot of PM2.5, temperature, and humidity"""
    global air_quality_data
    
    if air_quality_data is None or len(air_quality_data) == 0:
        return None
    
    try:
        # Get last 48 hours of data if available
        df_plot = air_quality_data.sort_values('from_date').tail(48)
        
        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        
        # PM2.5 plot
        ax1.plot(df_plot['from_date'], df_plot['pm25'], 'b-', label='PM2.5')
        ax1.set_ylabel('PM2.5 (μg/m³)')
        ax1.set_title('PM2.5, Temperature and Humidity Time Series')
        ax1.grid(True)
        
        # Temperature plot
        ax2.plot(df_plot['from_date'], df_plot['temperature'], 'r-', label='Temperature')
        ax2.set_ylabel('Temperature (°C)')
        ax2.grid(True)
        
        # Humidity plot
        ax3.plot(df_plot['from_date'], df_plot['humidity'], 'g-', label='Humidity')
        ax3.set_ylabel('Humidity (%)')
        ax3.set_xlabel('Date')
        ax3.grid(True)
        
        # Format x-axis
        plt.xticks(rotation=45)
        fig.tight_layout()
        
        # Save to bytes buffer
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        
        # Convert to base64 for embedding in HTML
        image_png = buffer.getvalue()
        buffer.close()
        encoded = base64.b64encode(image_png).decode('utf-8')
        
        plt.close()  # Close plot to prevent memory leak
        return encoded
    except Exception as e:
        logger.error(f"Error creating time series plot: {str(e)}")
        return None

def create_hourly_pattern_plot():
    """Create a plot showing hourly patterns of PM2.5, temperature, and humidity"""
    global air_quality_data
    
    if air_quality_data is None or len(air_quality_data) == 0:
        return None
    
    try:
        # Group by hour and calculate mean
        hourly_avg = air_quality_data.groupby('hour')[['pm25', 'temperature', 'humidity']].mean()
        
        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        
        # PM2.5 plot
        ax1.plot(hourly_avg.index, hourly_avg['pm25'], 'bo-', label='PM2.5')
        ax1.set_ylabel('PM2.5 (μg/m³)')
        ax1.set_title('Average Hourly Patterns')
        ax1.grid(True)
        
        # Temperature plot
        ax2.plot(hourly_avg.index, hourly_avg['temperature'], 'ro-', label='Temperature')
        ax2.set_ylabel('Temperature (°C)')
        ax2.grid(True)
        
        # Humidity plot
        ax3.plot(hourly_avg.index, hourly_avg['humidity'], 'go-', label='Humidity')
        ax3.set_ylabel('Humidity (%)')
        ax3.set_xlabel('Hour of Day')
        ax3.grid(True)
        
        # Set x-axis ticks to be 0-23 hours
        plt.xticks(range(0, 24))
        fig.tight_layout()
        
        # Save to bytes buffer
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        
        # Convert to base64 for embedding in HTML
        image_png = buffer.getvalue()
        buffer.close()
        encoded = base64.b64encode(image_png).decode('utf-8')
        
        plt.close()  # Close plot to prevent memory leak
        return encoded
    except Exception as e:
        logger.error(f"Error creating hourly pattern plot: {str(e)}")
        return None

@app.route('/')
def index():
    """
    Render the homepage for inputting pollution parameters.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to predict AQI from PM2.5, humidity, and temperature inputs.
    """
    try:
        data = request.get_json()
        pm25 = float(data['pm25'])
        humidity = float(data['humidity']) if 'humidity' in data else float(data['pm10'])  # Map pm10 to humidity
        temperature = float(data['temperature']) if 'temperature' in data else float(data['no2'])  # Map no2 to temperature
        
        # For backward compatibility - if old names are used
        pm10 = humidity if 'pm10' in data else 50.0  # Default value
        no2 = temperature if 'no2' in data else 40.0  # Default value
        
        # Input validation
        if pm25 < 0 or humidity < 0 or temperature < -50:
            return jsonify({'error': 'Invalid parameter values. Ensure they are within reasonable ranges.'}), 400
            
        # Calculate AQI using EPA method
        aqi_result = get_aqi_from_pollutants(pm25, pm10, no2)
        
        # Get health recommendations
        health_recs = get_health_recommendations(aqi_result['aqi'], temperature, humidity)
        
        # Predict PM2.5 from temperature and humidity if model exists
        pm25_prediction = None
        if prediction_model is not None and scaler is not None:
            try:
                features = np.array([[temperature, humidity]])
                features_scaled = scaler.transform(features)
                pm25_prediction = float(prediction_model.predict(features_scaled)[0])
            except Exception as e:
                logger.error(f"Prediction error: {str(e)}")
        
        logger.info(f"Prediction successful: AQI={aqi_result['aqi']}, Category={aqi_result['category']}")
        
        response = {
            'predicted_aqi': aqi_result['aqi'],
            'category': aqi_result['category'],
            'description': aqi_result['description'],
            'health_recommendations': health_recs,
            'pollutant_values': aqi_result['pollutant_values'],
            'weather': {
                'temperature': temperature,
                'humidity': humidity,
                'condition': health_recs['weather_condition']
            }
        }
        
        if pm25_prediction is not None:
            response['pm25_prediction'] = round(pm25_prediction, 2)
            response['pm25_accuracy'] = {
                'actual': pm25,
                'predicted': round(pm25_prediction, 2),
                'difference': round(pm25 - pm25_prediction, 2)
            }
        
        return jsonify(response)
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

@app.route('/correlation-analysis', methods=['GET'])
def correlation_analysis():
    """
    Return correlation analysis between PM2.5, temperature, and humidity
    """
    global air_quality_data
    
    if air_quality_data is None or len(air_quality_data) == 0:
        return jsonify({
            'error': 'No data available for correlation analysis'
        }), 404
    
    try:
        # Get correlation coefficients
        corr_matrix = air_quality_data[['pm25', 'temperature', 'humidity']].corr()
        
        # Convert to dictionary format
        corr_dict = {
            'pm25_temp': round(corr_matrix.loc['pm25', 'temperature'], 3),
            'pm25_humidity': round(corr_matrix.loc['pm25', 'humidity'], 3),
            'temp_humidity': round(corr_matrix.loc['temperature', 'humidity'], 3)
        }
        
        # Create correlation plot
        corr_plot = create_correlation_plot()
        
        # Interpret correlations
        interpretations = []
        for pair, value in corr_dict.items():
            strength = 'strong' if abs(value) > 0.7 else 'moderate' if abs(value) > 0.3 else 'weak'
            direction = 'positive' if value > 0 else 'negative'
            
            if pair == 'pm25_temp':
                interpretations.append(
                    f"There is a {strength} {direction} correlation between PM2.5 and temperature ({value})."
                )
            elif pair == 'pm25_humidity':
                interpretations.append(
                    f"There is a {strength} {direction} correlation between PM2.5 and humidity ({value})."
                )
            elif pair == 'temp_humidity':
                interpretations.append(
                    f"There is a {strength} {direction} correlation between temperature and humidity ({value})."
                )
        
        return jsonify({
            'correlations': corr_dict,
            'interpretation': interpretations,
            'plot': corr_plot
        })
    except Exception as e:
        logger.error(f"Error in correlation analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/time-series', methods=['GET'])
def time_series():
    """
    Return time series data for PM2.5, temperature, and humidity
    """
    global air_quality_data
    
    if air_quality_data is None or len(air_quality_data) == 0:
        return jsonify({
            'error': 'No data available for time series analysis'
        }), 404
    
    try:
        # Create time series plot
        time_plot = create_time_series_plot()
        
        # Create hourly pattern plot
        hourly_plot = create_hourly_pattern_plot()
        
        # Get statistics for each parameter
        stats = {
            'pm25': {
                'mean': round(float(air_quality_data['pm25'].mean()), 2),
                'min': round(float(air_quality_data['pm25'].min()), 2),
                'max': round(float(air_quality_data['pm25'].max()), 2),
                'std': round(float(air_quality_data['pm25'].std()), 2)
            },
            'temperature': {
                'mean': round(float(air_quality_data['temperature'].mean()), 2),
                'min': round(float(air_quality_data['temperature'].min()), 2),
                'max': round(float(air_quality_data['temperature'].max()), 2),
                'std': round(float(air_quality_data['temperature'].std()), 2)
            },
            'humidity': {
                'mean': round(float(air_quality_data['humidity'].mean()), 2),
                'min': round(float(air_quality_data['humidity'].min()), 2),
                'max': round(float(air_quality_data['humidity'].max()), 2),
                'std': round(float(air_quality_data['humidity'].std()), 2)
            }
        }
        
        return jsonify({
            'statistics': stats,
            'time_series_plot': time_plot,
            'hourly_pattern_plot': hourly_plot
        })
    except Exception as e:
        logger.error(f"Error in time series analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """
    Return information about the PM2.5 prediction model
    """
    global prediction_model, air_quality_data
    
    if prediction_model is None:
        result = train_prediction_model()
        if result is None:
            return jsonify({
                'error': 'Unable to train prediction model due to insufficient data'
            }), 404
    
    try:
        # Get model performance
        model_performance = {
            'r_squared_train': round(prediction_model.score(
                scaler.transform(air_quality_data[['temperature', 'humidity']].values),
                air_quality_data['pm25'].values
            ), 3),
            'coefficients': {
                'temperature': round(prediction_model.coef_[0], 3),
                'humidity': round(prediction_model.coef_[1], 3)
            },
            'intercept': round(prediction_model.intercept_, 3)
        }
        
        # Explain model
        explanation = []
        if abs(model_performance['coefficients']['temperature']) > abs(model_performance['coefficients']['humidity']):
            explanation.append("Temperature has a stronger influence on PM2.5 levels than humidity.")
        else:
            explanation.append("Humidity has a stronger influence on PM2.5 levels than temperature.")
            
        if model_performance['coefficients']['temperature'] > 0:
            explanation.append("As temperature increases, PM2.5 tends to increase.")
        else:
            explanation.append("As temperature increases, PM2.5 tends to decrease.")
            
        if model_performance['coefficients']['humidity'] > 0:
            explanation.append("As humidity increases, PM2.5 tends to increase.")
        else:
            explanation.append("As humidity increases, PM2.5 tends to decrease.")
            
        if model_performance['r_squared_train'] > 0.7:
            explanation.append("The model has a strong predictive power for PM2.5 based on temperature and humidity.")
        elif model_performance['r_squared_train'] > 0.4:
            explanation.append("The model has moderate predictive power for PM2.5 based on temperature and humidity.")
        else:
            explanation.append("The model has limited predictive power, suggesting PM2.5 is influenced by factors beyond temperature and humidity.")
            
        formula = f"PM2.5 = {model_performance['coefficients']['temperature']} × Temperature + {model_performance['coefficients']['humidity']} × Humidity + {model_performance['intercept']}"
        
        return jsonify({
            'performance': model_performance,
            'explanation': explanation,
            'formula': formula
        })
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)



