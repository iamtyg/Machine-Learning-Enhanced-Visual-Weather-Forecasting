# ClimaVizPro: Machine Learning-Enhanced Visual Weather Forecasting

# path/filename: ClimaVizPro

# Purpose:
# This program fetches weather forecast data from the OpenWeatherMap API for specified cities,
# processes the data, and performs various analyses including visualization, trend analysis,
# and forecasting using machine learning models. It supports interactive visualizations and
# comparisons among cities for temperature, humidity, and wind speed forecasts.


import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import os

# Your API key and other constants
API_KEY = "{api_key}"
cities = ["New York", "Paris", "Istanbul"]
base_url = "http://api.openweathermap.org/data/2.5/forecast?"
save_path = "{save_path}"


# Fetches weather forecast data for a given city using OpenWeatherMap API and processes it into a pandas DataFrame.
def fetch_and_process_data(city, api_key):
    complete_url = f"{base_url}q={city}&appid={api_key}&units=metric"
    response = requests.get(complete_url).json()
    if 'list' not in response:
        print(f"Error fetching data for {city}: {response.get('message', 'Unknown Error')}")
        return None  
    forecast_items = response['list']
    dates, temp_min, temp_max, humidity, wind_speed, descriptions = [], [], [], [], [], []
    for item in forecast_items:
        dates.append(item['dt_txt'])
        temp_min.append(item['main']['temp_min'])
        temp_max.append(item['main']['temp_max'])
        humidity.append(item['main']['humidity'])
        wind_speed.append(item['wind']['speed'])
        descriptions.append(item['weather'][0]['description'])
    return pd.DataFrame({
        'date': dates, 'temp_min': temp_min, 'temp_max': temp_max,
        'humidity': humidity, 'wind_speed': wind_speed, 'description': descriptions
    })


# Visualizes and saves interactive plots for temperature, humidity, and wind speed forecasts for a city.
def visualize_and_save_data_interactive(data, city):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=data['date'], y=data['temp_max'], name='Max Temperature (°C)', line=dict(color='red')), secondary_y=False)
    fig.add_trace(go.Scatter(x=data['date'], y=data['temp_min'], name='Min Temperature (°C)', line=dict(color='blue')), secondary_y=False)
    fig.update_layout(title_text=f'Temperature Forecast for {city}')
    fig.update_xaxes(title_text='Date and Time')
    fig.update_yaxes(title_text='Temperature (°C)', secondary_y=False)
    fig.write_image(os.path.join(save_path, f"{city}_interactive_temperature_forecast.png"))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['date'], y=data['humidity'], name='Humidity (%)', line=dict(color='green')))
    fig.update_layout(title=f'Humidity Forecast for {city}', xaxis_title='Date and Time', yaxis_title='Humidity (%)')
    fig.write_image(os.path.join(save_path, f"{city}_interactive_humidity_forecast.png"))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['date'], y=data['wind_speed'], name='Wind Speed (m/s)', line=dict(color='purple')))
    fig.update_layout(title=f'Wind Speed Forecast for {city}', xaxis_title='Date and Time', yaxis_title='Wind Speed (m/s)')
    fig.write_image(os.path.join(save_path, f"{city}_interactive_wind_speed_forecast.png"))

# Analyzes temperature trends and visualizes statistical information for forecast data.
def analyze_trends_and_statistics_interactive(data, city):
    temp_trend = data['temp_max'].rolling(window=5).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['date'], y=temp_trend, name='5-Day Rolling Avg Max Temp', line=dict(color='darkorange')))
    fig.update_layout(title=f'5-Day Rolling Average Max Temperature Trend for {city}', xaxis_title='Date and Time', yaxis_title='Temperature (°C)')
    fig.write_image(os.path.join(save_path, f"{city}_interactive_temp_trend.png"))

    stats = data[['temp_max', 'humidity', 'wind_speed']].describe()
    fig = go.Figure(data=[go.Bar(name=col, x=stats.index, y=stats[col]) for col in stats.columns])
    fig.update_layout(barmode='group', title_text=f'Statistical Information for {city}')
    fig.write_image(os.path.join(save_path, f"{city}_interactive_statistics.png"))


# Builds and evaluates an advanced weather forecast model using machine learning for maximum temperature prediction.
def advanced_weather_forecast_model(data, city):
    data['days'] = pd.to_datetime(data['date']).dt.dayofyear
    X = data[['days']]
    y = data['temp_max']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    
    fig = px.scatter(x=X_test.squeeze(), y=y_test, labels={'x': 'Day of Year', 'y': 'Max Temperature (°C)'})
    fig.add_scatter(x=X_test.squeeze(), y=predictions, mode='lines', name='Predicted Max Temp')
    fig.update_layout(title=f'Advanced Temperature Forecast Model for {city}')
    fig.write_image(os.path.join(save_path, f"{city}_advanced_forecast_model.png"))

# Builds a multi-variable forecast model for predicting temperature, humidity, and wind speed using machine learning.
def multi_variable_forecast_model(data, city):
    data['days'] = pd.to_datetime(data['date']).dt.dayofyear
    X = data[['days']]
    y = data[['temp_max', 'humidity', 'wind_speed']]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)

    fig = make_subplots(rows=3, cols=1, subplot_titles=("Temperature", "Humidity", "Wind Speed"))
    
    fig.add_trace(go.Scatter(x=X_test['days'], y=y_test['temp_max'], mode='markers', name='Actual Max Temp'), row=1, col=1)
    fig.add_trace(go.Scatter(x=X_test['days'], y=predictions[:,0], mode='lines', name='Predicted Max Temp'), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=X_test['days'], y=y_test['humidity'], mode='markers', name='Actual Humidity'), row=2, col=1)
    fig.add_trace(go.Scatter(x=X_test['days'], y=predictions[:,1], mode='lines', name='Predicted Humidity'), row=2, col=1)
    
    fig.add_trace(go.Scatter(x=X_test['days'], y=y_test['wind_speed'], mode='markers', name='Actual Wind Speed'), row=3, col=1)
    fig.add_trace(go.Scatter(x=X_test['days'], y=predictions[:,2], mode='lines', name='Predicted Wind Speed'), row=3, col=1)
    
    fig.update_layout(height=800, title_text=f"Multi-variable Weather Forecast for {city}")
    fig.write_image(os.path.join(save_path, f"{city}_multi_variable_forecast.png"))
    
# Compares temperature forecasts across specified cities using visualization.
def compare_cities_forecast(data_dict):
    fig = go.Figure()
    for city, data in data_dict.items():
        fig.add_trace(go.Scatter(x=data['date'], y=data['temp_max'], name=f'Max Temp {city}', mode='lines'))
    
    fig.update_layout(title="Max Temperature Comparison Among Cities", xaxis_title="Date", yaxis_title="Max Temperature (°C)")
    fig.write_image(os.path.join(save_path, "cities_temperature_comparison.png"))

# Visualizes the daily difference between maximum and minimum temperatures for a city.
def visualize_temperature_difference(data, city):
    data['temp_diff'] = data['temp_max'] - data['temp_min']
    fig = px.bar(data, x='date', y='temp_diff', title=f'Daily Temperature Difference for {city}')
    fig.update_layout(xaxis_title='Date', yaxis_title='Temperature Difference (°C)')
    fig.write_image(os.path.join(save_path, f"{city}_temp_difference.png"))

# Analyzes the relationship between temperature and humidity through scatter plots and trend lines.
def analyze_temp_humidity_relationship(data, city):
    fig = px.scatter(data, x='temp_max', y='humidity', title=f'Temperature vs Humidity for {city}',
                     trendline='ols', labels={'temp_max': 'Max Temperature (°C)', 'humidity': 'Humidity (%)'})
    fig.write_image(os.path.join(save_path, f"{city}_temp_humidity_relationship.png"))

# Compares weather conditions such as temperature, humidity, and wind speed among cities.
def compare_weather_conditions(data_dict):
    comparison_metrics = ['temp_max', 'humidity', 'wind_speed']
    for metric in comparison_metrics:
        fig = go.Figure()
        for city, data in data_dict.items():
            fig.add_trace(go.Scatter(x=data['date'], y=data[metric], mode='lines', name=city))
        fig.update_layout(title=f"{metric.capitalize()} Comparison Among Cities", xaxis_title="Date", yaxis_title=metric.capitalize())
        fig.write_image(os.path.join(save_path, f"cities_{metric}_comparison.png"))

# Main processing block
if __name__ == "__main__":
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    data_dict = {}
    for city in cities:
        data = fetch_and_process_data(city, API_KEY)
        if data is not None:
            data_dict[city] = data
            visualize_and_save_data_interactive(data, city)
            analyze_trends_and_statistics_interactive(data, city)
            multi_variable_forecast_model(data, city)
            visualize_temperature_difference(data, city)
            analyze_temp_humidity_relationship(data, city)
    compare_weather_conditions(data_dict)


