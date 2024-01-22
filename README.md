
# ClimaVizPro: Machine Learning-Enhanced Visual Weather Forecasting

ClimaVizPro is an advanced weather forecasting application that leverages the OpenWeatherMap API to fetch forecast data for specified cities and utilizes machine learning models for in-depth analysis and forecasting. Designed for both enthusiasts and professionals, it provides interactive visualizations, trend analysis, and predictive insights for temperature, humidity, and wind speed forecasts.

## Features

- **Weather Data Fetching**: Utilizes OpenWeatherMap API to retrieve detailed forecast data for multiple cities.
- **Interactive Visualizations**: Generates interactive plots for temperature, humidity, and wind speed, allowing for a dynamic user experience.
- **Trend Analysis**: Includes rolling average and statistical analysis to identify weather trends.
- **Machine Learning Forecasting**: Employs Linear Regression and Random Forest models to predict future weather conditions.
- **Comparative Analysis**: Supports comparisons among cities for key weather parameters.
- **Save & Export**: Ability to save generated visualizations as images for reporting and analysis.

## Getting Started

### Prerequisites

- Python 3.6+
- Requests, Pandas, Numpy, Matplotlib, Plotly, Scikit-learn

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/iamtyg/ClimaVizPro.git
   ```
2. Install the required libraries:
   ```
   pip install requests pandas numpy matplotlib plotly scikit-learn
   ```
3. Obtain an API key from [OpenWeatherMap](https://openweathermap.org/api) and set it in the application.

### Usage

1. Update the `API_KEY` and `save_path` in the script with your OpenWeatherMap API key and desired save path for the visualizations.
2. Add or remove cities in the `cities` list as per your requirement.
3. Run the script:
   ```
   python ClimaVizPro.py
   ```
4. Check the specified save path for the generated visualizations.

## How It Works

- **Data Fetching**: The script first fetches the forecast data for the specified cities using the OpenWeatherMap API.
- **Data Processing**: The fetched data is processed and structured into Pandas DataFrames for analysis.
- **Visualization**: Utilizes Matplotlib and Plotly for generating interactive charts and plots for a better understanding of the data.
- **Machine Learning Models**: Implements Linear Regression and Random Forest models to forecast future weather conditions based on historical data.
- **Comparative Analysis**: Provides functionality to compare weather parameters across different cities.

## Contributing

Contributions are welcome! Please feel free to submit pull requests, suggest features, or report bugs.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenWeatherMap API for providing the weather data.
- The Python community for the excellent libraries that made this project possible.
