# Airedellín 🌿 (In Progress 🏗️🚧)

<center>
<div style="text-align: center;">
<img src="https://raw.githubusercontent.com/Semillero-Inteligencia-Artificial-EAFIT/airedellin/main/docs/pictures/logo.png" alt="Description" style="width: 50%; height: 50%;">  
</div>
</center>

**Translations in**
[Español](https://github.com/Semillero-Inteligencia-Artificial-EAFIT/airedellin/blob/main/docs/readme_es.md)
[Deutsch](https://github.com/Semillero-Inteligencia-Artificial-EAFIT/airedellin/blob/main/docs/readme_de.md) 
[Русский](https://github.com/Semillero-Inteligencia-Artificial-EAFIT/airedellin/blob/main/docs/readme_ru.md)

**Airedellín**  to promote citizen science in Medellín, aiming to address the issue of air quality and PM2.5. The platform builds upon projects from citizen groups like Canairio and Unloquer, with a focus on visualizing and analyzing air quality data to improve public health.

taking inspiaration from sensor comunity and data from nasa satelites for space prediction pm2.5

## Project Overview

![](https://www.elmundo.com/assets/uploads/files/1deee-cajetillas-medehollin.jpg)
Medellín experiences heightened PM2.5 levels, particularly during September and April-May. PM2.5 particles, which are 2.5 micrometers in diameter, come from unclean fuels and various sources like cooking, industrial activities, and vehicles. These fine particles can penetrate deep into the lungs and bloodstream, posing severe health risks such as heart and lung diseases, strokes, and cancer. Winds from the Sahara also contribute to these increases.

Airedellín leverages cutting-edge technologies to tackle this challenge and visualize air quality data, using:

- **Python** & **FastAPI**: For backend and API development.
- **JavaScript**: For frontend interactions.
- **Deck.gl** & **MapLibre**: For beautiful, responsive map visualizations.
- **Bootstrap**: For a sleek and modern UI.
- **InfluxDB**: For efficient data storage and querying.
- **CanAirIO**: Real-time air quality data provider for Medellín.
- **Polars**: For fast, efficient data manipulation.
- **PyGeohash & H3**: For geospatial data processing.
- **Concurrency**: For handling muliples user at same time in the backend.
- **Chart.js**: For responsive and interactive charts on the frontend ploting pm2.5 and predictions.

- **Other libraris**: like Tensorflow, Xgboost ,Scikit-learn, Statsmodels



The platform includes machine learning models to analyze and predict air quality patterns based on historical data.

![](https://raw.githubusercontent.com/Semillero-Inteligencia-Artificial-EAFIT/airedellin/main/docs/pictures/2.png)


### What We Want

We aim to help people who like to visualize data, offering a platform that tracks the contamination levels in their city. This is designed for individuals with respiratory diseases, concerned citizens, and city scientists, forming part of a global network focused on air quality.

**Respiratory Diseases and Paranoia**

We understand that PM2.5 filters in cities are not enough—it is difficult to eliminate air pollution everywhere. However, we know that some areas are more contaminated than others. AirDMedellín lets you know where these areas are and when they are contaminated with PM2.5.

**People Who Like to Visualize Data**

We put in a great effort to ensure this platform is user-friendly. Users can visualize PM2.5 levels without needing to press many buttons—everything is designed to be simple and intuitive.

**City Scientists**

Our motivation was to provide [Unloquer](https://m.elcolombiano.com/antioquia/sensor-hecho-por-paisas-mide-la-calidad-del-aire-CC8307499) and [CanAirIO](https://canair.io/) with a better map for visualizing PM2.5, offering complete tools for analyzing air quality. We hope to inspire others to add their sensors to this platform as well.

**Global Network of Air Quality**

We want to bring together different initiatives and institutions that are already doing excellent work but remain separated. By integrating efforts from organizations like SIATA, CanAirIO, NASA, and SensorCommunity (our hexagon map is inspired by SensorCommunity), we aim to create a unified platform for air quality data.


### Machine Learning Models and Algorithms

The following models are used to predict and analyze air quality trends:

- **Linear Regression**: Predicts the relationship between air quality and various factors.
- **ARIMA & SARIMA**: Time series models, suitable for data with temporal and seasonal patterns.
- **Random Forest**: Effective for complex problems with multiple variables.
- **Lasso**: Regularized linear regression, helpful for reducing model complexity.
- **XGBoost**: Powerful decision tree-based algorithm using boosting .
- **Exponential Smoothing**: For smooth time series data changes.
- **LSTM**: A recurrent neural network ideal for sequence-based data like air quality.
- **Exponential Smoothing**: For smooth changes over time.
- **Prophet**: great for handling data is noisy or incomplete.
- **RNN**:sequence-based data, RNNs can model the temporal dependencies in air quality .
- **TCN** : handle long-range dependencies in sequential data like air quality, offering faster training times.
- **Polynomial Regression**: non-linear model captures more complex relationships.

and if you like you can add more or improve he models that we have :)


## Features

- **Map Visualization**: Airedellín includes a real-time, interactive map displaying air quality data from various sensors across the city. Sensors are color-coded based on the data, and clicking on a sensor displays pop-ups with detailed information.
- **Heatmap**: Users can toggle a heatmap to visualize the intensity of PM2.5 concentrations across the city.
- **3D Relief**: Adds an extra visual layer to make the map more informative.
- **Predictions**: Predict air quality levels using machine learning models.

### Pages


- **`/`**: Data Visualizer Home.
- **`/sensor/{sensor_name}`**: Displays sensor data and allows filtering by date range.
- **`/sensor/{sensor_name}/statistics`**: Shows sensor statistics like mean, variance, and standard deviation.
- **`/sensor/{sensor_name}/predictions`**: Allows users to select prediction algorithms and view results.
- **`/add_donation`**: Page to donate and support sensors.
- **`/index`**: Index page presenting the project overview.
- **`/maphex/{coordinates}/predictions`**: Uses PM2.5 data in the specified hexagon for different machine learning algorithms.
- **`/maphex/{coordinates}/statistics`**: Shows NASA satellite data statistics of PM2.5 in the specified hexagon.
- **`/maphex/{coordinates}`**: Displays NASA satellite PM2.5 data for the specified hexagon.
- **`/predictword`**: A hexagon map of Colombia with PM2.5 data from NASA satellites and prediction capabilities.
- **`/route`**: A future application that shows the path with the lowest PM2.5 concentration.


## Screenshots 🎑

![](https://raw.githubusercontent.com/Semillero-Inteligencia-Artificial-EAFIT/airedellin/main/docs/pictures/1.png)
![](https://raw.githubusercontent.com/Semillero-Inteligencia-Artificial-EAFIT/airedellin/main/docs/pictures/3_new.png)
![](https://raw.githubusercontent.com/Semillero-Inteligencia-Artificial-EAFIT/airedellin/main/docs/pictures/4_new.png)
![](https://raw.githubusercontent.com/Semillero-Inteligencia-Artificial-EAFIT/airedellin/main/docs/pictures/5.png)
![](https://raw.githubusercontent.com/Semillero-Inteligencia-Artificial-EAFIT/airedellin/main/docs/pictures/6.png)
![](https://raw.githubusercontent.com/Semillero-Inteligencia-Artificial-EAFIT/airedellin/main/docs/pictures/7.png)
![](https://raw.githubusercontent.com/Semillero-Inteligencia-Artificial-EAFIT/airedellin/main/docs/pictures/8.png)


---

## How to Run 🏃‍♀️

Follow these steps to get started with Airedellín:

1. **Clone the repository**:

    ```bash
    git clone https://github.com/Semillero-Inteligencia-Artificial-EAFIT/airedellin.git
    cd airedellin
    ```

2. **Set up your environment**:

    - Make sure you have Python 3.9+
    - Create a virtual environment:
    
        ```bash
        python -m venv env
        source env/bin/activate  # On Windows: `env\Scripts\activate`
        ```

    - Install the required dependencies:
    
        ```bash
        pip install -r requirements.txt
        ```

    - Add your Maptiler token in `data/token.txt`.
     you can get a token from [https://docs.maptiler.com/cloud/api](https://docs.maptiler.com/cloud/api)

3. **Start the application**:

    ```bash
    uvicorn airedellin:app --host 0.0.0.0 --port 9600
    ```

4. **Access the web interface**:

    Visit `http://localhost:9600` to start exploring Medellín’s air quality data.

---

## Contributing 🤝

Airedellín is an open-source, community-driven project. We appreciate all forms of contributions, including code, documentation, design, or feedback. Feel free to submit issues or pull requests on GitHub.

---

### Acknowledgements 💚

Special thanks to **Hackerspace Unloquer** and **CanAirIO** for inspiring and supporting this project. Your contribution to Medellín’s air quality efforts is invaluable!

Join us in improving Medellín’s air quality for everyone. 🚀🌱

---

### Important Notes

- **Paris Data**: The data used for Paris sensors is not real; it's for testing purposes.
- **Techincal Notes**: [Here](https://github.com/Semillero-Inteligencia-Artificial-EAFIT/airedellin/blob/main/docs/technical_documentation_en.md)
- **Useful data**: 

[hugging face](https://huggingface.co/datasets/jero98772/Pm25medellin)

[Nasa fireplaces](https://nrt3.modaps.eosdis.nasa.gov/archive/allData/61/MOD14/2024/) 

[Nasa Dust pm2.5](https://acdisc.gesdisc.eosdis.nasa.gov/data/HAQAST/MERRA2_CNN_HAQAST_PM25.1/2024/)

[Canario mobil data](https://mobile.canair.io/)

[Canairio Data](http://influxdb.canair.io:8000/dashboards)

### We want to

- [X] Heatmap
- [X] Plot sensors in map
- [X] Sensors with real pm2.5 values
- [X] Predict with some algorithms the value of pm2.5
- [X] Create dashboard for each sensor
- [X] Donations dummy system
- [X] Stadistical panel of pm2.5
- [X] Real location of sensors
- [X] Hexgon map like in [https://sensor.community/es/](https://sensor.community/es/)
- [ ] change the time when the sensor take data for better predictions , predict not from 30 sec to 30 sec. for 1 day to another day 
- [ ] Waze for pm2.5
- [ ] Web with layers for predict pm2.5 in a anothermap
