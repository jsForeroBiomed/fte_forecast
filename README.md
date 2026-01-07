# FTE Forecasting Dashboard

A time series forecasting application for Full-Time Equivalent (FTE) workforce planning, built with Python and Shiny. This project provides an interactive dashboard for visualizing historical FTE data, comparing forecasting models, and generating future predictions with uncertainty intervals.

## Features

- **Interactive Dashboard**: Modern web interface built with Shiny for Python
- **Time Series Forecasting**: ARIMA-based forecasting model for monthly FTE predictions
- **Model Comparison**: Compare multiple ARIMA models using train and validation metrics
- **Backtesting**: Rolling window backtesting on validation and test datasets
- **Uncertainty Quantification**: Forecast intervals with configurable confidence levels
- **Visual Analytics**: Interactive plots for historical trends, forecasts, and model performance

## Project Structure

```
fte_forecasting/
├── src/
│   └── app.py              # Main Shiny application
├── notebooks/
│   ├── 1_dataset_generation.ipynb    # Data generation pipeline
│   ├── 2_preprocessing.ipynb        # Data preprocessing
│   ├── 3_baseline.ipynb             # Baseline model evaluation
│   └── 4_arima.ipynb                # ARIMA model selection and training
├── data/                    # Training, validation, and test datasets
├── results/                 # Model evaluation metrics and results
├── requirements.txt         # Python dependencies
├── Dockerfile              # Container configuration for deployment
└── README.md               # This file
```

## Requirements

- Python 3.10+
- See `requirements.txt` for full list of dependencies

## Installation

### Local Development

1. Clone the repository:
```bash
git clone <repository-url>
cd fte_forecasting
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install shiny
```

4. Run the application:
```bash
cd src
shiny run app.py
```

The application will be available at `http://127.0.0.1:8000`

## Usage

### Dashboard Sections

1. **Overview**
   - General information about the KPI target (3300 FTE)
   - Historical FTE trends with train/validation/test splits
   - Last observed values and model information

2. **Model Comparison**
   - Comparison table of ARIMA models with train and validation RMSE
   - Bar chart visualization of model performance
   - Explanation of RMSE metrics for non-technical stakeholders

3. **Backtesting**
   - Rolling window backtesting results
   - Performance metrics (RMSE, MAE) for validation and test datasets
   - Visual comparison of actual vs predicted values

4. **Forecast**
   - Interactive forecast generation with configurable:
     - Forecast horizon (6, 12, or 24 months)
     - Confidence level (80% or 95%)
   - Point estimates and prediction intervals
   - Target FTE threshold visualization

### Model Details

- **Final Model**: ARIMA(0,1,0) - Random Walk
- **Data Frequency**: Monthly
- **Target KPI**: Less than 3300 FTE

## Deployment

### Docker

Build and run the container:

```bash
docker build -t fte-forecasting .
docker run -p 8000:8000 fte-forecasting
```

### Cloud Deployment

The application can be deployed to cloud platforms that support Docker:

- **Render.com**: Connect your GitHub repository and select "Web Service" with Docker
- **Railway.app**: Automatic deployment from GitHub
- **Fly.io**: Deploy using Dockerfile
- **Heroku**: Container registry deployment

The `Dockerfile` is configured to:
- Install all Python dependencies
- Copy application files and required data
- Expose port 8000
- Run the Shiny application

## Data

The project uses simulated pharmaceutical trials data for FTE forecasting. The datasets include:

- Training data: `data/3_internal_train_data.csv`
- Validation data: `data/3_internal_validation_data.csv`
- Test data: `data/.test/test_data.csv`
- Rolling backtest data: `data/4_rolling_validation_df.csv`, `data/4_rolling_test_df.csv`

## Results

Model evaluation results are stored in the `results/` directory:

- `train_results.csv`: Training set metrics for all models
- `validation_results.csv`: Validation set metrics
- `backtest_metrics.csv`: Rolling backtest performance metrics

## Development

### Notebooks Workflow

1. **Dataset Generation** (`1_dataset_generation.ipynb`): Generate synthetic FTE data
2. **Preprocessing** (`2_preprocessing.ipynb`): Clean and prepare data for modeling
3. **Baseline** (`3_baseline.ipynb`): Establish baseline forecasting performance
4. **ARIMA** (`4_arima.ipynb`): Model selection, training, and evaluation

### Key Technologies

- **Shiny for Python**: Interactive web application framework
- **statsmodels**: ARIMA time series modeling
- **pandas**: Data manipulation and analysis
- **matplotlib**: Static plotting and visualization
- **scikit-learn**: Machine learning utilities
