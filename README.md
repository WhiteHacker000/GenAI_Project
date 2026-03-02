# ⚡ EV Charging Station Energy Prediction

A machine learning project that predicts energy consumption (kWh) at EV charging stations based on time of day, day of week, and station characteristics. Includes a Streamlit web app for interactive predictions.

## Tech Stack

- **Python 3.10+**
- **Scikit-learn** — Random Forest Regressor for prediction
- **Pandas / NumPy** — Data processing and feature engineering
- **Streamlit** — Interactive web interface
- **Matplotlib / Seaborn** — Exploratory data analysis (notebook)

## Project Structure

```
├── app.py                  # Streamlit web app for predictions
├── scripts/
│   ├── train_model.py      # Model training pipeline
│   └── extracted_code.py   # EDA and experimentation code from notebook
├── notebooks/
│   └── GenAI_Ptoject.ipynb # Jupyter notebook with exploration & analysis
├── data/
│   └── (place EVChargingStationUsage.csv here)
├── requirements.txt        # Python dependencies
└── README.md
```

## Setup Instructions

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd GenAI_Project
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Place the `EVChargingStationUsage.csv` file in the `data/` directory.

### 5. Train the model

```bash
python scripts/train_model.py
```

This generates `model_artifacts.pkl` in the project root.

### 6. Run the web app

```bash
streamlit run app.py
```

Open the URL shown in the terminal (default: http://localhost:8501).

## How It Works

1. **Data Preprocessing** — Parses timestamps, removes zero-energy sessions, groups by station/hour/weekday.
2. **Feature Engineering** — Creates peak hour indicator, weekend flag, station average load, and interaction features.
3. **Model Training** — Trains a Random Forest Regressor on the engineered features.
4. **Prediction App** — Users select a station, hour, and day to get an energy consumption prediction with context against the station's historical average.
