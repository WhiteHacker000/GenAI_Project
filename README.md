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

## System Architecture

```
📂 User provides EVChargingStationUsage.csv
  |
  ▼
┌─────────────────────┐
│   DataLoader         │  ← Pandas
│  (Read the CSV)      │
└─────────────────────┘
        │ raw dataframe
        ▼
┌─────────────────────┐
│  DataPreprocessor    │  ← Pandas / NumPy
│  Parse timestamps    │
│  → Remove zero-energy│
│  → Filter columns    │
└─────────────────────┘
        │ clean dataframe
        ▼
┌─────────────────────┐
│  FeatureEngineer     │  ← Pandas / scikit-learn
│  Group by Station/   │
│  Hour/Weekday        │
│  → Peak_Hour flag    │
│  → Is_Weekend flag   │
│  → Station_Avg_Load  │
│  → Interaction feat  │
│  → One-Hot Encoding  │
└─────────────────────┘
        │ feature matrix
        ▼
┌─────────────────────┐
│  ModelTrainer        │  ← scikit-learn
│  Random Forest       │
│  Regressor           │
│  (n_estimators=100)  │
└─────────────────────┘
        │ trained model
        ▼
┌─────────────────────┐
│  ArtifactSerializer  │  ← joblib
│  Save model +        │
│  feature_columns +   │
│  station_avg_map     │
└─────────────────────┘
        │ model_artifacts.pkl
        ▼
┌─────────────────────┐
│  Streamlit Web App   │  ← Streamlit
│  Load artifacts      │
│  → User selects:     │
│    Station / Hour /  │
│    Day of week       │
│  → Reconstruct       │
│    feature vector    │
└─────────────────────┘
        │ input features
        ▼
┌─────────────────────┐
│  Predictor           │  ← scikit-learn (RF)
│  model.predict()     │
│  → Energy (kWh)      │
└─────────────────────┘
        │ prediction
        ▼
┌─────────────────────┐
│  ResultDisplay       │  ← Streamlit
│  Show predicted kWh  │
│  → Compare with      │
│    station average   │
│  → High/Low demand   │
│    indicator         │
└─────────────────────┘

📓 Jupyter Notebook (EDA & Experimentation)
   Insights inform feature design
   ← Matplotlib / Seaborn
```
