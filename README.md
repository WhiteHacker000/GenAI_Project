# ⚡ EV Charging Station Energy Prediction

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://genaiproject-yhnnwtxdp3dkjy4ifvdvpj.streamlit.app/)

> **🌐 Hosted App:** [https://genaiproject-yhnnwtxdp3dkjy4ifvdvpj.streamlit.app/](https://genaiproject-yhnnwtxdp3dkjy4ifvdvpj.streamlit.app/)

A machine learning project that predicts energy consumption (kWh) at EV charging stations based on time of day, day of week, and station characteristics. Includes a Streamlit web app for interactive predictions and an agentic LangGraph workflow for infrastructure planning.

## Tech Stack

- **Python 3.10+**
- **Scikit-learn** — Random Forest Regressor for prediction
- **Pandas / NumPy** — Data processing and feature engineering
- **Streamlit** — Interactive web interface
- **Matplotlib / Seaborn** — Exploratory data analysis (notebook)

## Project Structure

```
├── app.py                          # Streamlit frontend
├── api.py                          # FastAPI backend
├── agent.py                        # LangGraph agentic workflow
├── rag_builder.py                  # Script to build FAISS vector store
├── infrastructure_guidelines.md    # RAG knowledge base
├── faiss_index/                    # Saved FAISS vector store
├── model_artifacts.pkl             # Trained Random Forest model
├── Dockerfile                      # Container config
├── start.sh                        # Startup script (Docker/Render)
├── runtime.txt                     # Python version for Streamlit Cloud
├── requirements.txt                # Python dependencies
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

### 4. Start the FastAPI backend

```bash
python api.py
```

This starts the backend at `http://127.0.0.1:8000`.

### 5. Run the Streamlit app

```bash
streamlit run app.py
```

Open in browser: **http://localhost:8501**

> 🌐 Or use the hosted version: [https://genaiproject-yhnnwtxdp3dkjy4ifvdvpj.streamlit.app/](https://genaiproject-yhnnwtxdp3dkjy4ifvdvpj.streamlit.app/)

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
