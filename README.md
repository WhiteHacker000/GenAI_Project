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
├── start.sh                        # FastAPI startup script (Docker/Render)
├── render.yaml                     # Render backend service definition
├── runtime.txt                     # Python version for Streamlit Cloud
├── requirements.txt                # Python dependencies
└── README.md
```

## Deployment Architecture

The frontend and backend are independent services in production:

```
Streamlit frontend
        |
        | BACKEND_URL
        v
FastAPI backend
        |
        +--> ML model
        +--> LangGraph
        +--> Groq
```

Streamlit Cloud runs only `app.py`. Deploy the FastAPI backend separately and
set the Streamlit Cloud `BACKEND_URL` secret to its public HTTPS URL. The
Streamlit Cloud `/api/v2/subdomain/...` browser request is an internal
Streamlit request and is unrelated to these application routes.

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

### 4. Start the FastAPI backend locally

```bash
source .venv/bin/activate
uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

### 5. Start the Streamlit frontend locally

In a second terminal:

```bash
source .venv/bin/activate
streamlit run app.py
```

Open in browser: **http://localhost:8501**

> 🌐 Or use the hosted version: [https://genaiproject-yhnnwtxdp3dkjy4ifvdvpj.streamlit.app/](https://genaiproject-yhnnwtxdp3dkjy4ifvdvpj.streamlit.app/)

### Environment variables

Copy `.env.example` to `.env` for local development and provide your own
values. Never commit `.env` or real API keys.

- `GROQ_API_KEY`: required by the `/plan` endpoint.
- `GROQ_MODEL`: optional Groq model name.
- `BACKEND_URL`: frontend URL for the FastAPI service; defaults to local port 8000.
- `FRONTEND_URL`: comma-separated allowed frontend origins for backend CORS.

### Production backend deployment

The included `render.yaml` defines a standalone Render web service. Its start
command is `uvicorn api:app --host 0.0.0.0 --port $PORT`, and the service health
check uses `GET /`. Set `GROQ_API_KEY` and `FRONTEND_URL` in Render, then set
the resulting service URL as `BACKEND_URL` in Streamlit Cloud. Do not use
`127.0.0.1:8000` as the production backend URL.

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
