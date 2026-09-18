# EduRisk — AI-Powered Student Dropout Prediction System

> An end-to-end Machine Learning and Deep Learning system that predicts student dropout risk, segments students into risk clusters, and delivers actionable intervention recommendations through an interactive Streamlit dashboard.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Live Demo](#live-demo)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Running the Application](#running-the-application)
- [ML Models & Performance](#ml-models--performance)
- [Dataset](#dataset)
- [Application Pages](#application-pages)
- [System Architecture](#system-architecture)
- [Screenshots](#screenshots)
- [Internship Details](#internship-details)
- [Future Scope](#future-scope)
- [License](#license)

---

## Project Overview

Student dropout is one of the most critical problems in higher education. Traditional institutional approaches rely on **reactive intervention** — acting only after a student has already disengaged or left. EduRisk solves this by building an **intelligent early-warning system** that predicts dropout risk *before* it happens.

The system analyzes academic performance, financial status, attendance patterns, and demographic data to classify each student into one of three outcome categories:

| Class | Label | Description |
|-------|-------|-------------|
| 0 | **Dropout Risk** | High probability of leaving before completion |
| 1 | **Enrolled** | Currently on track, moderate risk |
| 2 | **Graduate** | Strong trajectory toward graduation |

Beyond prediction, EduRisk provides:
- **K-Means risk segmentation** (Low / Medium / High risk clusters)
- **Personalized intervention recommendations** for at-risk students
- **Multi-model performance comparison** across 5 ML algorithms
- **Interactive analytics dashboard** with 10+ chart types

---

## Live Demo

To run the application locally, follow the [Installation & Setup](#installation--setup) section below.

After setup, the app runs at:
```
http://localhost:8501
```

---

## Features

### Analytics Dashboard
- 6 real-time KPI metrics (total students, dropout count, graduate rate, avg grades, attendance)
- Outcome distribution donut chart
- Grade distribution violin plots by outcome
- Dropout rate by attendance band (color-coded bar chart)
- Grade trajectory (Semester 1 → Semester 2) per outcome group
- Key risk factor comparisons: scholarship, debt, tuition fees, gender
- Grade vs attendance scatter plot
- Full correlation heatmap across top academic features

### Student Risk Predictor
- Live input form with academic, personal, financial, and economic parameters
- Instant prediction with confidence percentage
- Probability breakdown bar chart for all 3 outcome classes
- Risk factor analysis with severity tags (Critical / High / Medium / Positive)
- Personalized intervention recommendations based on predicted outcome
- Snapshot metric summary (grades, attendance, pass rate)

### Model Analytics
- Scorecards for all 5 trained models
- Grouped bar chart comparing Accuracy, F1, Precision, Recall
- Radar chart for holistic model performance comparison
- Top 15 feature importances from Random Forest
- Confusion matrices for every model
- ANN deep learning architecture code display

### Cluster Analysis
- 3-cluster risk profile cards (Low / Medium / High risk)
- PCA 2D scatter visualization of student risk groups
- Grade comparison and dropout rate bar charts per cluster
- Outcome breakdown (Dropout / Enrolled / Graduate) within each cluster

### About Page
- Project abstract and problem statement
- 8-step ML pipeline architecture
- Full technology and model reference
- Student and internship details

---

## Tech Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| Language | Python 3.11 | Core development |
| Data | Pandas, NumPy | Data manipulation and computation |
| ML Models | Scikit-learn | Classification and clustering |
| Deep Learning | TensorFlow, Keras | ANN model |
| Visualization | Plotly | Interactive charts |
| Visualization | Matplotlib, Seaborn | Static EDA visuals |
| Web App | Streamlit | Dashboard and deployment |
| Serialization | Joblib | Model persistence (.pkl files) |
| IDE | Jupyter Notebook, VS Code | Development and experimentation |

---

## Project Structure

```
edurisk_fixed/
│
├── app.py                    # Main Streamlit application (758 lines)
├── train_models.py           # Model training script — run this first
├── generate_data.py          # Synthetic dataset generator
├── requirements.txt          # Python dependencies
│
├── data/
│   ├── dataset.csv           # Raw synthetic dataset (2,000 records, 37 features)
│   └── processed_dataset.csv # Feature-engineered dataset with cluster labels
│
└── models/
    ├── best_model.pkl        # Trained best model (Logistic Regression)
    ├── scaler.pkl            # Fitted StandardScaler
    ├── kmeans.pkl            # Trained K-Means model (k=3)
    ├── pca.pkl               # Fitted PCA transformer (2 components)
    ├── feature_cols.pkl      # List of 37 feature column names
    ├── results.json          # All model metrics and dataset statistics
    ├── cluster_profiles.csv  # Cluster summary statistics
    └── pca_data.csv          # 2D PCA coordinates for visualization
```

---

## Installation & Setup

### Prerequisites

- Python 3.9 or above
- pip package manager

Check your Python version:
```bash
python --version
```

### Step 1 — Download and extract

Download the project ZIP and extract it. You will get an `edurisk_fixed` folder.

### Step 2 — Navigate to the project folder

```bash
cd edurisk_fixed
```

> **Important:** Make sure you are inside the folder that contains `app.py` and `requirements.txt` before running any commands.

### Step 3 — Create a virtual environment (recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac / Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal line.

### Step 4 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs all required packages. Estimated time: 2–4 minutes depending on internet speed.

If the above fails, install manually:
```bash
pip install streamlit pandas numpy scikit-learn plotly joblib
```

---

## Running the Application

### Step 1 — Train the models (required on first run)

```bash
python train_models.py
```

This script trains all 5 ML models on your local environment and saves fresh `.pkl` files compatible with your installed sklearn version. Always run this before launching the app.

Expected output:
```
Logistic Regression: 81.75%
KNN: 56.75%
Decision Tree: 73.25%
Random Forest: 80.0%
Gradient Boosting: 77.25%
Best: Logistic Regression
All artifacts saved successfully!
```

### Step 2 — Launch the Streamlit app

```bash
python -m streamlit run app.py
```

The app opens automatically in your browser at `http://localhost:8501`.

> If port 8501 is busy, run: `python -m streamlit run app.py --server.port 8502`

### Troubleshooting

| Error | Fix |
|-------|-----|
| `streamlit not recognized` | Use `python -m streamlit run app.py` instead |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` inside the correct folder |
| `FileNotFoundError: models/best_model.pkl` | Run `python train_models.py` first |
| `multi_class AttributeError` | Run `python train_models.py` to regenerate models for your sklearn version |
| `KeyError: Avg_Sem1_Grade` | Run `python train_models.py` to regenerate `cluster_profiles.csv` |
| Port already in use | Add `--server.port 8502` to the run command |

---

## ML Models & Performance

All models are trained on the same 80/20 stratified split for fair comparison.

| Model | Accuracy | F1 Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| **Logistic Regression** ⭐ | **81.75%** | **81.84%** | **82.10%** | **81.75%** |
| Random Forest | 80.00% | 80.21% | 80.45% | 80.00% |
| Gradient Boosting | 77.25% | 77.43% | 77.68% | 77.25% |
| Decision Tree | 73.25% | 73.41% | 73.60% | 73.25% |
| KNN (k=7) | 56.75% | 57.27% | 58.10% | 56.75% |

### ANN Architecture

```python
model = Sequential([
    Dense(256, activation='relu', input_shape=(37,)),
    BatchNormalization(),
    Dropout(0.35),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.30),

    Dense(64, activation='relu'),
    Dropout(0.25),

    Dense(3, activation='softmax')    # Dropout / Enrolled / Graduate
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

---

## Dataset

The project uses a **synthetic dataset** generated to simulate realistic student academic patterns.

| Property | Value |
|----------|-------|
| Total records | 2,000 students |
| Features | 37 input features |
| Target classes | 3 (Dropout, Enrolled, Graduate) |
| Dropout records | 680 (34%) |
| Enrolled records | 660 (33%) |
| Graduate records | 660 (33%) |
| Avg Sem1 Grade | 12.96 / 20 |
| Avg Sem2 Grade | 12.71 / 20 |
| Avg Attendance | 70.5% |

### Key Features Used

- Curricular units enrolled, approved, and graded (Semester 1 & 2)
- Attendance percentage
- Admission grade and previous qualification grade
- Scholarship holder status
- Debtor status and tuition fees up to date
- Age at enrollment, gender, marital status
- Unemployment rate, inflation rate, GDP (economic context)

> For production use, replace `data/dataset.csv` with real institutional data and re-run `train_models.py`.

---

## Application Pages

| Page | Description |
|------|-------------|
| 📊 Dashboard | KPIs, distribution charts, risk factor analysis, correlation heatmap |
| 🔍 Predict Student | Live dropout risk predictor with confidence score and interventions |
| 🤖 Model Analytics | 5-model comparison, feature importances, confusion matrices, ANN code |
| 👥 Cluster Analysis | K-Means risk segments, PCA visualization, cluster profiles |
| 📋 About | Project abstract, ML pipeline, tech stack, student details |

---

## System Architecture

```
Raw Data (37 features)
        │
        ▼
┌─────────────────────┐
│   Preprocessing     │  StandardScaler · Label encoding · 80/20 stratified split
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   EDA & Analysis    │  Distributions · Risk factors · Grade trends · Heatmap
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  ML Model Training  │  LR · KNN · DT · RF · GB (5 models compared)
└──────────┬──────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
┌─────────┐  ┌─────────────────────┐
│   ANN   │  │  K-Means + PCA      │
│ 256→3   │  │  3 risk clusters    │
└────┬────┘  └─────────┬───────────┘
     └────────┬─────────┘
              │
              ▼
┌─────────────────────┐
│  Model Serialization│  best_model.pkl · scaler.pkl · kmeans.pkl · results.json
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Streamlit Web App  │  Dashboard · Predict · Models · Clusters · About
└──────────┬──────────┘
           │
           ▼
  Dropout Risk Score + Intervention Recommendations + Cluster Assignment
```

---

## Screenshots

> Add screenshots of the running app here after launching it locally.

Recommended screenshots to capture:
1. **Dashboard** — KPI row + outcome donut chart - ![alt text](<Screenshot 2026-05-13 202710.png>)
2. **Predict Student** — filled form + prediction result card - ![alt text](<Screenshot 2026-05-13 202848.png>)
3. **Model Analytics** — grouped bar chart + radar chart - ![alt text](<Screenshot 2026-05-13 202914.png>)
4. **Cluster Analysis** — PCA scatter plot + cluster profile cards - ![alt text](<Screenshot 2026-05-13 203016.png>)

---


## Future Scope

- **Real dataset integration** — Replace synthetic data with actual VTU or institutional records
- **Explainable AI (XAI)** — Integrate SHAP or LIME for per-student prediction explanations
- **Real-time monitoring** — Connect to live attendance/grade APIs for continuous risk tracking
- **Multi-institution support** — Generalize to multiple colleges with institution-specific calibration
- **Mobile application** — Lightweight mobile interface for academic advisors
- **Advanced models** — Explore TabTransformer or FT-Transformer for improved tabular accuracy

---

## License

This project was developed as part of an academic internship. It is intended for educational and research purposes.

---

*Built with Python · Scikit-learn · TensorFlow · Streamlit · Plotly*

---

# 👨‍💻 Author

Dhiksha C G

Built as a portfolio project for Data Analyst and Financial Data Governance roles.
