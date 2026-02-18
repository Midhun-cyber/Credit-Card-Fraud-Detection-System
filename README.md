# 💳 Credit Card Fraud Detection System

A production-style Machine Learning application that detects fraudulent credit card transactions using advanced classification algorithms and an interactive Streamlit dashboard.

This project demonstrates real-world ML engineering practices such as pipeline-based training, handling imbalanced datasets, artifact-based inference, and modular architecture.

---

## 🚀 Key Features

✅ Fraud detection using Machine Learning  
✅ Handles highly imbalanced data with **SMOTE**  
✅ Supports **XGBoost** and **LightGBM** models  
✅ Batch transaction analysis via CSV upload  
✅ Dynamic fraud probability threshold  
✅ Interactive analytics dashboard  
✅ Fraud-only result export  
✅ Clean and scalable project structure  

---

## 🧠 Tech Stack

### Machine Learning
- Scikit-learn  
- XGBoost  
- LightGBM  
- Imbalanced-learn  

### Data Processing
- Pandas  
- NumPy  

### Visualization
- Matplotlib  

### Application Layer
- Streamlit  

---

## 📂 Project Structure

```
credit_card_fraud_app/
│
├── data/                  # Dataset
├── models/                # Saved trained model
├── src/
│   ├── app.py             # Streamlit application
│   ├── train_pipeline.py  # Model training pipeline
│   ├── utils.py           # Helper functions
│   └── premium_mode.py    # Advanced analytics
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation Guide

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/credit_card_fraud_app.git
cd credit_card_fraud_app
```

---

### 2️⃣ Create a Virtual Environment (Recommended)

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🏋️ Train the Model

Run the training pipeline:

```bash
python src/train_pipeline.py \
--csv data/creditcard.csv \
--out models/model_artifact.joblib \
--model xgb
```

### Optional Parameters

```
--model lgbm     # Use LightGBM
--grid          # Enable GridSearch tuning
```

---

## ▶️ Run the Application

Start the Streamlit app:

```bash
streamlit run src/app.py
```

Then open:

```
http://localhost:8501
```

---

## 📊 Dataset

This project uses the **Kaggle Credit Card Fraud Detection Dataset**, which contains anonymized transaction features.

**Dataset Characteristics:**
- Highly imbalanced  
- Real-world transaction patterns  
- Ideal for fraud detection research  

---

## 🔥 Architecture Highlights

✔️ Pipeline-based model training  
✔️ Feature alignment before inference  
✔️ Artifact saving with Joblib  
✔️ No training during runtime  
✔️ Separation of ML logic and UI  

This design mirrors real production ML systems.

---

## 🎯 Future Enhancements

- Docker containerization  
- FastAPI integration for real-time predictions  
- Cloud deployment (AWS/GCP/Azure)  
- Model monitoring  
- Automated retraining pipeline  

---

## 👨‍💻 Author

**Midhun**  
B.Tech CSE — Data Science with Machine Learning  

---

## ⭐ If You Found This Useful

Consider giving the repository a ⭐ to support the project!
