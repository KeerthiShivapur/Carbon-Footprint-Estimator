
# 🏬 Carbon Footprint Estimator for Retail Operations

This project is a machine learning-powered Streamlit web application designed to estimate and visualize the carbon footprint of retail stores based on operational factors like energy usage, transportation, packaging, and more. It was developed as part of a hackathon and internship submission.

## 📌 Objective

Help retailers estimate their CO₂ emissions, understand contributing factors, and take actionable steps to reduce their environmental impact.

## 🚀 Features

- 🌱 Predicts carbon emissions (in kg CO₂e) using ML
- 📊 Visual breakdown of emissions by category
- 🧠 Shows model performance (MAE, R² Score)
- 📉 Visualizes feature importance (SHAP values or permutation)
- ⏳ Forecast future emissions (optional)
- 🏪 Optimized for Walmart-style retail operations
- 💡 Actionable sustainability suggestions

## 🧠 ML Model Info

- Model: Random Forest Regressor
- Dataset: Synthetic 1000-row retail dataset (`retail_carbon_data_1000.csv`)
- Evaluation:
  - MAE: ~40 kg CO₂e
  - R² Score: ~0.997

## 🧰 Tech Stack

- Python 3.8+
- Scikit-learn
- Pandas, NumPy
- Streamlit
- Matplotlib / Plotly (for charts)

## 🛠️ Installation & How to Run

1. Unzip the folder
2. Run on command Prompt
3. 📦 Install dependencies

```bash
pip install -r requirements.txt
```

4. ▶️ Run the Streamlit app

```bash
streamlit run app.py
```

5. 🌐 App will open at:

```plaintext
http://localhost:8501
```

---

## 📂 Project Structure

```bash
carbon-footprint-estimator/
│
├── app.py                   # Streamlit app
├── retail_carbon_model.pkl  # Trained ML model
├── requirements.txt         # Python dependencies
├── data/
│   └── retail_carbon_data_1000.csv
├── screenshots/
│   └── demo.png
├── README.md


