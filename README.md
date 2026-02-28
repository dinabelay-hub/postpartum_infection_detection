# Postpartum Infection Detection

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-green)](https://scikit-learn.org/)

Detect postpartum infections using **Heart Rate (HR)** and **Body Temperature (Temp)** with machine learning. Includes pre-trained pipelines ready for deployment or integration in workflows like **n8n**.

---

## 🔹 Dataset

* Key features: `Body Temp (°F)`, `Heart Rate (bpm)`, `Risk Level`, plus optional health metrics (Age, BP, BMI, Diabetes, etc.).
* Infection proxy label:

  * `1` if Temp ≥ 38°C or HR ≥ 100 bpm
  * `0` otherwise

---

## 🔹 Workflow

1. Load and inspect dataset
2. Preprocess data (temperature conversion, infection label creation)
3. Train/Test split (80/20) + scaling
4. Train baseline models: Logistic Regression, Random Forest
5. Evaluate models: precision, recall, F1, ROC-AUC, confusion matrix
6. Hyperparameter tuning for Random Forest (GridSearchCV)
7. Save pipelines (preprocessing + model)
8. Test pipelines and predict new patient readings

---

## 🔹 Example Performance

| Model                 | Accuracy | Precision     | Recall        | F1-Score      | ROC-AUC |
| --------------------- | -------- | ------------- | ------------- | ------------- | ------- |
| Logistic Regression   | 0.828    | 0.854 / 0.787 | 0.860 / 0.779 | 0.857 / 0.783 | 0.843   |
| Random Forest         | 0.849    | 0.869 / 0.817 | 0.881 / 0.800 | 0.875 / 0.808 | 0.897   |
| Random Forest (Tuned) | 0.849    | 0.874 / 0.811 | 0.874 / 0.811 | 0.874 / 0.811 | 0.894   |

> Logistic Regression is interpretable; Random Forest captures non-linear relationships and is tuned for better recall.

---

## 🔹 Saved Pipelines

* `logreg_pipeline.pkl` – Logistic Regression pipeline
* `rf_pipeline.pkl` – Tuned Random Forest pipeline

> Pipelines include preprocessing, so you can **load and predict** without manual scaling.

---

## 🔹 Usage Example

```python
import joblib
import pandas as pd

# Load pipeline
pipe = joblib.load("logreg_pipeline.pkl")

# Predict new patient
X_new = pd.DataFrame({"Heart Rate": [72], "temp_c": [37.8]})
pred = pipe.predict(X_new)
prob = pipe.predict_proba(X_new)[:, 1]

print("Prediction:", pred)
print("Infection probability:", prob)
```

---

## 🔹 Notes

* Currently uses only HR and Temp; more features can improve accuracy.
* Pipelines are ready for **deployment or integration** in automation workflows.
