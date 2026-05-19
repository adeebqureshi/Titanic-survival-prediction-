# Titanic Survival Prediction

Machine Learning project to predict passenger survival on the Titanic dataset using feature engineering, preprocessing pipelines, classification models, and explainable AI techniques.

---

## Features

- Title extraction from passenger names
- Family size feature creation
- Cabin presence detection
- Missing value handling
- Categorical encoding
- Model comparison
- SHAP explainability
- Model saving and inference pipeline

---

## Models Used

- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib
- Seaborn
- Joblib

---

## Project Structure

```text
Titanic-Survival-Prediction/
│
├── titanic.csv
├── Titanic_Survival_Prediction.ipynb
├── titanic_survival_model.pkl
├── README.md
└── requirements.txt
```

---

## Installation

```bash
pip install pandas numpy scikit-learn xgboost shap seaborn matplotlib joblib
```

---

## Run Project

```bash
jupyter notebook
```

Open:

```text
Titanic_Survival_Prediction.ipynb
```

---

## Example Inference

```python
import joblib
import pandas as pd

model = joblib.load("titanic_survival_model.pkl")

sample = pd.DataFrame({
    'Pclass': [1],
    'Sex': ['female'],
    'Age': [29],
    'SibSp': [0],
    'Parch': [0],
    'Fare': [100],
    'Embarked': ['C'],
    'Title': ['Miss'],
    'FamilySize': [1],
    'IsAlone': [1],
    'CabinPresent': [1]
})

prediction = model.predict(sample)

print(prediction)
```

---

## Results

| Model | Accuracy |
|---|---|
| Logistic Regression | ~80% |
| Random Forest | ~84% |
| XGBoost | ~87% |

---

## Future Improvements

- Hyperparameter tuning
- Streamlit deployment
- FastAPI integration
- Docker support
- CI/CD pipeline
