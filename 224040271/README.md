# SVM Classification on Iris Dataset

This project demonstrates how to build a multi-class classification model using Support Vector Machines (SVM) and ChatGPT-4 on the classic Iris dataset.

---

## 📊 Dataset

- **Name**: Iris Dataset
- **Samples**: 150
- **Classes**: 
  - Iris setosa
  - Iris versicolor
  - Iris virginica
- **Features**: 
  - Sepal length (cm)
  - Sepal width (cm)
  - Petal length (cm)
  - Petal width (cm)

---

## 🚀 Project Workflow

1. **Load and preprocess the data**
   - Stratified train-test split
   - Feature normalization using `StandardScaler`

2. **Model Training**
   - Classifier: `SVC` (Support Vector Classifier), ChatGPT-4o/mini
   - Hyperparameter tuning via `GridSearchCV`:
     - `C`, `gamma`, `kernel`, `degree`
    

3. **Model Evaluation**
   - Accuracy
   - Classification report (precision, recall, f1-score)
   - Visualization of report as a table

4. **Output**
   - Best hyperparameters
   - Summary performance visualization

---

## 📈 Visualizations

- Tabular plot of classification metrics (with accuracy included)

---

## 🛠 Requirements

- Python 3.x
- scikit-learn
- random
- numpy
- openai

---

## ✅ Results

- Achieved high classification accuracy on the test set
- Best parameters selected via cross-validation
- Full evaluation metrics visualized for interpretability

---
