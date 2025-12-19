# Week 1: Foundation - Detailed Action Plan

## Overview
By end of Week 1, you'll have:
- ✅ Complete project structure
- ✅ Working ML training pipeline
- ✅ Baseline model with 85%+ accuracy
- ✅ MLflow experiment tracking
- ✅ Version control setup

---

## Day 1-2: Environment Setup & Project Structure

### Tasks:
1. **Create project directory and folder structure**
   ```bash
   mkdir mlops-drift-detection
   cd mlops-drift-detection

   # Create all folders
   mkdir -p data/raw data/processed
   mkdir -p src/models src/api src/monitoring src/retraining src/dashboard
   mkdir -p tests docker .github/workflows
   ```

2. **Initialize Git repository**
   ```bash
   git init
   # Create .gitignore for Python
   ```

3. **Create virtual environment**
   ```bash
   python -m venv venv
   # On Windows: venv\Scripts\activate
   # On Mac/Linux: source venv/bin/activate
   ```

4. **Create requirements.txt** (starter version)
   ```txt
   # Core ML
   pandas==2.1.4
   numpy==1.26.2
   scikit-learn==1.3.2
   xgboost==2.0.3

   # Experiment Tracking
   mlflow==2.9.2

   # Data Visualization
   matplotlib==3.8.2
   seaborn==0.13.0

   # Utilities
   python-dotenv==1.0.0
   pyyaml==6.0.1
   joblib==1.3.2

   # Development
   jupyter==1.0.0
   ipykernel==6.27.1
   ```

5. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

6. **Create empty Python files** (so structure is visible)
   ```bash
   # Windows PowerShell
   New-Item src/models/train.py
   New-Item src/models/predict.py
   New-Item src/models/model_utils.py
   New-Item src/__init__.py
   New-Item src/models/__init__.py

   # Or use: type nul > filename.py
   ```

### Research Topics:
- **MLflow Basics**: What is experiment tracking? How does MLflow organize runs?
  - Read: MLflow Quickstart (https://mlflow.org/docs/latest/quickstart.html)
  - Understand: Experiments, Runs, Parameters, Metrics, Artifacts

- **Project Structure Best Practices**
  - Why separate `src/` from `data/` and `tests/`?
  - What is the purpose of `__init__.py` files?

---

## Day 3: Data Acquisition & Exploration

### Tasks:
1. **Download Telco Customer Churn dataset**
   - Option A: Kaggle (https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
   - Option B: IBM Sample Data (search "IBM Telco Customer Churn")
   - Save to `data/raw/telco_churn.csv`

2. **Create exploratory notebook**
   ```bash
   jupyter notebook
   # Create: notebooks/01_data_exploration.ipynb
   ```

3. **Perform EDA (Exploratory Data Analysis)**
   In the notebook:
   ```python
   import pandas as pd
   import matplotlib.pyplot as plt
   import seaborn as sns

   # Load data
   df = pd.read_csv('../data/raw/telco_churn.csv')

   # Basic exploration
   df.head()
   df.info()
   df.describe()

   # Check target distribution
   df['Churn'].value_counts()

   # Check missing values
   df.isnull().sum()

   # Visualize correlations
   # Identify categorical vs numerical features
   # Check for imbalanced classes
   ```

4. **Document findings**
   Create a markdown cell with:
   - Number of samples and features
   - Target variable distribution (is it imbalanced?)
   - Missing values (how to handle?)
   - Feature types (categorical vs numerical)
   - Potential issues (outliers, skewed distributions)

### Research Topics:
- **Binary Classification Fundamentals**
  - What is precision vs recall vs F1 score?
  - Why does class imbalance matter?
  - What is AUC-ROC and why is it useful?

- **Data Preprocessing for ML**
  - When to use one-hot encoding vs label encoding?
  - Why standardize/normalize numerical features?
  - How to handle missing values? (mean imputation, median, mode, drop)

- **Churn Prediction Domain Knowledge**
  - What features typically predict churn? (tenure, contract type, monthly charges)
  - Why is churn prediction valuable for businesses?

---

## Day 4-5: Feature Engineering & Data Preprocessing

### Tasks:
1. **Create data preprocessing script: `src/models/data_preprocessing.py`**
   ```python
   import pandas as pd
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import StandardScaler, LabelEncoder

   def load_and_preprocess_data(file_path):
       """
       Load data and perform preprocessing.
       Returns: X_train, X_val, X_test, y_train, y_val, y_test
       """
       df = pd.read_csv(file_path)

       # Handle missing values
       # Convert categorical variables
       # Feature engineering (e.g., tenure_months, avg_charge_per_month)
       # Encode target variable

       return preprocessed_data

   def split_data(X, y, test_size=0.2, val_size=0.1, random_state=42):
       """
       Split data into train/val/test sets.
       Train: 70%, Val: 10%, Test: 20%
       """
       # Split into train+val and test
       X_temp, X_test, y_temp, y_test = train_test_split(
           X, y, test_size=test_size, random_state=random_state, stratify=y
       )

       # Split train+val into train and val
       val_ratio = val_size / (1 - test_size)
       X_train, X_val, y_train, y_val = train_test_split(
           X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
       )

       return X_train, X_val, X_test, y_train, y_val, y_test
   ```

2. **Implement feature engineering**
   - Handle categorical variables (one-hot encoding for nominal, label encoding for ordinal)
   - Create new features if valuable (e.g., `TotalCharges / tenure`)
   - Scale numerical features using StandardScaler
   - Encode target variable (Yes/No → 1/0)

3. **Save preprocessed data**
   Save train/val/test splits to `data/processed/`:
   ```python
   X_train.to_csv('data/processed/X_train.csv', index=False)
   y_train.to_csv('data/processed/y_train.csv', index=False)
   # Same for val and test
   ```

4. **Save preprocessing artifacts**
   Save scaler and encoders for later use in production:
   ```python
   import joblib
   joblib.dump(scaler, 'models/scaler.pkl')
   joblib.dump(label_encoders, 'models/label_encoders.pkl')
   ```

### Research Topics:
- **Train/Validation/Test Split Strategy**
  - Why do we need 3 splits instead of just train/test?
  - What is stratification and why use it for imbalanced data?
  - Typical split ratios: 70/10/20 or 60/20/20

- **Feature Scaling**
  - StandardScaler vs MinMaxScaler vs RobustScaler - when to use each?
  - Why do tree-based models (Random Forest, XGBoost) not require scaling?
  - Why do linear models require scaling?

- **Handling Categorical Variables**
  - One-hot encoding: creates binary columns for each category
  - Label encoding: converts categories to numbers (0, 1, 2, ...)
  - When to use each approach?
  - Beware: label encoding can introduce false ordinal relationships

---

## Day 6-7: Model Training & MLflow Integration

### Tasks:
1. **Install and set up MLflow**
   ```bash
   # Already in requirements.txt, but verify
   pip install mlflow

   # Test MLflow UI
   mlflow ui
   # Opens at http://localhost:5000
   ```

2. **Create training script: `src/models/train.py`**
   ```python
   import mlflow
   import mlflow.sklearn
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
   import pandas as pd

   def train_model(X_train, y_train, X_val, y_val, hyperparameters):
       """
       Train a Random Forest model and log to MLflow.
       """
       # Set MLflow experiment
       mlflow.set_experiment("churn-prediction-baseline")

       with mlflow.start_run():
           # Log hyperparameters
           mlflow.log_params(hyperparameters)

           # Train model
           model = RandomForestClassifier(**hyperparameters)
           model.fit(X_train, y_train)

           # Make predictions
           y_pred = model.predict(X_val)
           y_pred_proba = model.predict_proba(X_val)[:, 1]

           # Calculate metrics
           accuracy = accuracy_score(y_val, y_pred)
           precision = precision_score(y_val, y_pred)
           recall = recall_score(y_val, y_pred)
           f1 = f1_score(y_val, y_pred)
           auc = roc_auc_score(y_val, y_pred_proba)

           # Log metrics
           mlflow.log_metric("accuracy", accuracy)
           mlflow.log_metric("precision", precision)
           mlflow.log_metric("recall", recall)
           mlflow.log_metric("f1_score", f1)
           mlflow.log_metric("auc_roc", auc)

           # Log model
           mlflow.sklearn.log_model(model, "model")

           print(f"Accuracy: {accuracy:.4f}")
           print(f"AUC-ROC: {auc:.4f}")

           return model

   if __name__ == "__main__":
       # Load preprocessed data
       X_train = pd.read_csv('data/processed/X_train.csv')
       y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
       X_val = pd.read_csv('data/processed/X_val.csv')
       y_val = pd.read_csv('data/processed/y_val.csv').values.ravel()

       # Define hyperparameters
       hyperparameters = {
           'n_estimators': 100,
           'max_depth': 10,
           'min_samples_split': 5,
           'random_state': 42
       }

       # Train model
       model = train_model(X_train, y_train, X_val, y_val, hyperparameters)
   ```

3. **Run training script**
   ```bash
   python src/models/train.py
   ```

4. **View results in MLflow UI**
   ```bash
   mlflow ui
   # Navigate to http://localhost:5000
   # Explore experiments, compare runs, view metrics
   ```

5. **Experiment with hyperparameters**
   Run training multiple times with different hyperparameters:
   - Vary `n_estimators`: 50, 100, 200
   - Vary `max_depth`: 5, 10, 15, None
   - Try XGBoost if Random Forest doesn't reach 85% accuracy

   Compare results in MLflow UI to find best configuration.

6. **Save best model**
   ```python
   # Register best model to MLflow Model Registry
   # Or save manually:
   import joblib
   joblib.dump(best_model, 'models/baseline_model.pkl')
   ```

### Research Topics:
- **Random Forest vs XGBoost**
  - How do ensemble methods work?
  - Random Forest: bagging (parallel trees with random subsets)
  - XGBoost: boosting (sequential trees that correct previous errors)
  - When to use each?

- **Hyperparameter Tuning**
  - What is a hyperparameter vs a model parameter?
  - Common hyperparameters for Random Forest:
    - `n_estimators`: number of trees
    - `max_depth`: maximum tree depth (prevents overfitting)
    - `min_samples_split`: minimum samples to split a node
  - GridSearchCV vs RandomSearchCV for automated tuning

- **MLflow Concepts**
  - Experiment: collection of runs for a specific task
  - Run: single execution of model training with specific parameters
  - Artifacts: files logged during a run (model, plots, data)
  - Model Registry: centralized store for production models

- **Evaluation Metrics for Binary Classification**
  - Accuracy: (TP + TN) / Total - can be misleading for imbalanced data
  - Precision: TP / (TP + FP) - of predicted positives, how many are correct?
  - Recall: TP / (TP + FN) - of actual positives, how many did we find?
  - F1 Score: harmonic mean of precision and recall
  - AUC-ROC: area under ROC curve - measures model's ability to distinguish classes
  - When to optimize for precision vs recall?

---

## Deliverables Checklist

By end of Week 1, you should have:

- [ ] Project directory with complete folder structure
- [ ] Git repository initialized with .gitignore
- [ ] Virtual environment with all dependencies installed
- [ ] requirements.txt with core ML packages
- [ ] EDA notebook documenting dataset characteristics
- [ ] `data_preprocessing.py` with feature engineering pipeline
- [ ] Preprocessed train/val/test splits saved to `data/processed/`
- [ ] `train.py` script with MLflow integration
- [ ] Baseline model trained with 85%+ accuracy
- [ ] At least 5 MLflow runs comparing different hyperparameters
- [ ] MLflow UI running and accessible
- [ ] Best model saved to `models/` directory
- [ ] README.md with setup instructions (start documenting!)

---

## Common Issues & Solutions

### Issue 1: Dataset has "TotalCharges" as object instead of float
**Solution:** Some values may be empty strings. Convert to numeric:
```python
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
```

### Issue 2: Model accuracy below 85%
**Solutions:**
- Check class imbalance - if severe (e.g., 90/10 split), try:
  - SMOTE for oversampling minority class
  - Class weights: `RandomForestClassifier(class_weight='balanced')`
- Try XGBoost instead of Random Forest
- Feature engineering: create interaction features
- Hyperparameter tuning with more iterations

### Issue 3: MLflow UI not showing experiments
**Solution:**
- Ensure `mlflow.set_experiment()` is called before `mlflow.start_run()`
- Check that `mlruns/` directory exists
- Restart MLflow UI: `mlflow ui --backend-store-uri ./mlruns`

### Issue 4: Import errors when running scripts
**Solution:**
- Add project root to PYTHONPATH:
  ```bash
  # Windows
  set PYTHONPATH=%PYTHONPATH%;C:\path\to\mlops-drift-detection

  # Mac/Linux
  export PYTHONPATH="${PYTHONPATH}:/path/to/mlops-drift-detection"
  ```
- Or run from project root with: `python -m src.models.train`

---

## Next Steps Preview (Week 2)

Once Week 1 is complete, you'll move to:
- Building FastAPI REST API for model serving
- Creating prediction endpoint that accepts JSON requests
- Implementing request/response validation with Pydantic
- Setting up PostgreSQL for logging predictions
- Loading trained model from MLflow

---

## Recommended Learning Resources

**MLflow:**
- Official docs: https://mlflow.org/docs/latest/index.html
- Tutorial: "MLflow in 10 Minutes" (YouTube)

**Binary Classification:**
- StatQuest: "Random Forests Part 1 & 2" (YouTube)
- Google's ML Crash Course: Classification section

**Feature Engineering:**
- Book: "Feature Engineering for Machine Learning" by Alice Zheng
- Article: "Practical Guide to Feature Engineering" (Medium/Towards Data Science)

**Python Project Structure:**
- Article: "Structuring Your Project" (python-guide.org)
- Example: Cookiecutter Data Science template

---

## Success Criteria for Week 1

You've successfully completed Week 1 if you can:
1. ✅ Run `python src/models/train.py` and train a model
2. ✅ Open MLflow UI and see your experiment runs
3. ✅ Achieve validation accuracy >= 85%
4. ✅ Explain what each metric (precision, recall, F1, AUC-ROC) measures
5. ✅ Describe your feature engineering choices
6. ✅ Load a saved model and make predictions on test set

---

## Time Estimates

- Day 1-2 (Setup): 4-6 hours
- Day 3 (EDA): 3-4 hours
- Day 4-5 (Preprocessing): 6-8 hours
- Day 6-7 (Training & MLflow): 6-8 hours

**Total: ~20-26 hours** (manageable for 1 week with 3-4 hours/day)

---

Good luck! Focus on understanding each step rather than rushing through. Week 1 lays the foundation for everything else.
