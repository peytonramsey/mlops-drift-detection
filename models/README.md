# Model Artifacts

## Large Files Not Included in Git

The following files are excluded from git due to GitHub's 100MB file size limit:

- `best_model_real_features.pkl` (206MB) - The trained Random Forest model
- `best_model.pkl` (299KB) - Alternative model
- Training data CSVs (101MB+)

## How to Get the Model

### Option 1: Train the Model Yourself

```bash
# 1. Get the dataset (place in data/raw/)
# Download from: [Add your dataset source]

# 2. Run preprocessing
python src/models/data_preprocessing_no_indicators.py

# 3. Train the model
python src/models/train_real_features.py
```

### Option 2: Download Pre-trained Model

Download the pre-trained model from:
- **Google Drive**: [Add link if you upload it]
- **Hugging Face**: [Add link if you upload it]
- **Release Assets**: Check GitHub releases

Place the downloaded files in the `models/` directory.

## Files Included in Git

The following smaller files ARE version controlled:

✅ `scaler_no_indicators.pkl` (3KB) - Feature scaler
✅ `numerical_medians_no_indicators.pkl` (244 bytes) - Median imputation values
✅ `categorical_modes_no_indicators.pkl` (146 bytes) - Mode imputation values
✅ `baseline_stats.json` (28KB) - Baseline statistics for drift detection
✅ `feature_names.json` (1KB) - Feature column names

## Alternative: Use Docker

The Docker image can include the model:

```bash
# Build with local model
docker-compose build

# The model will be copied into the container
```

## Model Specifications

- **Algorithm**: Random Forest Classifier
- **Estimators**: 300 trees
- **Max Depth**: 20
- **Features**: 51
- **Size**: 206 MB
- **Performance**: 88.93% accuracy, 73.52% F1 score
