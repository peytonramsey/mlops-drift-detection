import pandas as pd
import json

X_train = pd.read_csv('data/processed_no_indicators/X_train.csv')
features = list(X_train.columns)

with open('models/feature_names.json', 'w') as f:
    json.dump(features, f, indent=2)

print(f'Saved {len(features)} feature names to models/feature_names.json')
print('\nFirst 10 features:')
for i, feat in enumerate(features[:10]):
    print(f'  {i+1}. {feat}')
