import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import joblib

# 1. Load data and include the new columns
dataFrame = pd.read_csv('winemag-data-130k-v2.csv')
dataFrame = dataFrame[['variety', 'region_1', 'province', 'winery', 'points', 'price']]
dataFrame = dataFrame.dropna()

# 2. Filter for top 10 (keep it simple)
top_varieties = dataFrame['variety'].value_counts().nlargest(10).index
top_regions = dataFrame['region_1'].value_counts().nlargest(10).index
dataFrame = dataFrame[dataFrame['variety'].isin(top_varieties) & dataFrame['region_1'].isin(top_regions)]

# 3. Create AND SAVE all 4 encoders
encoders = {}
for col in ['variety', 'region_1', 'province', 'winery']:
    le = LabelEncoder()
    dataFrame[f'{col}_encoded'] = le.fit_transform(dataFrame[col])
    encoders[col] = le
    joblib.dump(le, f'{col}_encoder.pkl') # Save each encoder

# 4. Train Price Estimator (NO POINTS)
X_price = dataFrame[['variety_encoded', 'region_1_encoded', 'province_encoded', 'winery_encoded']]
y_price = dataFrame['price']

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_price, y_price, test_size=0.2, random_state=42)
price_model = RandomForestRegressor(n_estimators=100, random_state=42)
price_model.fit(X_train_p, y_train_p)
joblib.dump(price_model, 'xgboost_price_model.pkl') # Saved as the name main.py expects
print(f"Price Model R^2: {price_model.score(X_test_p, y_test_p):.2f}")

# 5. Train Quality Classifier
dataFrame['is_good_quality'] = (dataFrame['points'] >= 90).astype(int)
X_quality = dataFrame[['variety_encoded', 'region_1_encoded', 'price']]
y_quality = dataFrame['is_good_quality']

X_train_q, X_test_q, y_train_q, y_test_q = train_test_split(X_quality, y_quality, test_size=0.2, random_state=42)
quality_model = RandomForestClassifier(n_estimators=100, random_state=42)
quality_model.fit(X_train_q, y_train_q)
joblib.dump(quality_model, 'quality_model.pkl')
print(f"Quality Model Accuracy: {quality_model.score(X_test_q, y_test_q):.2f}")

print("Training Complete! Models and all 4 encoders saved.")