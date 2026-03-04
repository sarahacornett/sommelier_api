import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import joblib

trainingDataCSV = 'winemag-data-130k-v2.csv'

# Load the data
dataFrame = pd.read_csv(trainingDataCSV)

# Keeping only the columns we care about. Points are the average score from WineEnthusiest (1-100) 
dataFrame = dataFrame[['variety', 'region_1', 'points', 'price']]

# Dropping the rows with missing data
dataFrame = dataFrame.dropna()

# Filter for top 10 varieties and regions to keep the app simple, for now
top_varieties = dataFrame['variety'].value_counts().nlargest(10).index
top_regions = dataFrame['region_1'].value_counts().nlargest(10).index

# reconfigure the data frame to only contain top 10 values
dataFrame = dataFrame[dataFrame['variety'].isin(top_varieties) & dataFrame['region_1'].isin(top_regions)]




# Encoding Text

# Initialize encoders
# converts labels into numerical values for the model to use.
variety_encoder = LabelEncoder()
region_encoder = LabelEncoder()

# Fit and transform the text columns into numbers
dataFrame['variety_encoded'] = variety_encoder.fit_transform(dataFrame['variety'])
dataFrame['region_encoded'] = region_encoder.fit_transform(dataFrame['region_1'])

# Define our input features (X)
X = dataFrame[['variety_encoded', 'region_encoded', 'points']]




# Train Model A: Price Estimator
# Using Random Forest Algorithm for regression to predict the price of the wine based on its variety, region, and points.
# Target variable for Model A
y_price = dataFrame['price']

# Split data into training and testing sets
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X, y_price, test_size=0.2, random_state=42)

print("Training Price Estimator...")
price_model = RandomForestRegressor(n_estimators=100, random_state=42)
price_model.fit(X_train_p, y_train_p)
print(f"Price Model Accuracy (R^2): {price_model.score(X_test_p, y_test_p):.2f}")




# Train Model B: Quality Classifier
# We will create a binary classification model to predict if a wine is of "good quality" (points >= 90) or not, based on its variety, region, and price.
# Create a new binary target variable: 1 if points >= 90, else 0
dataFrame['is_good_quality'] = (dataFrame['points'] >= 90).astype(int)

# Input features for Quality (we don't use 'points' to predict 'is_good_quality'!)
X_quality = dataFrame[['variety_encoded', 'region_encoded', 'price']]
y_quality = dataFrame['is_good_quality']

X_train_q, X_test_q, y_train_q, y_test_q = train_test_split(X_quality, y_quality, test_size=0.2, random_state=42)

print("Training Quality Classifier...")
quality_model = RandomForestClassifier(n_estimators=100, random_state=42)
quality_model.fit(X_train_q, y_train_q)
print(f"Quality Model Accuracy: {quality_model.score(X_test_q, y_test_q):.2f}")




# Save the models
joblib.dump(price_model, 'price_model.pkl')
joblib.dump(quality_model, 'quality_model.pkl')

# Save the encoders
joblib.dump(variety_encoder, 'variety_encoder.pkl')
joblib.dump(region_encoder, 'region_encoder.pkl')

print("Day 1 Complete! Models and encoders saved successfully.")