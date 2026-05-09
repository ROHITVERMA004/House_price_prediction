"""
House Price Prediction Web Application
Author: Sharda Vatsal Bhat
University Showcase Project
"""

import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load and prepare data
print("Loading dataset...")
df = pd.read_csv('train.csv')

# Selected features for the model
selected_features = [
    'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea',
    'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd',
    'YearBuilt', 'YearRemodAdd', 'Fireplaces', 'MasVnrArea', 'LotArea'
]

# Prepare data
data = df[selected_features + ['SalePrice']].copy()
for col in selected_features:
    if data[col].isnull().sum() > 0:
        data[col].fillna(data[col].median(), inplace=True)

# Train Random Forest model
X = data[selected_features]
y = data['SalePrice']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Calculate model metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_eval = RandomForestRegressor(n_estimators=100, random_state=42)
model_eval.fit(X_train, y_train)
y_pred_eval = model_eval.predict(X_test)

model_metrics = {
    'r2': round(r2_score(y_test, y_pred_eval), 4),
    'rmse': round(np.sqrt(mean_squared_error(y_test, y_pred_eval)), 2)
}

# Feature importance
feature_importance = pd.Series(model.feature_importances_, index=selected_features).sort_values(ascending=False)

# Dataset statistics
dataset_stats = {
    'total_houses': len(df),
    'avg_price': round(df['SalePrice'].mean(), 2),
    'min_price': df['SalePrice'].min(),
    'max_price': df['SalePrice'].max(),
    'total_features': len(df.columns)
}

@app.route('/')
def home():
    return render_template('index.html',
                           metrics=model_metrics,
                           stats=dataset_stats,
                           feature_importance=feature_importance)

@app.route('/predict')
def predict_page():
    return render_template('predict.html',
                           features=selected_features,
                           stats=dataset_stats)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extract features from request
        features = []
        for feat in selected_features:
            val = float(data.get(feat, 0))
            if val < 0:
                val = 0
            features.append(val)

        # Make prediction
        prediction = model.predict([features])[0]

        return jsonify({
            'success': True,
            'prediction': round(prediction, 2),
            'formatted': f"${prediction:,.2f}"
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html',
                           metrics=model_metrics,
                           stats=dataset_stats,
                           feature_importance=feature_importance)

if __name__ == '__main__':
    print("Starting House Price Prediction App...")
    print(f"Model R² Score: {model_metrics['r2']}")
    print(f"Model RMSE: ${model_metrics['rmse']:,.2f}")
    app.run(debug=True, port=5000)
