import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib

# Load the dataset
ames = fetch_openml(name="house_prices", as_frame=True)
data = ames.frame

# Select relevant features and target
features = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
target = 'SalePrice'

data = data[features + [target]].dropna()

# Feature scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[features])
data_scaled = pd.DataFrame(scaled_features, columns=features)
data_scaled[target] = data[target]

# Split the data into training and testing sets
X = data_scaled[features]
y = data_scaled[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model and scaler using joblib
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler saved successfully.")
