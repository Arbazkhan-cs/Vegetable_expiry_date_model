import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler

# Step 1: Load the data
df = pd.read_csv('vegetable_expiry_data.csv')

# Step 2: Preprocessing
# Handle missing values (if any)
df = df.dropna()

# Features and target variable
X = df[["Temperature (°C)", "Humidity (%)", "pH", "Microbial Count (CFU/g)"]]
y = df["Expiry Date (Days)"]

# Standardize the features (optional but recommended for some models)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 4: Model - Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred)}")
