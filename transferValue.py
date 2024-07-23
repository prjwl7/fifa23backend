import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error
import numpy as np

url = 'https://drive.google.com/file/d/1ZDEEBvZIPGDI0hyxi7WPhcRxEi7r7AEm/view?usp=sharing'
path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
data = pd.read_csv(path ,dtype='object')
features = ["overall", "potential", "age", "value_eur", "pace", "shooting", "passing", "dribbling", "defending", "physic"]
data = data[features]


data[features] = data[features].apply(pd.to_numeric, errors='coerce')

# Handle missing values only for numeric columns
data[features] = data[features].fillna(data.mean())

data["value_eur"].describe()

# Apply log transformation to target variable if it is skewed
data["value_eur"] = np.log1p(data["value_eur"])


# Separate features and target
X = data.drop("value_eur", axis=1)
y = data["value_eur"]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Feature Selection (try different values for k)
k_values = [5, 10, 15]  # Experiment with different numbers of features
best_mse = float('inf')
best_k = 0

for k in k_values:
    selector = SelectKBest(f_regression, k=k)
    X_selected = selector.fit_transform(X_scaled, y)
    selected_features = [X.columns[i] for i in selector.get_support(indices=True)]
    print(f"Selected Features with k={k}: {selected_features}")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    # Train a Random Forest Regression model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model performance
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error with k={k}: {mse}")

    # Track the best k based on MSE
    if mse < best_mse:
        best_mse = mse
        best_k = k

print(f"Best k value: {best_k} with MSE: {best_mse}")


# Use the best k value for final model
selector = SelectKBest(f_regression, k=best_k)
X_selected = selector.fit_transform(X_scaled, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Train a Random Forest Regression model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluate model performance
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Final Mean Squared Error: {mse}")


def predict_market_value(input_data):
    
    # Example feature values for prediction
    input_df = pd.DataFrame([input_data])

    # Scale the input features using the same scaler
    input_scaled = scaler.transform(input_df)

    # Select the same features used during training
    input_selected = selector.transform(input_scaled)

    # Make predictions using the trained model
    predicted_value = model.predict(input_selected)

    # If log transformation was applied, convert back
    predicted_value = np.expm1(predicted_value)

    return predicted_value[0]

