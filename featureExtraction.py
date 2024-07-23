import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error

# Load data (replace with your actual dataset path)
url = 'https://drive.google.com/file/d/1ZDEEBvZIPGDI0hyxi7WPhcRxEi7r7AEm/view?usp=sharing'
path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
data = pd.read_csv(path ,dtype='object')

# Define columns of interest
attributes = [
    'overall', 'value_eur'
]

# Filter data to include only numeric attributes
data = data[attributes]

# Convert relevant columns to numeric
data = data.apply(pd.to_numeric, errors='coerce')

# Handle missing values
data = data.fillna(data.mean())
print(data.columns)
# Separate features (X) and target (y)
X = data.drop(columns=['value_eur'])
y = data['value_eur']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Feature selection using SelectKBest with f_regression
selector = SelectKBest(score_func=f_regression, k='all')
selector.fit(X_scaled, y)

# Get selected feature scores and names
selected_features_scores = selector.scores_
selected_features_names = X.columns[selector.get_support(indices=True)]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a RandomForestRegressor
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluate model performance
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Feature importances from RandomForestRegressor
feature_importances = pd.DataFrame(model.feature_importances_, index=X.columns, columns=['importance'])
feature_importances_sorted = feature_importances.sort_values(by='importance', ascending=False)

print("\nTop contributing features to value_eur:")
print(feature_importances_sorted.head(10))
