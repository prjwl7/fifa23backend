import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
import joblib

def run_script():
    url = 'https://drive.google.com/file/d/1ZDEEBvZIPGDI0hyxi7WPhcRxEi7r7AEm/view?usp=sharing'
    path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
    data = pd.read_csv(path ,dtype='object')    
    features = ["overall", "potential", "age", "value_eur", "pace", "shooting", "passing", "dribbling", "defending", "physic"]
    data = data[features]

    data[features] = data[features].apply(pd.to_numeric, errors='coerce')
    data[features] = data[features].fillna(data.mean())

    data["value_eur"] = np.log1p(data["value_eur"])

    X = data.drop("value_eur", axis=1)
    y = data["value_eur"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    selector = SelectKBest(f_regression, k=10)
    X_selected = selector.fit_transform(X_scaled, y)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_selected, y)

    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(selector, 'selector.pkl')
    joblib.dump(model, 'model.pkl')

run_script()
