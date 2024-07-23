from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import subprocess
from flask_cors import CORS
from scipy.spatial.distance import euclidean
from search_player import get_matched_long_names, get_player_data, get_player_overall_ratings, find_similar_players
app = Flask(__name__)

CORS(app)
url = 'https://drive.google.com/file/d/1ZDEEBvZIPGDI0hyxi7WPhcRxEi7r7AEm/view?usp=sharing'
path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
data = pd.read_csv(path ,dtype='object')
def initialize_model():
    if not (os.path.exists('scaler.pkl') and os.path.exists('selector.pkl') and os.path.exists('model.pkl')):
        subprocess.run(["python", "marketValue.py"])

def predict_market_value(input_data):
    scaler = joblib.load('scaler.pkl')
    selector = joblib.load('selector.pkl')
    model = joblib.load('model.pkl')

    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    input_selected = selector.transform(input_scaled)
    predicted_value = model.predict(input_selected)
    predicted_value = np.expm1(predicted_value)
    return predicted_value[0]

@app.before_request
def before_first_request():
    initialize_model()

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.get_json()
    predicted_value = predict_market_value(input_data)
    print(f"Predicted Value: {predicted_value}")
    return jsonify({
        'predicted_value': predicted_value,
    })

@app.route('/api/search-player', methods=['GET'])
def search_player():
    long_name = request.args.get('long_name', None)
    fifa_version = request.args.get('fifa_version','23')
    player_data =  get_player_data(fifa_version, long_name)
    return player_data

@app.route('/api/matched-long-names', methods=['GET'])
def matched_long_names():
    query = request.args.get('query', None)
    return get_matched_long_names(query)

@app.route('/api/player-overall-ratings', methods=['GET'])
def player_overall_ratings():
    long_name = request.args.get('long_name')
    player_list, player_data_value, status_code = get_player_overall_ratings(long_name)
    if status_code == 404:
        return jsonify(player_list), status_code
    return jsonify({'player_list': player_list, 'player_data_value': player_data_value}), status_code



@app.route('/api/similar-players', methods=['GET'])
def similar_players():
    long_name = request.args.get('long_name')
    fifa_version = request.args.get('fifa_version')

    # Ensure the player exists in the specified FIFA version
    player_data = data[(data['long_name'] == long_name) & (data['fifa_version'] == fifa_version)]
    if player_data.empty:
        return jsonify({'error': 'Player not found'}), 404

    # Get the relevant attributes for similarity comparison
    player_attributes = ['pace', 'shooting', 'dribbling', 'passing', 'defending', 'physic']

    # Ensure all relevant columns are numeric
    data[player_attributes] = data[player_attributes].apply(pd.to_numeric, errors='coerce')

    # Filter data to include only players from the same FIFA version
    data_cleaned = data[(data['fifa_version'] == fifa_version)]

    # Drop rows with any NaN values in these columns
    data_cleaned = data_cleaned.dropna(subset=player_attributes)

    player_vector = player_data[player_attributes].values[0]

    # Compute Euclidean distances using numpy directly for efficiency
    data_cleaned.loc[:, 'distance'] = np.linalg.norm(data_cleaned[player_attributes].values - player_vector, axis=1)

    # Exclude the player himself/herself
    data_cleaned = data_cleaned[data_cleaned['distance'] > 0.0]

    # Get top 5 similar players
    similar_players = data_cleaned.nsmallest(5, 'distance')[['long_name', 'distance']]

    # Convert to a list of dictionaries
    similar_players_list = similar_players.to_dict(orient='records')

    return jsonify({'similar_players': similar_players_list}), 200



if __name__ == '__main__':
    app.run(debug=True)
