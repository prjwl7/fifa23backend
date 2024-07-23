import pandas as pd
from flask import jsonify
import re
from scipy.spatial.distance import cdist
url = 'https://drive.google.com/file/d/1ZDEEBvZIPGDI0hyxi7WPhcRxEi7r7AEm/view?usp=sharing'
path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
data = pd.read_csv(path ,dtype='object')

def get_player_data(fifa_version, long_name):
    if not long_name or not fifa_version:
        return jsonify({'error': 'Missing parameters: long_name or fifa_version'})
    filtered_data = data[(data['long_name'] == long_name) & (data['fifa_version'] == fifa_version)]

    if filtered_data.empty:
        return jsonify({'error': f'Player with long_name "{long_name}" and FIFA version "{fifa_version}" not found'})
    player_data = filtered_data.iloc[0].to_dict()
    player_data = {k: (None if pd.isna(v) else v) for k, v in player_data.items()}
    return jsonify(player_data)

def get_matched_long_names(query):
    if not query:
        return jsonify({'error': 'Missing query parameter'})
    query_words = query.split()
    #regex_parts = [rf"(?=.*\b{re.escape(word)}\b)" for word in query_words]
    regex_pattern = re.compile(rf"\b\w*{re.escape(query)}\w*\b", re.IGNORECASE)
    matched_players = data[data['long_name'].str.contains(regex_pattern, regex=True)]
    # Extract unique matched long names
    matched_long_names = matched_players['long_name'].unique().tolist()

    return jsonify({'matched_long_names': matched_long_names})

def get_player_overall_ratings(long_name):
    # Filter the dataset based on the player's long name
    player_data1 = data[data['long_name'] == long_name]

    if player_data1.empty:
        return {'error': 'Player not found'}, 404
    
    player_data = player_data1[['fifa_version', 'overall']].sort_values(by='fifa_version')
    player_list = player_data.to_dict(orient='records')
    player_data_value = player_data1[['fifa_version', 'value_eur']].sort_values(by='fifa_version').to_dict(orient='records')
    
    return player_list, player_data_value, 200


def find_similar_players(long_name, top_n=5):
    player = data[data['long_name'] == long_name]
    
    if player.empty:
        return {'error': 'Player not found'}, 404

    player_attributes = player[['pace', 'shooting', 'dribbling', 'passing', 'defending', 'physic']].values
    other_players = data[data['long_name'] != long_name]
    other_players_attributes = other_players[['long_name', 'pace', 'shooting', 'dribbling', 'passing', 'defending', 'physic']]

    distances = cdist(player_attributes, other_players_attributes.iloc[:, 1:], 'euclidean')
    other_players['distance'] = distances[0]

    similar_players = other_players.nsmallest(top_n, 'distance')[['long_name', 'distance']].to_dict(orient='records')
    
    return similar_players, 200