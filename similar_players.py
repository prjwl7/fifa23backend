# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import NearestNeighbors
# import pickle

# # Load the scaler
# with open('scaler.pkl', 'rb') as f:
#     scaler = pickle.load(f)

# # Load and preprocess the original dataset
# data = pd.read_csv('./fifa23dataset/male_players (legacy).csv', dtype='object')
# features = ["overall", "potential", "age", "pace", "shooting", "passing", "dribbling", "defending", "physic"]

# # Convert relevant columns to numeric
# data[features] = data[features].apply(pd.to_numeric, errors='coerce')

# # Handle missing values
# data[features] = data[features].fillna(data.mean())

# # Scale the features
# data[features] = scaler.transform(data[features])

# # Function to find similar players
# def find_similar_players(input_data, num_neighbors=5):
#     # Convert input data to DataFrame
#     input_df = pd.DataFrame([input_data], columns=features)

#     # Scale the input features using the same scaler
#     input_scaled = scaler.transform(input_df)

#     # Use Nearest Neighbors to find similar players
#     nn = NearestNeighbors(n_neighbors=num_neighbors, algorithm='ball_tree', leaf_size=30)
#     nn.fit(data[features])

#     # Find similar players
#     distances, indices = nn.kneighbors(input_scaled)
#     similar_players = data.iloc[indices[0]]

#     return similar_players

# # Example input data
# input_data = {
#     "overall": 87,
#     "potential": 87,
#     "age": 28,
#     "pace": 79,
#     "shooting": 61,
#     "passing": 71,
#     "dribbling": 66,
#     "defending": 87,
#     "physic": 82
# }

# # Find similar players
# similar_players = find_similar_players(input_data)
# print("Similar Players:")
# print(similar_players)
