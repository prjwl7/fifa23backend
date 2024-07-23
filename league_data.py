import pandas as pd

# Initialize an empty DataFrame to hold the aggregated results
league_metrics = pd.DataFrame()

# Define a function to process each chunk
def process_chunk(chunk):
    chunk['overall'] = chunk['overall'].astype(int)
    chunk['value_eur'] = chunk['value_eur'].astype(float)
    league_agg = chunk.groupby('league_name').agg({
        'overall': 'mean',
        'value_eur': 'mean'
    })
    return league_agg

# Read the CSV file in chunks
chunk_size = 10000
for chunk in pd.read_csv('./fifa23dataset/male_players.csv', chunksize=chunk_size, dtype='object'):
    chunk_metrics = process_chunk(chunk)
    league_metrics = league_metrics.add(chunk_metrics, fill_value=0)

# Calculate the final averages
league_metrics = league_metrics.div(len(league_metrics))
league_metrics.reset_index(inplace=True)
league_metrics.rename(columns={'overall': 'avg_overall', 'value_eur': 'avg_market_value'}, inplace=True)

# Filter top 5 leagues based on avg_overall
top_5_leagues = league_metrics.sort_values(by='avg_overall', ascending=False).head(5)

# Save the top 5 leagues to a CSV file
top_5_leagues.to_csv('./fifa23dataset/top_5_leagues.csv', index=False)

# Define specific positions to filter
positions = ['CF', 'ST', 'LW', 'RW']

# Initialize an empty DataFrame to hold the top players
top_players = pd.DataFrame()

# Define a function to expand multiple positions
def expand_positions(df):
    rows = []
    for _, row in df.iterrows():
        positions = row['club_position'].split(', ')
        for position in positions:
            new_row = row.copy()
            new_row['club_position'] = position
            rows.append(new_row)
    return pd.DataFrame(rows)

# Define a function to get top players for each position for each league
def get_top_players(chunk, top_leagues):
    top_league_players = chunk[chunk['league_name'].isin(top_leagues)].copy()
    top_league_players['overall'] = top_league_players['overall'].astype(int)
    expanded_players = expand_positions(top_league_players)
    top_players_by_position = expanded_players.groupby(['league_name', 'club_position']).apply(lambda x: x.loc[x['overall'].idxmax()])
    return top_players_by_position

# Read the CSV file in chunks and process top players
for chunk in pd.read_csv('./fifa23dataset/male_players.csv', chunksize=chunk_size, dtype='object'):
    chunk_players = get_top_players(chunk, top_5_leagues['league_name'])
    top_players = pd.concat([top_players, chunk_players])

# Filter unique positions from the dataset
unique_positions = top_players['club_position'].unique()

# Combine specific positions with unique positions
all_positions = list(unique_positions.tolist())

# Filter top players for all positions
filtered_top_players = top_players[top_players['club_position'].isin(all_positions)]

# Reset index and save to CSV
filtered_top_players.reset_index(drop=True, inplace=True)
filtered_top_players.to_csv('./fifa23dataset/top_players_by_position.csv', index=False)
