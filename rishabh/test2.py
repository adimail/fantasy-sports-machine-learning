import pandas as pd
import numpy as np
import pulp
from scipy.stats import zscore

# --------------------------------------------------
# 1. Data Loading and Cleaning
# --------------------------------------------------
# Load CSV file (ensure the file exists in your working directory)
df = pd.read_csv('players_with_roles.csv')

# Convert relevant columns to numeric; force errors to NaN if necessary
df['Runs_x'] = pd.to_numeric(df['Runs_x'], errors='coerce')
df['SR_x']   = pd.to_numeric(df['SR_x'], errors='coerce')
df['Econ']   = pd.to_numeric(df['Econ'], errors='coerce')
df['Wkts']   = pd.to_numeric(df['Wkts'], errors='coerce')
df['Mat_x']  = pd.to_numeric(df['Mat_x'], errors='coerce')

# Fill missing values with appropriate substitutes:
df['Runs_x'].fillna(df['Runs_x'].mean(), inplace=True)
df['SR_x'].fillna(df['SR_x'].mean(), inplace=True)
df['Econ'].fillna(df['Econ'].mean(), inplace=True)
df['Wkts'].fillna(0, inplace=True)
df['Mat_x'].fillna(df['Mat_x'].mean(), inplace=True)

# --------------------------------------------------
# 2. Compute Averages and Safe Z-Scores
# --------------------------------------------------
# Calculate per-match averages
df['bat_avg'] = df['Runs_x'] / df['Mat_x']     # Runs per match
df['bowl_avg'] = df['Wkts'] / df['Mat_x']        # Wickets per match

def safe_zscore(series):
    """
    Returns the z-score of a pandas Series. If the standard deviation is 0,
    returns a series of zeros.
    """
    std = series.std()
    if std == 0:
        return series - series.mean()
    else:
        return (series - series.mean()) / std

# Compute safe z-scores for our new metrics.
df['bat_avg_z'] = safe_zscore(df['bat_avg'])
df['SR_z']      = safe_zscore(df['SR_x'])
df['bowl_avg_z'] = safe_zscore(df['bowl_avg'])
df['Econ_z']    = safe_zscore(df['Econ'])

# For economy, since lower values are better, invert the z-score.
df['Econ_adj'] = -df['Econ_z']

# --------------------------------------------------
# 3. Define a Role-Specific Scoring Function
# --------------------------------------------------
# Set weights (adjust these if you want to favor one aspect over another)
batter_weight     = 1.0
bowler_weight     = 1.0
allrounder_weight = 1.0

def calculate_score(row):
    """
    Calculate a player score using per-match averages.
    - For batters: use batting average and strike rate.
    - For bowlers: use wickets per match and economy.
    - For all-rounders: combine both batting and bowling contributions.
    """
    role = row['Role'].strip().lower()
    if role == 'batter':
        # Higher runs per match and higher strike rate are better.
        return batter_weight * (row['bat_avg_z'] + row['SR_z'])
    elif role == 'bowler':
        # Higher wickets per match and lower economy (reflected in Econ_adj) are better.
        return bowler_weight * (row['bowl_avg_z'] + row['Econ_adj'])
    elif role == 'all-rounder':
        # Combine both batting and bowling contributions.
        return allrounder_weight * (row['bat_avg_z'] + row['SR_z'] + row['bowl_avg_z'] + row['Econ_adj'])
    else:
        return 0

# Calculate the overall score for each player and ensure it’s valid.
df['Score'] = df.apply(calculate_score, axis=1)
df['Score'] = df['Score'].replace([np.inf, -np.inf], 0)
df['Score'].fillna(0, inplace=True)

# --------------------------------------------------
# 4. Optimization: Select the Fantasy Team
# --------------------------------------------------
def optimize_team(team1, team2, total_players=11):
    """
    Optimize the fantasy team selection from players in team1 and team2.
    
    Parameters:
        team1 (str): Name of the first team.
        team2 (str): Name of the second team.
        total_players (int): Total number of players to select.
        
    Returns:
        A DataFrame with the selected players, their scores, and assigned roles
        (e.g., Captain, Vice Captain).
    """
    # Filter players belonging to the selected teams.
    team_df = df[df['Team'].isin([team1, team2])].copy()
    
    # Create the LP optimization problem.
    prob = pulp.LpProblem("FantasyTeam", pulp.LpMaximize)
    
    # Create binary decision variables for each player.
    players = team_df.index.tolist()
    x = pulp.LpVariable.dicts("player", players, cat='Binary')
    
    # Objective: maximize the total team score.
    prob += pulp.lpSum([x[i] * team_df.loc[i, 'Score'] for i in players]), "Total_Score"
    
    # Constraint: select exactly 'total_players' players.
    prob += pulp.lpSum([x[i] for i in players]) == total_players, "Total_Players"
    
    # Role constraints (adjust these as per your contest rules):
    # Example: Require at least 1 Batter and 1 Bowler (if available in the filtered set)
    batter_indices = team_df[team_df['Role'].str.strip().str.lower() == 'batter'].index
    if len(batter_indices) > 0:
        prob += pulp.lpSum([x[i] for i in batter_indices]) >= 1, "Min_Batters"
    
    bowler_indices = team_df[team_df['Role'].str.strip().str.lower() == 'bowler'].index
    if len(bowler_indices) > 0:
        prob += pulp.lpSum([x[i] for i in bowler_indices]) >= 1, "Min_Bowlers"
    
    # Ensure at least one player is selected from each team.
    team1_indices = team_df[team_df['Team'] == team1].index
    team2_indices = team_df[team_df['Team'] == team2].index
    prob += pulp.lpSum([x[i] for i in team1_indices]) >= 1, f"Min_{team1}"
    prob += pulp.lpSum([x[i] for i in team2_indices]) >= 1, f"Min_{team2}"
    
    # Solve the optimization problem.
    prob.solve()
    
    if pulp.LpStatus[prob.status] != 'Optimal':
        print("No optimal solution found!")
        return None

    # Extract the indices of the selected players.
    selected = [i for i in players if pulp.value(x[i]) == 1]
    selected_df = team_df.loc[selected].copy()
    
    # Sort the selected players by their score (highest first).
    selected_df.sort_values('Score', ascending=False, inplace=True)
    
    # Assign special roles (Captain and Vice Captain) to the top players.
    selected_df['Role_In_Team'] = 'Player'
    if len(selected_df) > 0:
        selected_df.iloc[0, selected_df.columns.get_loc('Role_In_Team')] = 'Captain'
    if len(selected_df) > 1:
        selected_df.iloc[1, selected_df.columns.get_loc('Role_In_Team')] = 'Vice Captain'
    
    return selected_df[['Player', 'Team', 'Role', 'Score', 'Role_In_Team']]

# --------------------------------------------------
# 5. Example Usage
# --------------------------------------------------
if __name__ == '__main__':
    # For demonstration with the sample snippet, choose teams that yield enough players.
    # In this sample, we use "South Africa" and "India" and select 3 players.
    team1 = "Australia"
    team2 = "India"
    optimal_team = optimize_team(team1, team2, total_players=11)
    
    if optimal_team is not None:
        print(f"Optimal Team for {team1} vs {team2}:")
        print(optimal_team)
