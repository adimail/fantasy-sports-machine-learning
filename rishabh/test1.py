import pandas as pd
import pulp

# Load and preprocess data
df = pd.read_csv('players_with_roles.csv')

# Clean data
df['Econ'].fillna(0, inplace=True)
df['SR_x'].fillna(0, inplace=True)
df['Wkts'].fillna(0, inplace=True)
df['Runs_x'] = df['Runs_x'].astype(float)
df['SR_x'] = df['SR_x'].astype(float)

# Calculate player scores
def calculate_score(row):
    if row['Role'] == 'Batter':
        return row['Runs_x'] * (row['SR_x'] / 100)
    elif row['Role'] == 'Bowler':
        return row['Wkts'] * (1 / (row['Econ'] + 1e-6))  # Avoid division by zero
    elif row['Role'] == 'All-rounder':
        return (row['Runs_x'] * (row['SR_x'] / 100) + (row['Wkts'] * (1 / (row['Econ'] + 1e-6))))
    return 0

df['Score'] = df.apply(calculate_score, axis=1)

def optimize_team(team1, team2):
    # Filter players from selected teams
    team_df = df[df['Team'].isin([team1, team2])].copy()
    
    # Create LP problem
    prob = pulp.LpProblem("FantasyTeam", pulp.LpMaximize)
    
    # Decision variables
    players = team_df.index.tolist()
    x = pulp.LpVariable.dicts("player", players, cat='Binary')
    
    # Objective function
    prob += pulp.lpSum([x[i] * team_df.loc[i, 'Score'] for i in players])
    
    # Constraints
    prob += pulp.lpSum([x[i] for i in players]) == 11  # Exactly 11 players
    
    # Minimum batters
    batters = team_df[team_df['Role'] == 'Batter'].index
    prob += pulp.lpSum([x[i] for i in batters]) >= 4
    
    # Minimum bowling contributors (bowlers + all-rounders)
    bowlers_all = team_df[team_df['Role'].isin(['Bowler', 'All-rounder'])].index
    prob += pulp.lpSum([x[i] for i in bowlers_all]) >= 5
    
    # Minimum pure bowlers
    pure_bowlers = team_df[team_df['Role'] == 'Bowler'].index
    prob += pulp.lpSum([x[i] for i in pure_bowlers]) >= 3
    
    # At least 1 player from each team
    prob += pulp.lpSum([x[i] for i in team_df[team_df['Team'] == team1].index]) >= 1
    prob += pulp.lpSum([x[i] for i in team_df[team_df['Team'] == team2].index]) >= 1
    
    # Solve the problem
    prob.solve()
    
    # Get selected players
    selected = [i for i in players if x[i].value() == 1]
    selected_df = team_df.loc[selected].sort_values('Score', ascending=False)
    
    # Assign captain and vice-captain
    selected_df['Role_In_Team'] = 'Player'
    selected_df.iloc[0, selected_df.columns.get_loc('Role_In_Team')] = 'Captain'
    selected_df.iloc[1, selected_df.columns.get_loc('Role_In_Team')] = 'Vice Captain'
    
    return selected_df[['Player', 'Team', 'Role', 'Score', 'Role_In_Team']]

# Example usage
team1 = "India"
team2 = "Pakistan"
best_team = optimize_team(team1, team2)
print(f"Optimal Team for {team1} vs {team2}:")
print(best_team)