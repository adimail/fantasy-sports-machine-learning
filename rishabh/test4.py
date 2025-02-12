import pandas as pd
import numpy as np
import pulp

# -----------------------------------------
# 1. Data Loading and Cleaning
# -----------------------------------------
df = pd.read_csv('players_with_roles.csv')

# -----------------------------------------
# 2. Compute Per-Match Averages
# -----------------------------------------
# Calculate average runs and average wickets per match.
df['Avg_Runs'] = df['Runs_x'] / df['Mat_x']
df['Avg_Wkts'] = df['Wkts'] / df['Mat_x']

# -----------------------------------------
# 3. Define the Role-Specific Scoring Function
# -----------------------------------------
# Tunable parameters: adjust these to balance scoring across roles.
batter_weight = 1.0
bowler_weight = 1.2     # Increased weight for bowlers
allrounder_weight = 0.65

def calculate_score(row):
    """
    Calculates the player's score based on their role using per-match averages.
    
    For Batters: Score = Avg_Runs * (SR_x / 100)
    For Bowlers: Score = bowler_weight * (Avg_Wkts * (100 / (Econ + epsilon)))
    For All-rounders: Combination of batting and bowling scores.
    """
    role = row['Role'].strip().lower()
    
    if role == 'batter':
        # Batter: Use average runs and strike rate.
        return batter_weight * (row['Avg_Runs'] * (row['SR_x'] / 100))
    elif role == 'bowler':
        # Bowler: Use average wickets and economy scaled by 100.
        return bowler_weight * (row['Avg_Wkts'] * (100 / (row['Econ'] + 0.01)))
    elif role == 'all-rounder':
        # All-rounder: Sum of batting and bowling contributions.
        batting_part = batter_weight * (row['Avg_Runs'] * (row['SR_x'] / 100))
        bowling_part = bowler_weight * (row['Avg_Wkts'] * (100 / (row['Econ'] + 0.01)))
        return allrounder_weight * (batting_part + bowling_part)
    else:
        return 0

df['Score'] = df.apply(calculate_score, axis=1)

# -----------------------------------------
# 4. Optimization: Selecting the Fantasy Team
# -----------------------------------------
def optimize_team(team1, team2, total_players=11):
    """
    Optimize the fantasy team selection from players belonging to team1 and team2
    using these constraints:
      - Exactly total_players are selected.
      - At least 4 batters.
      - At least 5 bowling contributors (bowlers and all-rounders).
      - At least 3 pure bowlers.
      - At least one player from each team.
      
    Returns a DataFrame of selected players with assigned roles (Captain, Vice Captain, etc.).
    """
    # Filter players from the selected teams.
    team_df = df[df['Team'].isin([team1, team2])].copy()
    
    # Create the LP problem (maximize total team score)
    prob = pulp.LpProblem("FantasyTeam", pulp.LpMaximize)
    
    # Create binary decision variables for each player.
    players = team_df.index.tolist()
    x = pulp.LpVariable.dicts("player", players, cat='Binary')
    
    # Objective: maximize total score of selected players.
    prob += pulp.lpSum([x[i] * team_df.loc[i, 'Score'] for i in players]), "Total_Score"
    
    # Constraint: exactly total_players are selected.
    prob += pulp.lpSum([x[i] for i in players]) == total_players, "Total_Players"
    
    # Constraint: At least 4 batters.
    batters = team_df[team_df['Role'].str.strip().str.lower() == 'batter'].index
    if len(batters) > 0:
        prob += pulp.lpSum([x[i] for i in batters]) >= 4, "Min_Batters"
    
    # Constraint: At least 5 bowling contributors (bowlers + all-rounders).
    bowlers_all = team_df[team_df['Role'].str.strip().str.lower().isin(['bowler', 'all-rounder'])].index
    if len(bowlers_all) > 0:
        prob += pulp.lpSum([x[i] for i in bowlers_all]) >= 5, "Min_Bowlers_Allrounders"
    
    # Constraint: At least 3 pure bowlers.
    pure_bowlers = team_df[team_df['Role'].str.strip().str.lower() == 'bowler'].index
    if len(pure_bowlers) > 0:
        prob += pulp.lpSum([x[i] for i in pure_bowlers]) >= 3, "Min_Bowlers"
    
    # Constraint: At least one player from each team.
    team1_indices = team_df[team_df['Team'] == team1].index
    team2_indices = team_df[team_df['Team'] == team2].index
    if len(team1_indices) > 0:
        prob += pulp.lpSum([x[i] for i in team1_indices]) >= 1, f"Min_from_{team1}"
    if len(team2_indices) > 0:
        prob += pulp.lpSum([x[i] for i in team2_indices]) >= 1, f"Min_from_{team2}"
    
    # Solve the optimization problem.
    prob.solve()
    
    if pulp.LpStatus[prob.status] != 'Optimal':
        print("No optimal solution found!")
        return None
    
    # Retrieve selected players.
    selected = [i for i in players if pulp.value(x[i]) == 1]
    selected_df = team_df.loc[selected].copy()
    selected_df.sort_values('Score', ascending=False, inplace=True)
    
    # Assign Captain and Vice Captain to the top two scoring players.
    selected_df['Role_In_Team'] = 'Player'
    if len(selected_df) >= 1:
        selected_df.iloc[0, selected_df.columns.get_loc('Role_In_Team')] = 'Captain'
    if len(selected_df) >= 2:
        selected_df.iloc[1, selected_df.columns.get_loc('Role_In_Team')] = 'Vice Captain'
    
    return selected_df[['Player', 'Team', 'Role', 'Score', 'Role_In_Team']]

# -----------------------------------------
# 5. Example Usage
# -----------------------------------------
if __name__ == '__main__':
    # Set the teams (modify these as needed)
    team1 = "India"
    team2 = "Australia"
    
    # Optimize team selection. For real contests, total_players should be 11.
    best_team = optimize_team(team1, team2, total_players=15)
    
    if best_team is not None:
        print(f"Optimal Team for {team1} vs {team2}:")
        print(best_team)
