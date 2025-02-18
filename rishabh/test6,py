import pandas as pd
import pulp

# ============================================================
# Adjustable Weight Parameters (tweak these as needed)
# ============================================================
batter_weight     = 1.0
bowler_weight     = 1.0
allrounder_weight = 1.15  # A slight boost for all-rounders relative to the average
keeper_weight     = 1.0  # Weight for keepers based on batting form

# ============================================================
# 1. Load the CSV Data
# ============================================================
# The CSV file should have these columns:
#   Player, Team, Batting Form, Bowling Form, Role
df = pd.read_csv('good_data(keep).csv')

# ============================================================
# 2. Compute a Role-Based Score for Each Player
# ============================================================
def calculate_score(row):
    """
    Calculate a player's score based on their role:
      - Batter: Score = batter_weight * Batting Form
      - Bowler: Score = bowler_weight * Bowling Form
      - All-rounder: Score = allrounder_weight * (Average of Batting Form and Bowling Form)
      - Keeper: Score = keeper_weight * Batting Form  (deciding keeper solely on batting)
    """
    role = row["Role"].strip().lower()
    if role == "batter":
        return batter_weight * row["Batting Form"]
    elif role == "bowler":
        return bowler_weight * row["Bowling Form"]
    elif role == "all-rounder":
        return allrounder_weight * ((row["Batting Form"] + row["Bowling Form"]) / 2)
    elif role == "keeper":
        return keeper_weight * row["Batting Form"]
    else:
        return 0

df["Score"] = df.apply(calculate_score, axis=1)

# ============================================================
# 3. Define the Optimization Function
# ============================================================
def optimize_team(team1, team2, total_players=11):
    """
    Optimize fantasy team selection from players in team1 and team2.
    
    Constraints:
      - Exactly total_players are selected.
      - At least 4 batters.
      - At least 5 players with bowling contributions (Bowler or All-rounder).
      - At least 3 pure bowlers.
      - At least 1 keeper.
      - At least 1 player from each of the two selected teams.
    
    Returns:
        A DataFrame of selected players with their computed Score and assigned team role (e.g., Captain, Vice Captain).
    """
    # Filter to include only players from team1 and team2.
    team_df = df[df["Team"].isin([team1, team2])].copy()
    
    # Create the LP problem to maximize total team score.
    prob = pulp.LpProblem("FantasyTeam", pulp.LpMaximize)
    
    # Create binary decision variables for each player.
    players = team_df.index.tolist()
    x = pulp.LpVariable.dicts("player", players, cat="Binary")
    
    # Objective: maximize the sum of scores of selected players.
    prob += pulp.lpSum([x[i] * team_df.loc[i, "Score"] for i in players]), "Total_Score"
    
    # Constraint: exactly total_players are selected.
    prob += pulp.lpSum([x[i] for i in players]) == total_players, "Total_Players"
    
    # Constraint: at least 4 batters.
    batter_indices = team_df[team_df["Role"].str.strip().str.lower() == "batter"].index
    prob += pulp.lpSum([x[i] for i in batter_indices]) >= 4, "Min_Batters"
    
    # Constraint: at least 5 players with bowling contributions (Bowler or All-rounder).
    bowling_indices = team_df[team_df["Role"].str.strip().str.lower().isin(["bowler", "all-rounder"])].index
    prob += pulp.lpSum([x[i] for i in bowling_indices]) >= 5, "Min_Bowling_Contributors"
    
    # Constraint: at least 3 pure bowlers.
    pure_bowler_indices = team_df[team_df["Role"].str.strip().str.lower() == "bowler"].index
    prob += pulp.lpSum([x[i] for i in pure_bowler_indices]) >= 3, "Min_Bowlers"
    
    # Constraint: at least 1 keeper.
    keeper_indices = team_df[team_df["Role"].str.strip().str.lower() == "keeper"].index
    prob += pulp.lpSum([x[i] for i in keeper_indices]) >= 1, "Min_Keepers"
    
    # Constraint: at least one player from each team.
    team1_indices = team_df[team_df["Team"] == team1].index
    team2_indices = team_df[team_df["Team"] == team2].index
    prob += pulp.lpSum([x[i] for i in team1_indices]) >= 1, f"Min_from_{team1}"
    prob += pulp.lpSum([x[i] for i in team2_indices]) >= 1, f"Min_from_{team2}"
    
    # Solve the optimization problem.
    prob.solve()
    
    if pulp.LpStatus[prob.status] != "Optimal":
        print("No optimal solution found!")
        return None
    
    # Retrieve selected players.
    selected = [i for i in players if pulp.value(x[i]) == 1]
    selected_df = team_df.loc[selected].copy()
    selected_df.sort_values("Score", ascending=False, inplace=True)
    
    # Assign team roles: highest scorer -> Captain, second highest -> Vice Captain.
    selected_df["Role_In_Team"] = "Player"
    if len(selected_df) > 0:
        selected_df.iloc[0, selected_df.columns.get_loc("Role_In_Team")] = "Captain"
    if len(selected_df) > 1:
        selected_df.iloc[1, selected_df.columns.get_loc("Role_In_Team")] = "Vice Captain"
    
    return selected_df[["Player", "Team", "Role", "Score", "Role_In_Team"]]

# ============================================================
# 4. Example Usage
# ============================================================
if __name__ == "__main__":
    # Example teams (adjust these based on your dataset).
    team1 = "India"
    team2 = "Australia"
    
    # Optimize team selection (selecting a total of 11 players).
    best_team = optimize_team(team1, team2, total_players=11)
    
    if best_team is not None:
        print("Optimal Fantasy Team:")
        print(best_team)
