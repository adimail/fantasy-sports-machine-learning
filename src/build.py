import pandas as pd
import pulp

# ============================================================
# Adjustable Weight Parameters (tweak these as needed)
# ============================================================
batter_weight     = 1.0
bowler_weight     = 1.0
allrounder_weight = 1.15  # A slight boost for all rounders relative to the average
keeper_weight     = 1.0   # Weight for keepers based on batting form


class FantasyTeamOptimizer:
    def __init__(self):
        self.evaluation_df = None
        self.roster_df = None
        self.merged_df = None

    def load_data(self):
        """Load evaluation and roster data from CSV files."""
        print("Loading data...")
        self.evaluation_df = pd.read_csv("data/recent_player_form.csv")
        self.roster_df = pd.read_csv("Downloads/SquadPlayerNames.csv")
        print("Data loaded successfully.")

    def filter_and_merge(self):
        """
        Filter the roster to only include players marked as PLAYING,
        merge with evaluation data using player name,
        drop unnecessary columns, and rename columns as needed.
        """
        # Map the roster's "Player Name" to a new column "Player"
        self.roster_df['Player'] = self.roster_df['Player Name']
        
        # Merge on "Player"
        merging_df = pd.merge(self.evaluation_df, self.roster_df, on="Player", how="inner")
        
        # Drop unnecessary columns (adjust as needed based on your file structure)
        dropping_cols = ['Fielding Form', 'Credits_x', 'Player Type_x', 'Credits_y', 
                         'Player Name', 'Team_y', 'lineupOrder']
        merging_df.drop(columns=[col for col in dropping_cols if col in merging_df.columns], 
                        inplace=True)
        
        # Rename columns to ensure required names exist
        merging_df.rename(columns={"Team_x": "Team", "Player Type_y": "Role"}, inplace=True)
        
        # Filter: Only include players whose IsPlaying status equals "PLAYING" (case-insensitive)
        merging_df = merging_df[merging_df["IsPlaying"].str.strip().str.lower() == "playing"]
        
        self.merged_df = merging_df
        print(f"Merged data has {self.merged_df.shape[0]} players.")

    # ============================================================
    # 1. Compute a Role-Based Score for Each Player
    # ============================================================
    @staticmethod
    def calculate_score(row):
        """
        Calculate a player's score based on their role:
          - Batsmen: Score = batter_weight * Batting Form
          - Bowlers: Score = bowler_weight * Bowling Form
          - All Rounder: Score = allrounder_weight * ((Batting Form + Bowling Form) / 2)
          - Wicket Keeper: Score = keeper_weight * Batting Form
        """
        role = row["Role"].strip()
        if role == "Batsmen":
            return batter_weight * row["Batting Form"]
        elif role == "Bowlers":
            return bowler_weight * row["Bowling Form"]
        elif role == "All Rounder":
            return allrounder_weight * ((row["Batting Form"] + row["Bowling Form"]) / 2)
        elif role == "Wicket Keeper":
            return keeper_weight * row["Batting Form"]
        else:
            return 0

    def compute_target_and_role(self):
        """Compute the role-based score for each player."""
        self.merged_df["Score"] = self.merged_df.apply(FantasyTeamOptimizer.calculate_score, axis=1)

    # ============================================================
    # 2. Define the Optimization Function for the Best 11 Players
    # ============================================================
    def optimize_team(self, total_players=11):
        """
        Optimize fantasy team selection from the merged data.
        
        Constraints:
          - Exactly total_players (default 11) are selected.
          - At least 4 Batsmen.
          - At least 5 players with bowling contributions (Bowlers or All Rounder).
          - At least 3 Bowlers.
          - At least 1 Wicket Keeper.
          - At least 2 All Rounder.
        
        Returns:
            A DataFrame of selected players with their computed Score and assigned team role (e.g., Captain, Vice Captain).
        """
        team_df = self.merged_df.copy()
        
        # Create the LP problem to maximize the total team score.
        prob = pulp.LpProblem("FantasyTeam", pulp.LpMaximize)
        
        # Create binary decision variables for each player.
        players = team_df.index.tolist()
        x = pulp.LpVariable.dicts("player", players, cat="Binary")
        
        # Objective: maximize the sum of scores of selected players.
        prob += pulp.lpSum([x[i] * team_df.loc[i, "Score"] for i in players]), "Total_Score"
        
        # Constraint: exactly total_players are selected.
        prob += pulp.lpSum([x[i] for i in players]) == total_players, "Total_Players"
        
        # Constraint: at least 4 Batsmen.
        batter_indices = team_df[team_df["Role"] == "Batsmen"].index
        prob += pulp.lpSum([x[i] for i in batter_indices]) >= 4, "Min_Batsmen"
        
        # Constraint: at least 5 players with bowling contributions (Bowlers or All Rounder).
        bowling_indices = team_df[team_df["Role"].isin(["Bowlers", "All Rounder"])].index
        prob += pulp.lpSum([x[i] for i in bowling_indices]) >= 5, "Min_Bowling_Contributors"
        
        # Constraint: at least 3 Bowlers.
        pure_bowler_indices = team_df[team_df["Role"] == "Bowlers"].index
        prob += pulp.lpSum([x[i] for i in pure_bowler_indices]) >= 3, "Min_Bowlers"
        
        # Constraint: at least 1 Wicket Keeper.
        keeper_indices = team_df[team_df["Role"] == "Wicket Keeper"].index
        prob += pulp.lpSum([x[i] for i in keeper_indices]) >= 1, "Min_WicketKeepers"
        
        # Constraint: at least 2 All Rounder.
        allrounder_indices = team_df[team_df["Role"] == "All Rounder"].index
        prob += pulp.lpSum([x[i] for i in allrounder_indices]) >= 2, "Min_AllRounders"
        
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
        
        return selected_df


def BuildTeam():
    optimizer = FantasyTeamOptimizer()
    optimizer.load_data()
    optimizer.filter_and_merge()
    optimizer.compute_target_and_role()
    team = optimizer.optimize_team()

    print("\n\nSelected Team\n")
    if team is not None:
        print(team[['Player','Team','Role_In_Team']])

if __name__ == "__main__":
    BuildTeam()
