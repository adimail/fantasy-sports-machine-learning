import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import pulp


class FantasyTeamOptimizer:
    def __init__(self):
        self.model = XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            verbosity=2,
        )
        self.evaluation_df = None
        self.roster_df = None
        self.merged_df = None

    def load_data(self):
        """Load evaluation and roster data from CSV files."""
        print("Loading data...")
        self.evaluation_df = pd.read_csv("output/recent_player_form.csv")
        self.roster_df = pd.read_csv("Downloads/SquadPlayerNames.csv")
        print("Data loaded successfully.")

    def filter_and_merge(self):
        """
        Filter the roster to only include players marked as PLAYING,
        map the player types to match the evaluation file,
        and merge with evaluation data using player name and type.
        """
        # Filter only PLAYING players from roster (using case-insensitive match)
        roster_filtered = self.roster_df[
            self.roster_df["IsPlaying"].str.upper() == "PLAYING"
        ].copy()

        # Rename "Player Name" to "Player" for merging purposes
        roster_filtered.rename(columns={"Player Name": "Player"}, inplace=True)

        # Use only the required columns
        roster_filtered = roster_filtered[
            ["Player Type", "Player", "IsPlaying", "Team"]
        ]

        # Map roster player types to match evaluation file values.
        # For example, convert "All Rounder" to "ALL", "Batsmen" to "BAT",
        # "Bowlers" to "BOWL", and treat "Wicket Keeper" as a batsman ("BAT").
        type_mapping = {
            "All Rounder": "ALL",
            "Batsmen": "BAT",
            "Bowlers": "BOWL",
            "Wicket Keeper": "BAT",
        }
        roster_filtered["Player Type"] = roster_filtered["Player Type"].map(
            type_mapping
        )

        # Merge on "Player" and "Player Type"
        self.merged_df = pd.merge(
            roster_filtered,
            self.evaluation_df,
            on=["Player", "Player Type"],
            how="inner",
        )
        print(f"Merged data has {self.merged_df.shape[0]} players.")
        if self.merged_df.empty:
            print(
                "Warning: Merged DataFrame is empty. Please check that the player names and types match between the CSV files."
            )

    def compute_target_and_role(self):
        """
        For each player, compute the predicted fantasy points and assign a role for constraint purposes.
        - For players with type "BOWL": use Bowling Form and assign role "BOWL".
        - For players with type "ALL": use the maximum among Batting Form, Bowling Form, and Fielding Form.
          * If Batting Form is highest, assign role "BAT".
          * If Bowling Form is highest, assign role "BOWL".
          * If Fielding Form is highest, assign role "BAT" (defaulting to batsman).
        - For players with type "BAT": use Batting Form and assign role "BAT".
        """
        if self.merged_df.empty:
            print("Merged DataFrame is empty. Skipping compute_target_and_role.")
            return

        def compute(row):
            ptype = row["Player Type"].upper()
            if ptype == "BOWL":
                return row["Bowling Form"], "BOWL"
            elif ptype == "ALL":
                forms = {
                    "BAT": row["Batting Form"],
                    "BOWL": row["Bowling Form"],
                    "FIELD": row["Fielding Form"],
                }
                best_category = max(forms, key=forms.get)
                assigned_role = best_category if best_category != "FIELD" else "BAT"
                return forms[best_category], assigned_role
            elif ptype == "BAT":
                return row["Batting Form"], "BAT"
            else:
                return row["Batting Form"], "BAT"

        # The apply will return a DataFrame with two columns; assign them to "Predicted" and "AssignedRole"
        self.merged_df[["Predicted", "AssignedRole"]] = self.merged_df.apply(
            lambda row: pd.Series(compute(row)), axis=1
        )
        print("Computed predicted scores and assigned roles.")

    def train_model(self):
        """
        Train an XGBoost model to predict fantasy points using the three form scores as features.
        Although the target is directly computed from the form scores, this step is included to use
        the provided parameters and simulate the ML training process.
        """
        if self.merged_df.empty:
            print("Merged DataFrame is empty. Skipping training.")
            return

        print("Training model...")
        X = self.merged_df[["Batting Form", "Bowling Form", "Fielding Form"]]
        y = self.merged_df["Predicted"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        print(
            f"Model trained. Train score: {train_score:.2f}, Test score: {test_score:.2f}"
        )

        self.merged_df["Predicted"] = self.model.predict(X)

    def optimize_team(self):
        """
        Optimize team selection using PuLP.
        The objective is to maximize the total predicted fantasy points, applying multipliers for
        captain (full extra points) and vice-captain (0.5 extra points).
        Constraints:
          - Exactly 11 players selected.
          - At least 3 players assigned as bowlers.
          - At least 3 players assigned as batsmen.
          - Exactly one captain and one vice-captain.
        """
        if self.merged_df.empty:
            print("Merged DataFrame is empty. Cannot optimize team.")
            return []

        print("Starting team optimization...")
        problem = pulp.LpProblem("OptimalTeamSelection", pulp.LpMaximize)
        players = list(self.merged_df["Player"])
        selection = pulp.LpVariable.dicts("Select", players, cat="Binary")
        captain = pulp.LpVariable.dicts("Captain", players, cat="Binary")
        vice_captain = pulp.LpVariable.dicts("ViceCaptain", players, cat="Binary")

        # Objective function: Each player's predicted score is multiplied by:
        # 1 (if just selected) + 1 (if captain) + 0.5 (if vice-captain)
        objective_terms = []
        for idx, row in self.merged_df.iterrows():
            p = row["Player"]
            score = row["Predicted"]
            term = score * (selection[p] + captain[p] + 0.5 * vice_captain[p])
            objective_terms.append(term)
        problem += pulp.lpSum(objective_terms)

        # Constraint: Exactly 11 players must be selected.
        problem += pulp.lpSum([selection[p] for p in players]) == 11, "TotalPlayers"
        # Constraint: Exactly one captain.
        problem += pulp.lpSum([captain[p] for p in players]) == 1, "OneCaptain"
        # Constraint: Exactly one vice-captain.
        problem += pulp.lpSum([vice_captain[p] for p in players]) == 1, "OneViceCaptain"

        # Ensure that captain and vice-captain are selected.
        for p in players:
            problem += captain[p] <= selection[p], f"CaptainSelected_{p}"
            problem += vice_captain[p] <= selection[p], f"ViceCaptainSelected_{p}"
            problem += captain[p] + vice_captain[p] <= selection[p], f"NoDualRole_{p}"

        # Role constraints:
        # At least 3 bowlers (AssignedRole == "BOWL")
        bowlers = self.merged_df[self.merged_df["AssignedRole"] == "BOWL"][
            "Player"
        ].tolist()
        problem += pulp.lpSum([selection[p] for p in bowlers]) >= 3, "MinBowlers"
        # At least 3 batsmen (AssignedRole == "BAT")
        batsmen = self.merged_df[self.merged_df["AssignedRole"] == "BAT"][
            "Player"
        ].tolist()
        problem += pulp.lpSum([selection[p] for p in batsmen]) >= 3, "MinBatsmen"

        print("Solving optimization problem...")
        problem.solve(pulp.PULP_CBC_CMD(msg=True))
        print("Optimization status:", pulp.LpStatus[problem.status])

        self.merged_df = self.merged_df.sort_values("Predicted", ascending=False)

        team = []
        for idx, row in self.merged_df.iterrows():
            p = row["Player"]
            if pulp.value(selection[p]) == 1:
                role = row["AssignedRole"]
                marker = ""
                if pulp.value(captain[p]) == 1:
                    marker = "(Captain)"
                elif pulp.value(vice_captain[p]) == 1:
                    marker = "(Vice Captain)"
                team.append((p, role, marker, row["Predicted"]))
        return team


def BuildTeam():
    optimizer = FantasyTeamOptimizer()
    optimizer.load_data()
    optimizer.filter_and_merge()
    optimizer.compute_target_and_role()
    optimizer.train_model()
    team = optimizer.optimize_team()

    print("\n\nSelected Team\n")
    for player, role, marker, score in team:
        print(f"{score:.2f} \t {role} \t {player} {marker}")


if __name__ == "__main__":
    BuildTeam()
