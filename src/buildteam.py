import pandas as pd
import pulp
import sys
import yaml


class FantasyTeamOptimizer:
    def __init__(self):
        try:
            with open("config.yaml", "r") as stream:
                config = yaml.safe_load(stream)
        except Exception as e:
            print(f"Error reading YAML config file: {e}")
            sys.exit(1)

        self.config = config
        self.evaluation_df = None
        self.roster_df = None
        self.merged_df = None

    def load_data(self):
        """Load evaluation and roster data from CSV files."""
        print("Loading data...")
        self.evaluation_df = pd.read_csv(self.config["data"]["player_form"])
        self.roster_df = pd.read_csv(self.config["data"]["squad_input"])
        print("Data loaded successfully.")

    def filter_and_merge(self):
        """
        Filter the roster to only include players marked as PLAYING,
        map the player types to match the evaluation file,
        and merge with evaluation data using player name and type.
        """
        type_mapping = {
            "All Rounder": "ALL",
            "Batsmen": "BAT",
            "Bowlers": "BOWL",
            "Wicket Keeper": "Keeper",
        }

        roster_filtered = self.roster_df[
            self.roster_df["IsPlaying"].str.upper() == "PLAYING"
        ].copy()
        roster_filtered.rename(columns={"Player Name": "Player"}, inplace=True)
        roster_filtered["Player Type"] = roster_filtered["Player Type"].map(
            type_mapping
        )
        roster_filtered = roster_filtered[
            ["Player Type", "Player", "IsPlaying", "Team"]
        ]

        self.merged_df = pd.merge(
            roster_filtered,
            self.evaluation_df,
            on=["Player", "Player Type"],
            how="inner",
            suffixes=("", "_eval"),
        )

        if "Team_eval" in self.merged_df.columns:
            self.merged_df.drop(columns=["Team_eval"], inplace=True)

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

        self.merged_df[["Predicted", "AssignedRole"]] = self.merged_df.apply(
            lambda row: pd.Series(compute(row)), axis=1
        )
        print("Computed predicted scores and assigned roles.")

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
            t = row["Team"]
            if pulp.value(selection[p]) == 1:
                role = row["AssignedRole"]
                marker = ""
                if pulp.value(captain[p]) == 1:
                    marker = "(Captain)"
                elif pulp.value(vice_captain[p]) == 1:
                    marker = "(Vice Captain)"
                team.append((p, role, marker, row["Predicted"], t))
        return team


def BuildTeam():
    optimizer = FantasyTeamOptimizer()
    optimizer.load_data()
    optimizer.filter_and_merge()
    optimizer.compute_target_and_role()
    team = optimizer.optimize_team()

    print("\n\nSelected Team\n")
    for player, role, marker, score, t in team:
        print(f"{score:.2f} \t {role} \t {t} \t {player} {marker}")


if __name__ == "__main__":
    BuildTeam()
