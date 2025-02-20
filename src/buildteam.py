import pandas as pd
import pulp
import sys
import yaml


class FantasyTeamOptimizer:
    def __init__(self):
        """Initialize configuration and placeholders for dataframes."""
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
        self.evaluation_df = pd.read_csv(self.config["data"]["player_form"])
        self.roster_df = pd.read_csv(self.config["data"]["squad_input"])

    def filter_and_merge(self):
        """
        Filter the roster to only include players marked as PLAYING,
        and merge with evaluation data using player name and type.
        """
        roster_filtered = self.roster_df[
            self.roster_df["IsPlaying"].str.upper() == "PLAYING"
        ].copy()
        roster_filtered.rename(columns={"Player Name": "Player"}, inplace=True)
        roster_filtered = roster_filtered[["Player Type", "Player", "Team"]]

        self.merged_df = pd.merge(
            roster_filtered,
            self.evaluation_df,
            on=["Player", "Player Type"],
            how="inner",
            suffixes=("", "_eval"),
        )
        if self.merged_df.empty:
            print(
                "Warning: Merged DataFrame is empty. Please verify that the player names and types match between the input files."
            )

        # Standardize role names to match optimization constraints.
        # For example, mapping: "ALL" -> "All Rounder", "BOWL" -> "Bowler", "BAT" -> "Batsmen", "WK" -> "Wicket Keeper"
        role_mapping = {
            "ALL": "All Rounder",
            "BOWL": "Bowler",
            "BAT": "Batsmen",
            "WK": "Wicket Keeper",
        }
        self.merged_df["Player Type"] = self.merged_df["Player Type"].replace(
            role_mapping
        )
        self.merged_df.drop(
            columns=["Team_eval", "Credits"], inplace=True, errors="ignore"
        )
        self.merged_df = self.merged_df.rename(columns={"Player Type": "Role"})
        print("\nRole counts after normalization:")
        print(self.merged_df["Role"].value_counts())

    def calculate_score(self, row):
        """
        Calculate a player's score based on their role:
          - Batsmen: Score = batter_weight * Batting Form
          - Bowler: Score = bowler_weight * Bowling Form
          - All Rounder: Score = allrounder_weight * ((Batting Form + Bowling Form) / 2)
          - Wicket Keeper: Score = keeper_weight * Batting Form
        """
        role = row["Role"].strip()
        if role == "Batsmen":
            return self.config["algorithm"]["batter_weight"] * row["Batting Form"]
        elif role == "Bowler":
            return self.config["algorithm"]["bowler_weight"] * row["Bowling Form"]
        elif role == "All Rounder":
            return self.config["algorithm"]["allrounder_weight"] * (
                (row["Batting Form"] + row["Bowling Form"]) / 2
            )
        elif role == "Wicket Keeper":
            return self.config["algorithm"]["keeper_weight"] * row["Batting Form"]
        else:
            return 0

    def compute_target_and_role(self):
        """Compute the role-based score for each player."""
        self.merged_df["Score"] = self.merged_df.apply(self.calculate_score, axis=1)

    def optimize_team(self):
        """
        Optimize fantasy team selection from the merged data.

        Constraints:
          - Exactly total_players (default 11) are selected.
          - At least 4 Batsmen.
          - At least 5 players with bowling contributions (Bowler or All Rounder).
          - At least 3 Bowler.
          - At least 1 Wicket Keeper.
          - At least 2 All Rounder.

        Returns:
            A DataFrame of selected players with their computed Score and assigned team role (e.g., Captain, Vice Captain).
        """
        team_df = self.merged_df.copy()

        prob = pulp.LpProblem("FantasyTeam", pulp.LpMaximize)
        players = team_df.index.tolist()
        x = pulp.LpVariable.dicts("player", players, cat="Binary")
        prob += (
            pulp.lpSum([x[i] * team_df.loc[i, "Score"] for i in players]),
            "Total_Score",
        )
        prob += pulp.lpSum([x[i] for i in players]) == 11, "Total_Players"

        batter_indices = team_df[team_df["Role"] == "Batsmen"].index
        prob += pulp.lpSum([x[i] for i in batter_indices]) >= 4, "Min_Batsmen"

        bowling_indices = team_df[team_df["Role"].isin(["Bowler", "All Rounder"])].index
        prob += (
            pulp.lpSum([x[i] for i in bowling_indices]) >= 5,
            "Min_Bowling_Contributors",
        )

        pure_bowler_indices = team_df[team_df["Role"] == "Bowler"].index
        prob += pulp.lpSum([x[i] for i in pure_bowler_indices]) >= 3, "Min_Bowlers"

        keeper_indices = team_df[team_df["Role"] == "Wicket Keeper"].index
        prob += pulp.lpSum([x[i] for i in keeper_indices]) >= 1, "Min_WicketKeepers"

        allrounder_indices = team_df[team_df["Role"] == "All Rounder"].index
        prob += pulp.lpSum([x[i] for i in allrounder_indices]) >= 2, "Min_AllRounders"

        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        print(f"\n\nLP Status: {pulp.LpStatus[prob.status]}")
        if pulp.LpStatus[prob.status] != "Optimal":
            print("No optimal solution found!")
            return None

        selected = [i for i in players if pulp.value(x[i]) == 1]
        team = team_df.loc[selected].copy()
        team.sort_values("Score", ascending=False, inplace=True)
        team["Position"] = "Player"
        if len(team) > 0:
            team.iloc[0, team.columns.get_loc("Position")] = "Captain"
        if len(team) > 1:
            team.iloc[1, team.columns.get_loc("Position")] = "Vice Captain"

        team.set_index("Player", inplace=True)
        return team


def BuildTeam():
    """Build the fantasy team by running the optimizer steps sequentially."""
    optimizer = FantasyTeamOptimizer()
    optimizer.load_data()
    optimizer.filter_and_merge()
    optimizer.compute_target_and_role()
    team = optimizer.optimize_team()
    print("\nSelected Team\n")
    print(team)


if __name__ == "__main__":
    BuildTeam()
