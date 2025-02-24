import pandas as pd
import pulp
import sys
import yaml
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FantasyTeamOptimizer:
    def __init__(self):
        """Initialize configuration and placeholders for dataframes."""
        try:
            with open("config.yaml", "r") as stream:
                config = yaml.safe_load(stream)
        except Exception as e:
            logger.error(f"Error reading YAML config file: {e}")
            sys.exit(1)
        self.config = config
        self.evaluation_df = None
        self.roster_df = None
        self.merged_df = None

    def load_data(self):
        """
        Load evaluation and roster data from CSV files.
        Loads the recent player form and overall performance data,
        then combines them using a weighted average (0.6 for recent form and 0.4 for overall performance).
        The resulting DataFrame is stored in self.evaluation_df.
        """
        recent_df = pd.read_csv(self.config["data"]["player_form"])
        overall_df = pd.read_csv(self.config["data"]["overall_performance"])

        self.roster_df = pd.read_csv(self.config["data"]["squad_input"])

        combined_df = pd.merge(
            recent_df,
            overall_df,
            on=["Player", "Player Type"],
            suffixes=("_recent", "_overall"),
        )

        combined_df["Batting Form"] = (
            self.config["algorithm"]["recent_player_form"]
            * combined_df["Batting Form_recent"]
            + self.config["algorithm"]["overall_performance"]
            * combined_df["Batting Form_overall"]
        )
        combined_df["Bowling Form"] = (
            self.config["algorithm"]["recent_player_form"]
            * combined_df["Bowling Form_recent"]
            + self.config["algorithm"]["overall_performance"]
            * combined_df["Bowling Form_overall"]
        )

        if (
            "Fielding Form_recent" in combined_df.columns
            and "Fielding Form_overall" in combined_df.columns
        ):
            combined_df["Fielding Form"] = (
                self.config["algorithm"]["recent_player_form"]
                * combined_df["Fielding Form_recent"]
                + self.config["algorithm"]["overall_performance"]
                * combined_df["Fielding Form_overall"]
            )

        drop_columns = [
            col
            for col in combined_df.columns
            if col.endswith("_recent") or col.endswith("_overall")
        ]
        combined_df.drop(columns=drop_columns, inplace=True)

        self.evaluation_df = combined_df

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
            logger.warning(
                "Merged DataFrame is empty. Please verify that the player names and types match between the input files."
            )

        # Standardize role names to match optimization constraints.
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
          - Exactly 3 All Rounder, with at least one being the best in batting form,
            one the best in bowling form, and one the best in average form among all-rounders.

        Returns:
            A DataFrame of selected players with their computed Score and assigned team role (e.g., Captain, Vice Captain).
        """
        team_df = self.merged_df.copy()

        print()

        role_counts = team_df["Role"].value_counts()
        logger.info("Available players by role: %s", role_counts.to_dict())
        if role_counts.get("Batsmen", 0) < 4:
            logger.warning("Fewer than 4 Batsmen available. Optimization may fail.")
        if role_counts.get("Bowler", 0) < 3:
            logger.warning("Fewer than 3 Bowlers available. Optimization may fail.")
        if role_counts.get("Wicket Keeper", 0) < 1:
            logger.warning("No Wicket Keeper available. Optimization may fail.")
        if role_counts.get("All Rounder", 0) < 3:
            logger.warning("Fewer than 3 All Rounders available. Optimization may fail.")
        if sum(role_counts.get(role, 0) for role in ["Bowler", "All Rounder"]) < 5:
            logger.warning(
                "Fewer than 5 bowling contributors available. Optimization may fail."
            )

        prob = pulp.LpProblem("FantasyTeam", pulp.LpMaximize)
        players = team_df.index.to_list()
        x = pulp.LpVariable.dicts("player", players, cat="Binary")
        c = pulp.LpVariable.dicts("captain", players, cat="Binary")
        v = pulp.LpVariable.dicts("vice_captain", players, cat="Binary")

        # Objective: Maximize total score including captain and vice-captain bonuses
        prob += (
            pulp.lpSum(
                [
                    team_df.loc[i, "Score"] * x[i]
                    + (2.0 - 1) * team_df.loc[i, "Score"] * c[i]
                    + (1.5 - 1) * team_df.loc[i, "Score"] * v[i]
                    for i in players
                ]
            ),
            "Total_Score_with_Bonuses",
        )

        # Constraints
        prob += pulp.lpSum([x[i] for i in players]) == 11, "Total_Players"

        batter_indices = team_df[team_df["Role"] == "Batsmen"].index.to_list()
        prob += pulp.lpSum([x[i] for i in batter_indices]) >= 4, "Min_Batsmen"

        bowling_indices = team_df[team_df["Role"].isin(["Bowler", "All Rounder"])].index.to_list()
        prob += pulp.lpSum([x[i] for i in bowling_indices]) >= 5, "Min_Bowling_Contributors"

        pure_bowler_indices = team_df[team_df["Role"] == "Bowler"].index.to_list()
        prob += pulp.lpSum([x[i] for i in pure_bowler_indices]) >= 3, "Min_Bowlers"

        keeper_indices = team_df[team_df["Role"] == "Wicket Keeper"].index.to_list()
        prob += pulp.lpSum([x[i] for i in keeper_indices]) >= 1, "Min_WicketKeepers"

        allrounder_indices = team_df[team_df["Role"] == "All Rounder"].index.to_list()
        prob += pulp.lpSum([x[i] for i in allrounder_indices]) == 3, "Exactly_3_AllRounders"

        # Find best all-rounders
        allrounders_df = team_df[team_df["Role"] == "All Rounder"]
        if not allrounders_df.empty:
            max_batting_form = allrounders_df["Batting Form"].max()
            batting_best_indices = allrounders_df[allrounders_df["Batting Form"] == max_batting_form].index.to_list()
            max_bowling_form = allrounders_df["Bowling Form"].max()
            bowling_best_indices = allrounders_df[allrounders_df["Bowling Form"] == max_bowling_form].index.to_list()
            allrounders_df["Average"] = (allrounders_df["Batting Form"] + allrounders_df["Bowling Form"]) / 2
            max_average = allrounders_df["Average"].max()
            average_best_indices = allrounders_df[allrounders_df["Average"] == max_average].index.to_list()
        else:
            batting_best_indices = []
            bowling_best_indices = []
            average_best_indices = []

        # Add constraints for best all-rounders
        prob += pulp.lpSum([x[i] for i in allrounder_indices if i in batting_best_indices]) >= 1, "At_least_one_best_batter"
        prob += pulp.lpSum([x[i] for i in allrounder_indices if i in bowling_best_indices]) >= 1, "At_least_one_best_bowler"
        prob += pulp.lpSum([x[i] for i in allrounder_indices if i in average_best_indices]) >= 1, "At_least_one_best_average"

        # Captain and Vice Captain constraints
        prob += pulp.lpSum([c[i] for i in players]) == 1, "One_Captain"
        prob += pulp.lpSum([v[i] for i in players]) == 1, "One_ViceCaptain"
        for i in players:
            prob += c[i] <= x[i], f"Captain_Selected_{i}"
            prob += v[i] <= x[i], f"ViceCaptain_Selected_{i}"
            prob += c[i] + v[i] <= 1, f"No_Dual_Role_{i}"

        # Solve the optimization problem with a fallback
        try:
            prob.solve(pulp.PULP_CBC_CMD(msg=0))  # Attempt to use CBC solver
        except AttributeError:
            logger.warning("CBC solver not found. Falling back to default solver.")
            prob.solve()  # Use default solver if CBC is unavailable
        logger.info("LP Status: %s", pulp.LpStatus[prob.status])
        if pulp.LpStatus[prob.status] != "Optimal":
            logger.warning("No optimal solution found! Check solver output for details.")
            return None

        selected = [i for i in players if pulp.value(x[i]) == 1]
        team = team_df.loc[selected].copy()
        team.sort_values("Score", ascending=False, inplace=True)
        team["Position"] = "Player"
        if len(team) > 0:
            team.iloc[0, team.columns.get_loc("Position")] = "Captain"
        if len(team) > 1:
            team.iloc[1, team.columns.get_loc("Position")] = "Vice Captain"

        logger.info("Selected team size: %d", len(team))
        selected_roles = team["Role"].value_counts()
        logger.info("Selected team roles: %s", selected_roles.to_dict())

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