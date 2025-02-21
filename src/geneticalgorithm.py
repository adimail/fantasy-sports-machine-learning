import pandas as pd
import sys
import yaml
import logging
import random
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
            print(
                "Warning: Merged DataFrame is empty. Please verify that the player names and types match between the input files."
            )

        # Standardize role names to match optimization constraints.
        # Mapping: "ALL" -> "All Rounder", "BOWL" -> "Bowler", "BAT" -> "Batsmen", "WK" -> "Wicket Keeper"
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

    def _generate_candidate(self, team_df):
        """
        Generate a candidate solution as a set of 11 random player indices.
        """
        return sorted(random.sample(list(team_df.index), 11))

    def _fitness(self, candidate, team_df):
        """
        Compute the fitness of a candidate team.
        The fitness is the sum of scores plus bonus: highest scoring player gets a bonus equal to its score
        (captain) and second highest gets 0.5 bonus (vice-captain). A penalty is applied for any constraint violation.
        Constraints:
          - At least 4 Batsmen.
          - At least 5 players with bowling contributions (Bowler or All Rounder).
          - At least 3 Bowler.
          - At least 1 Wicket Keeper.
          - At least 2 All Rounder.
        """
        selected = team_df.loc[candidate]
        total_score = selected["Score"].sum()
        sorted_scores = selected["Score"].sort_values(ascending=False)
        bonus = 0
        if len(sorted_scores) >= 1:
            bonus += sorted_scores.iloc[0]  # Captain bonus
        if len(sorted_scores) >= 2:
            bonus += 0.5 * sorted_scores.iloc[1]  # Vice-captain bonus

        # Count players by role
        role_counts = selected["Role"].value_counts().to_dict()
        missing = 0
        missing += max(0, 4 - role_counts.get("Batsmen", 0))
        missing += max(0, 3 - role_counts.get("Bowler", 0))
        missing += max(0, 2 - role_counts.get("All Rounder", 0))
        missing += max(0, 1 - role_counts.get("Wicket Keeper", 0))
        bowling_contrib = role_counts.get("Bowler", 0) + role_counts.get(
            "All Rounder", 0
        )
        missing += max(0, 5 - bowling_contrib)

        penalty = 1000 * missing  # Penalty factor
        fitness = total_score + bonus - penalty
        return fitness

    def _tournament_selection(self, population, fitnesses, tournament_size=3):
        """
        Perform tournament selection and return one candidate from the population.
        """
        selected = random.sample(list(zip(population, fitnesses)), tournament_size)
        selected.sort(key=lambda x: x[1], reverse=True)
        return selected[0][0]

    def _crossover(self, parent1, parent2, team_df):
        """
        Produce a child candidate by performing a set-based crossover.
        The child contains common players from both parents, and fills remaining positions randomly.
        """
        common = set(parent1) & set(parent2)
        remaining = list(set(team_df.index) - common)
        child = list(common)
        # Fill up until 11 players; if needed, randomly pick from the union of parents first.
        union_parents = list(set(parent1) | set(parent2))
        random.shuffle(union_parents)
        for p in union_parents:
            if p not in child and len(child) < 11:
                child.append(p)
        # If still not 11, fill randomly from remaining players.
        while len(child) < 11:
            child.append(random.choice(remaining))
        return sorted(child)

    def _mutate(self, candidate, team_df, mutation_rate=0.1):
        """
        Mutate the candidate solution by replacing a random player with a new random player.
        Mutation occurs with probability equal to mutation_rate.
        """
        candidate = candidate.copy()
        if random.random() < mutation_rate:
            pos = random.randint(0, 10)
            available = list(set(team_df.index) - set(candidate))
            if available:
                candidate[pos] = random.choice(available)
        return sorted(candidate)

    def optimize_team(self):
        """
        Optimize fantasy team selection from the merged data using a genetic algorithm.

        Constraints:
          - Exactly 11 players are selected.
          - At least 4 Batsmen.
          - At least 5 players with bowling contributions (Bowler or All Rounder).
          - At least 3 Bowlers.
          - At least 1 Wicket Keeper.
          - At least 2 All Rounders.

        Returns:
            A DataFrame of selected players with their computed Score and assigned team role (Captain, Vice Captain, Player).
        """
        team_df = self.merged_df.copy()
        population_size = self.config["ga"].get("population_size", 50)
        generations = self.config["ga"].get("generations", 100)
        tournament_size = self.config["ga"].get("tournament_size", 3)
        mutation_rate = self.config["ga"].get("mutation_rate", 0.1)

        # Initialize population
        population = [self._generate_candidate(team_df) for _ in range(population_size)]
        best_candidate = None
        best_fitness = -np.inf

        for gen in range(generations):
            fitnesses = [self._fitness(candidate, team_df) for candidate in population]
            # Update best candidate
            for candidate, fit in zip(population, fitnesses):
                if fit > best_fitness:
                    best_fitness = fit
                    best_candidate = candidate
            logger.info("Generation %d: Best fitness = %.2f", gen, best_fitness)

            # Create new population using tournament selection, crossover, and mutation
            new_population = []
            while len(new_population) < population_size:
                parent1 = self._tournament_selection(
                    population, fitnesses, tournament_size
                )
                parent2 = self._tournament_selection(
                    population, fitnesses, tournament_size
                )
                child = self._crossover(parent1, parent2, team_df)
                child = self._mutate(child, team_df, mutation_rate)
                new_population.append(child)
            population = new_population

        # Retrieve the best candidate found
        selected = best_candidate
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
