import pandas as pd
import yaml


class FantasyTeamBuilder:
    """Builds optimal fantasy cricket teams using historical performance and recent form"""

    def __init__(self, config):
        """
        Initialize team builder with configuration parameters

        :param config: Dictionary containing:
            - data_paths: Dictionary of file paths for input data
            - scoring_weights: Dictionary of weights for composite scoring
        """
        self.config = config
        self.data = {}
        self.aggregated_data = None
        self.composite_scores = None

    def load_data(self):
        """Load and validate all required datasets"""
        try:
            self.data = {
                "batting": pd.read_csv(self.config["data_paths"]["batting"]),
                "bowling": pd.read_csv(self.config["data_paths"]["bowling"]),
                "fielding": pd.read_csv(self.config["data_paths"]["fielding"]),
                "form": pd.read_csv(self.config["data_paths"]["form"]),
                "roster": pd.read_csv(self.config["data_paths"]["roster"]),
            }
        except Exception as e:
            raise RuntimeError(f"Data loading failed: {str(e)}")

    def calculate_performance_scores(self):
        """Calculate and aggregate fantasy points across all disciplines"""
        self._calculate_discipline_points()
        self.aggregated_data = self._aggregate_player_performance()

    def calculate_composite_scores(self):
        """Calculate final composite scores combining historical performance and recent form"""
        if self.aggregated_data is None:
            raise ValueError("Must calculate performance scores first")

        merged_data = pd.merge(
            self.aggregated_data, self.data["form"], on="Player", how="inner"
        )

        merged_data["Fantasy Norm"] = self._normalize(merged_data["Total Fantasy"])
        merged_data["Form Norm"] = self._normalize(
            merged_data[["Batting Form", "Bowling Form", "Fielding Form"]].mean(axis=1)
        )

        weights = self.config.get("scoring_weights", {"historical": 0.5, "form": 0.5})
        merged_data["Composite Score"] = (
            weights["historical"] * merged_data["Fantasy Norm"]
            + weights["form"] * merged_data["Form Norm"]
        )

        self.composite_scores = merged_data

    def select_optimal_team(self):
        """Select best 11 players based on composite scores"""
        if self.composite_scores is None:
            raise ValueError("Must calculate composite scores first")

        roster = self.data["roster"].rename(columns={"Player Name": "Player"})
        selection_pool = pd.merge(
            roster, self.composite_scores, on="Player", how="inner"
        )

        return selection_pool.sort_values("Composite Score", ascending=False).head(11)

    def _calculate_discipline_points(self):
        """Calculate fantasy points for each discipline"""
        self.data["batting"]["Fantasy"] = self.data["batting"].apply(
            self._batting_points, axis=1
        )
        self.data["bowling"]["Fantasy"] = self.data["bowling"].apply(
            self._bowling_points, axis=1
        )
        self.data["fielding"]["Fantasy"] = self.data["fielding"].apply(
            self._fielding_points, axis=1
        )

    def _aggregate_player_performance(self):
        """Aggregate performance across all disciplines"""
        batting_agg = self._aggregate_discipline("batting", "Batting Fantasy")
        bowling_agg = self._aggregate_discipline("bowling", "Bowling Fantasy")
        fielding_agg = self._aggregate_discipline("fielding", "Fielding Fantasy")

        agg_df = batting_agg.merge(bowling_agg, on="Player", how="outer")
        agg_df = agg_df.merge(fielding_agg, on="Player", how="outer")

        agg_df["Total Fantasy"] = agg_df[
            ["Batting Fantasy", "Bowling Fantasy", "Fielding Fantasy"]
        ].sum(axis=1)

        return agg_df.fillna(0)

    def _aggregate_discipline(self, discipline, col_name):
        """Aggregate performance for a single discipline"""
        return (
            self.data[discipline]
            .groupby("Player")["Fantasy"]
            .mean()
            .reset_index()
            .rename(columns={"Fantasy": col_name})
        )

    @staticmethod
    def _normalize(series):
        """Normalize a pandas series to 0-100 scale"""
        return (series - series.min()) / (series.max() - series.min()) * 100

    @staticmethod
    def _batting_points(row):
        points = row.get("Runs", 0)
        points += row.get("4s", 0) * 4
        points += row.get("6s", 0) * 6

        runs = row.get("Runs", 0)
        if runs >= 25:
            points += 4
        if runs >= 50:
            points += 8
        if runs >= 75:
            points += 12
        if runs >= 100:
            points += 16
        if runs >= 125:
            points += 20
        if runs >= 150:
            points += 24

        if runs == 0 and row.get("Inns", 1) > 0:
            points -= 3

        if row.get("BF", 0) >= 20:
            sr = row.get("SR", 0)
            if sr > 140:
                points += 6
            elif sr > 120:
                points += 4
            elif sr >= 100:
                points += 2
            elif 40 <= sr <= 50:
                points -= 2
            elif 30 <= sr < 40:
                points -= 4
            elif sr < 30:
                points -= 6

        return points

    @staticmethod
    def _bowling_points(row):
        points = row.get("Wkts", 0) * 25
        wkts = row.get("Wkts", 0)
        if wkts >= 4:
            points += 4
        if wkts >= 5:
            points += 8
        if wkts >= 6:
            points += 12

        points += row.get("Mdns", 0) * 4

        if row.get("Overs", 0) >= 5:
            econ = row.get("Econ", 0)
            if econ < 2.5:
                points += 6
            elif econ < 3.5:
                points += 4
            elif econ < 4.5:
                points += 2
            elif 7 <= econ <= 8:
                points -= 2
            elif 8 < econ <= 9:
                points -= 4
            elif econ > 9:
                points -= 6

        return points

    @staticmethod
    def _fielding_points(row):
        points = row.get("Ct", 0) * 8
        if row.get("Ct", 0) >= 3:
            points += 4
        points += row.get("St", 0) * 12
        points += row.get("Ct Wk", 0) * 6
        return points


if __name__ == "__main__":
    with open("config.yaml") as stream:
        try:
            config = yaml.safe_load(stream)
            config = config["model"]["composite"]
        except yaml.YAMLError as exc:
            print(exc)
            exit(1)

    try:
        builder = FantasyTeamBuilder(config)
        builder.load_data()
        builder.calculate_performance_scores()
        builder.calculate_composite_scores()
        best_team = builder.select_optimal_team()

        teams = builder.data["roster"]["Team Name"].unique()
        print(f"\nPlaying 11 ({teams[0]} vs {teams[1]})\n")
        for player in best_team["Player"]:
            print(f"- {player}")

    except Exception as e:
        print(f"Error building team: {str(e)}")
        exit(1)
