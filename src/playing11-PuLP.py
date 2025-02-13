import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pulp
import yaml


class FantasyTeamOptimizer:
    """Optimizes fantasy cricket team selection using historical data and machine learning"""

    def __init__(self, config):
        """
        Initialize optimizer with configuration parameters

        :param config: Dictionary containing:
            - data_paths: Dictionary of file paths for input data
            - model_params: Parameters for RandomForestRegressor
            - team_rules: Constraints for team composition
        """
        self.config = config
        self.model = RandomForestRegressor(**config.get("model_params", {}))
        self.data = {}
        self.features = None
        self.target = None

    def load_data(self):
        """Load and validate all required datasets"""
        try:
            self.data["batting"] = pd.read_csv(self.config["data_paths"]["batting"])
            self.data["bowling"] = pd.read_csv(self.config["data_paths"]["bowling"])
            self.data["fielding"] = pd.read_csv(self.config["data_paths"]["fielding"])
            self.data["form"] = pd.read_csv(self.config["data_paths"]["form"])
            self.data["roster"] = pd.read_csv(self.config["data_paths"]["roster"])
        except Exception as e:
            raise RuntimeError(f"Data loading failed: {str(e)}")

    def calculate_fantasy_points(self):
        """Calculate fantasy points for all historical performances"""
        # Batting points
        self.data["batting"]["Fantasy"] = self.data["batting"].apply(
            self._calculate_batting_points, axis=1
        )

        # Bowling points
        self.data["bowling"]["Fantasy"] = self.data["bowling"].apply(
            self._calculate_bowling_points, axis=1
        )

        # Fielding points
        self.data["fielding"]["Fantasy"] = self.data["fielding"].apply(
            self._calculate_fielding_points, axis=1
        )

    def prepare_features(self):
        """Create feature matrix and target variable"""
        batting_agg = self._aggregate_performance("batting", "Batting Fantasy")
        bowling_agg = self._aggregate_performance("bowling", "Bowling Fantasy")
        fielding_agg = self._aggregate_performance("fielding", "Fielding Fantasy")

        features_df = batting_agg.merge(bowling_agg, on="Player", how="outer")
        features_df = features_df.merge(fielding_agg, on="Player", how="outer")
        features_df = features_df.fillna(0)

        features_df = features_df.merge(self.data["form"], on="Player", how="inner")

        features_df["Total Fantasy"] = features_df[
            ["Batting Fantasy", "Bowling Fantasy", "Fielding Fantasy"]
        ].sum(axis=1)

        self.features = features_df
        self.target = features_df["Total Fantasy"]

    def train_model(self):
        """Train the machine learning model"""
        X = self.features[self.config["feature_columns"]]
        y = self.target

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model.fit(X_train, y_train)

    def optimize_team(self):
        """Select optimal team using linear programming"""
        current_players = self._get_current_players()
        predictions = self.model.predict(
            current_players[self.config["feature_columns"]]
        )
        current_players["Predicted"] = predictions

        problem = pulp.LpProblem("OptimalTeam", pulp.LpMaximize)
        selection = pulp.LpVariable.dict(
            "Select", current_players["Player"], cat="Binary"
        )

        problem += pulp.lpSum(
            [
                current_players.loc[current_players["Player"] == p, "Predicted"].values[
                    0
                ]
                * selection[p]
                for p in current_players["Player"]
            ]
        )

        problem += pulp.lpSum(selection.values()) == 11

        problem.solve(pulp.PULP_CBC_CMD(msg=False))

        return [p for p in current_players["Player"] if pulp.value(selection[p]) == 1]

    def _aggregate_performance(self, discipline, col_name):
        """Helper to aggregate performance metrics"""
        return (
            self.data[discipline]
            .groupby("Player")["Fantasy"]
            .mean()
            .reset_index()
            .rename(columns={"Fantasy": col_name})
        )

    def _get_current_players(self):
        """Process current roster data"""
        roster = self.data["roster"].rename(columns={"Player Name": "Player"})
        return roster.merge(self.features, on="Player", how="inner")

    @staticmethod
    def _calculate_batting_points(row):
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
    def _calculate_bowling_points(row):
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
    def _calculate_fielding_points(row):
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
            config = config["model"]["PuLP"]
        except yaml.YAMLError as exc:
            print(exc)
            exit(1)

    try:
        optimizer = FantasyTeamOptimizer(config)
        optimizer.load_data()
        optimizer.calculate_fantasy_points()
        optimizer.prepare_features()
        optimizer.train_model()
        best_team = optimizer.optimize_team()

        print("\nOptimal Playing 11:")
        for player in best_team:
            print(f"- {player}")

    except Exception as e:
        print(f"Error optimizing team: {str(e)}")
        exit(1)
