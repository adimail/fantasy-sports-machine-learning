import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.model_selection import train_test_split
import pulp

class FantasyTeamOptimizer:


    def __init__(self):
        """Initialize the optimizer with an ensemble model."""
        print("Initializing Fantasy Team Optimizer...")

        # Define individual models
        self.xgb_model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
        self.rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        self.gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)

        # Ensemble model
        self.model = VotingRegressor(estimators=[
            ('xgb', self.xgb_model),
            ('rf', self.rf_model),
            ('gb', self.gb_model)
        ])

        self.data = {}
        self.features = []
        self.target = []
        print("Ensemble model initialized.")

    def load_data(self):
       
        print("Loading data from CSV files...")
        try:
            self.data["batting"] = pd.read_csv("batting_data.csv")
            self.data["bowling"] = pd.read_csv("bowling_data.csv")
            self.data["fielding"] = pd.read_csv("fielding_data.csv")
            self.data["input"] = pd.read_csv("input.csv")  # Contains Match Number, Team Name, Player Name
            print("Data loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Data loading failed: {str(e)}")

    def filter_players(self):
        
        print("Filtering players based on input.csv...")
        valid_players = self.data["input"]["Player Name"].unique()
        self.data["batting"] = self.data["batting"][self.data["batting"]["Player"].isin(valid_players)]
        self.data["bowling"] = self.data["bowling"][self.data["bowling"]["Player"].isin(valid_players)]
        self.data["fielding"] = self.data["fielding"][self.data["fielding"]["Player"].isin(valid_players)]
        print("Players filtered successfully.")

    def calculate_fantasy_points(self):
        
        print("Calculating fantasy points...")
        self.data["batting"]["Fantasy"] = self.data["batting"].apply(self._calculate_batting_points, axis=1)
        self.data["bowling"]["Fantasy"] = self.data["bowling"].apply(self._calculate_bowling_points, axis=1)
        self.data["fielding"]["Fantasy"] = self.data["fielding"].apply(self._calculate_fielding_points, axis=1)
        print("Fantasy points calculated.")

    def _aggregate_performance(self, dataset, col_name):
        
        if dataset in self.data and not self.data[dataset].empty:
            return (
                self.data[dataset]
                .groupby("Player")["Fantasy"]
                .mean()
                .reset_index()
                .rename(columns={"Fantasy": col_name})
            )
        else:
            return pd.DataFrame(columns=["Player", col_name])  # Return empty DataFrame if no data
    def classify_role(self, player_name):
        
        player_stats = self.features[self.features["Player"] == player_name]
        if player_stats.empty:
            return "Unknown"

        batting_points = player_stats["Batting Fantasy"].values[0]
        bowling_points = player_stats["Bowling Fantasy"].values[0]
        fielding_points = player_stats["Fielding Fantasy"].values[0]

        if batting_points > 2 * bowling_points:
            return "Batter"
        elif bowling_points > 2 * batting_points:
            return "Bowler"
        elif batting_points > 0 and bowling_points > 0:
            return "All-Rounder"
        elif fielding_points > 0:
            return "Wicketkeeper"
        return "Unknown"


    def prepare_features(self):
        
        print("Preparing feature matrix...")
        batting_agg = self._aggregate_performance("batting", "Batting Fantasy")
        bowling_agg = self._aggregate_performance("bowling", "Bowling Fantasy")
        fielding_agg = self._aggregate_performance("fielding", "Fielding Fantasy")

        features_df = batting_agg.merge(bowling_agg, on="Player", how="outer").merge(fielding_agg, on="Player", how="outer")
        features_df = features_df.fillna(0)
        features_df["Total Fantasy"] = features_df[["Batting Fantasy", "Bowling Fantasy", "Fielding Fantasy"]].sum(axis=1)

        self.features = features_df
        self.target = features_df["Total Fantasy"]
        print("Feature matrix prepared with", len(self.features), "players.")

    def train_model(self):
        
        print("Training the ensemble model...")
        X = self.features[["Batting Fantasy", "Bowling Fantasy", "Fielding Fantasy"]]
        y = self.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        print("Model training completed.")

    def optimize_team(self):
        
        print("Optimizing fantasy team selection...")
        selected_players = self.features.copy()
        selected_players["Predicted"] = self.model.predict(selected_players[["Batting Fantasy", "Bowling Fantasy", "Fielding Fantasy"]])
        selected_players = selected_players.sort_values(by="Predicted", ascending=False)

        prob = pulp.LpProblem("BestPlaying11", pulp.LpMaximize)
        players = selected_players["Player"].tolist()
        player_vars = pulp.LpVariable.dicts("Player", players, cat="Binary")

        # Objective: Maximize predicted points
        prob += pulp.lpSum([selected_players.loc[selected_players["Player"] == p, "Predicted"].values[0] * player_vars[p] for p in players])

        # Role-based constraints
        prob += pulp.lpSum([player_vars[p] for p in players]) == 11
        prob += pulp.lpSum([player_vars[p] for p in players if self.classify_role(p) == "Batter"]) >= 3
        prob += pulp.lpSum([player_vars[p] for p in players if self.classify_role(p) == "Bowler"]) >= 4
        prob += pulp.lpSum([player_vars[p] for p in players if self.classify_role(p) == "All-Rounder"]) >= 2
        prob += pulp.lpSum([player_vars[p] for p in players if self.classify_role(p) == "Wicketkeeper"]) >= 1

        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        final_team = selected_players[selected_players["Player"].isin([p for p in players if pulp.value(player_vars[p]) == 1])]
        return final_team

    @staticmethod
    def _calculate_batting_points(row):
        
        points = row.get("Runs", 0) + row.get("4s", 0) * 4 + row.get("6s", 0) * 6
        return points

    @staticmethod
    def _calculate_bowling_points(row):
        
        points = row.get("Wkts", 0) * 25 + (row.get("Dot Balls", 0) // 3) * 1
        return points

    @staticmethod
    def _calculate_fielding_points(row):
        
        points = row.get("Ct", 0) * 8 + row.get("St", 0) * 12 + row.get("RunOut", 0) * 6
        return points


if __name__ == "__main__":
    print("Starting FantasyTeamOptimizer process...")

    optimizer = FantasyTeamOptimizer()
    optimizer.load_data()
    optimizer.filter_players()
    optimizer.calculate_fantasy_points()
    optimizer.prepare_features()
    optimizer.train_model()

    final_team = optimizer.optimize_team()

    print("\n🔥 Best Playing 11:")
    print(final_team[["Player", "Predicted"]].to_string(index=False))
