import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import pulp
import yaml


class FantasyTeamOptimizer:
    """Advanced fantasy cricket optimizer with XGBoost and detailed constraints"""

    def __init__(self, config):
        self.config = config
        self.model = None
        self.data = {}
        self.feature_processor = None
        self.optimization_problem = None

        self.xgb_params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            **config.get("xgb_params", {}),
        }
        print("Initialized FantasyTeamOptimizer with config:", config)

    def load_and_preprocess_data(self):
        """Load and preprocess data from multiple sources"""
        try:
            print("Loading raw data...")
            self._load_raw_data()
            print("Raw data loaded. DataFrame columns for each source:")
            for key, df in self.data.items():
                print(f"  {key}: {df.columns.tolist()}")

            print("Creating feature pipeline...")
            self._create_feature_pipeline()

            print("Processing player features...")
            self._process_player_features()
            print(
                "Data preprocessing completed. Processed features columns:",
                self.data["features"].columns.tolist(),
            )
        except Exception as e:
            raise RuntimeError(f"Data processing failed: {str(e)}")

    def train_performance_model(self):
        """Train XGBoost model with hyperparameter tuning"""
        try:
            print("Training performance model...")
            X = self.data["features"]
            y = self.data["target"]
            print("Feature matrix shape:", X.shape)
            print("Target vector shape:", y.shape)

            tscv = TimeSeriesSplit(n_splits=5)
            param_grid = {
                "xgbregressor__n_estimators": [100, 200],
                "xgbregressor__max_depth": [3, 5],
                "xgbregressor__learning_rate": [0.01, 0.1],
            }

            model_pipe = Pipeline(
                [
                    ("preprocessor", self.feature_processor),
                    ("xgbregressor", xgb.XGBRegressor(**self.xgb_params)),
                ]
            )

            grid_search = GridSearchCV(
                model_pipe,
                param_grid,
                cv=tscv,
                scoring="neg_mean_squared_error",
                n_jobs=-1,
                error_score="raise",
                verbose=2,
            )

            grid_search.fit(X, y)
            self.model = grid_search.best_estimator_

            print("Model training completed.")
            print(f"Best parameters: {grid_search.best_params_}")
            best_rmse = (-grid_search.best_score_) ** 0.5
            print(f"Best RMSE: {best_rmse:.2f}")

        except Exception as e:
            raise RuntimeError(f"Model training failed: {str(e)}")

    def optimize_team_selection(self):
        """Optimize team selection with detailed constraints"""
        try:
            print("Optimizing team selection...")
            roster = self.data["roster"].copy()
            merged_players = roster.merge(
                self.data["features"], on="Player", how="left"
            )
            print(
                f"Merged roster: {merged_players.shape[0]} players; columns: {merged_players.columns.tolist()}"
            )

            feature_cols = (
                self.config["feature_columns"]["numeric"]
                + self.config["feature_columns"]["categorical"]
            )
            merged_players["Predicted_Points"] = self.model.predict(
                merged_players[feature_cols]
            )
            self.data["roster"] = merged_players

            self._create_optimization_model(merged_players)
            result = self._solve_optimization()
            print("Team optimization completed.")
            return result

        except Exception as e:
            raise RuntimeError(f"Optimization failed: {str(e)}")

    def _load_raw_data(self):
        """Load data from the configured CSV sources"""
        data_paths = self.config["data_paths"]
        self.data.update(
            {
                "batting": pd.read_csv(data_paths["batting"]),
                "bowling": pd.read_csv(data_paths["bowling"]),
                "fielding": pd.read_csv(data_paths["fielding"]),
                "form": pd.read_csv(data_paths["form"]),
                "roster": pd.read_csv(data_paths["roster"]),
            }
        )
        if "Player Name" in self.data["roster"].columns:
            self.data["roster"].rename(columns={"Player Name": "Player"}, inplace=True)

    def _create_feature_pipeline(self):
        """Set up the feature processing pipeline"""
        numeric_features = self.config["feature_columns"]["numeric"]
        categorical_features = self.config["feature_columns"]["categorical"]
        print("Numeric features:", numeric_features)
        print("Categorical features:", categorical_features)

        self.feature_processor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ]
        )

    def _process_player_features(self):
        """Merge data from various sources and compute advanced features"""
        print("Calculating fantasy points for each discipline...")
        self._calculate_fantasy_points()

        print("Merging batting, bowling, fielding, and form data...")
        features = self._create_advanced_features()

        initial_count = features.shape[0]
        features = features.dropna(subset=["Total_Fantasy"])
        dropped = initial_count - features.shape[0]
        if dropped > 0:
            print(f"Dropped {dropped} rows due to missing Total_Fantasy.")

        self.data["features"] = features
        self.data["target"] = features["Total_Fantasy"]
        print("Advanced features created.")

    def _create_advanced_features(self):
        """Merge batting, bowling, fielding, and form data to create features"""
        merged = (
            self.data["batting"]
            .merge(self.data["bowling"], on="Player", suffixes=("_bat", "_bowl"))
            .merge(self.data["fielding"], on="Player")
        )
        merged = merged.merge(self.data["form"], on="Player", how="left")

        if "Ave_bat" in merged.columns:
            merged.rename(columns={"Ave_bat": "Batting_Avg"}, inplace=True)
            print("Renamed 'Ave_bat' to 'Batting_Avg'")
        else:
            print("Warning: 'Ave_bat' not found in merged data.")

        if "Econ" in merged.columns:
            merged.rename(columns={"Econ": "Bowling_Econ"}, inplace=True)
            print("Renamed 'Econ' to 'Bowling_Econ'")
        else:
            print("Warning: 'Econ' not found in merged data.")

        for col in ["Fantasy_bat", "Fantasy_bowl", "Fantasy"]:
            if col not in merged.columns:
                raise ValueError(f"Expected fantasy points column '{col}' is missing.")

        merged["Total_Fantasy"] = (
            merged["Fantasy_bat"] + merged["Fantasy_bowl"] + merged["Fantasy"]
        )

        form_cols = ["Batting Form", "Bowling Form", "Fielding Form"]
        if all(col in merged.columns for col in form_cols):
            merged["Recent_Form"] = merged[form_cols].mean(axis=1)
            print("Computed 'Recent_Form' as average of form columns.")
        else:
            raise ValueError("Expected form columns not found in merged data.")

        for col, default_val in zip(
            ["Role", "Venue_Type", "Opposition_Strength"], ["Batsman", "Home", "Medium"]
        ):
            if col not in merged.columns:
                merged[col] = default_val
                print(f"Added default column '{col}' with value '{default_val}'.")

        return merged

    def _calculate_fantasy_points(self):
        """Calculate fantasy points for batting, bowling, and fielding data"""
        for df_name in ["batting", "bowling", "fielding"]:
            print(f"Calculating fantasy points for {df_name}...")
            self.data[df_name]["Fantasy"] = self.data[df_name].apply(
                getattr(self, f"_calculate_{df_name}_points"), axis=1
            )
            print(f"Fantasy points calculated for {df_name}.")

    def _create_optimization_model(self, players):
        """Define the optimization problem with team constraints"""
        print("Creating optimization model...")

        unique_players = [str(p) for p in players["Player"].unique()]
        print("Unique players count for optimization:", len(unique_players))

        player_vars = pulp.LpVariable.dicts("Player", unique_players, cat="Binary")

        # Objective: maximize total predicted points.
        # Iterate over the unique list so that p is a string.
        prob = pulp.LpProblem("FantasyTeamOptimization", pulp.LpMaximize)
        prob += pulp.lpSum(
            [
                players.loc[players["Player"] == p, "Predicted_Points"].values[0]
                * player_vars[p]
                for p in unique_players
            ]
        )

        self._add_basic_constraints(prob, players, player_vars)
        self._add_team_composition_constraints(prob, players, player_vars)
        self._add_budget_constraint(prob, players, player_vars)

        self.optimization_problem = prob
        print("Optimization model created.")

    def _add_basic_constraints(self, prob, players, variables):
        """Add core constraints such as total player count and team limits"""
        print("Adding basic team constraints...")
        prob += pulp.lpSum(variables.values()) == 11
        if "Team" in players.columns:
            teams = players["Team"].unique()
            for team in teams:
                prob += pulp.lpSum(
                    variables[p] for p in players[players["Team"] == team]["Player"]
                ) <= self.config["team_rules"].get("max_players_per_team", 7)
        print("Basic constraints added.")

    def _add_team_composition_constraints(self, prob, players, variables):
        """Enforce role-based team composition constraints"""
        print("Adding role-based composition constraints...")
        roles = self.config["team_rules"]["required_roles"]
        for role, (min_count, max_count) in roles.items():
            prob += (
                pulp.lpSum(
                    variables[p] for p in players[players["Role"] == role]["Player"]
                )
                >= min_count
            )
            prob += (
                pulp.lpSum(
                    variables[p] for p in players[players["Role"] == role]["Player"]
                )
                <= max_count
            )
        print("Role-based constraints added.")

    def _add_budget_constraint(self, prob, players, variables):
        """Apply the salary cap constraint if Salary information is available"""
        print("Adding budget constraint...")
        if "Salary" in players.columns:
            prob += (
                pulp.lpSum(
                    players.loc[players["Player"] == p, "Salary"].values[0]
                    * variables[p]
                    for p in players["Player"]
                )
                <= self.config["team_rules"]["salary_cap"]
            )
        print("Budget constraint added.")

    def _solve_optimization(self):
        """Solve the optimization problem and select captain and vice-captain"""
        print("Solving optimization problem...")
        self.optimization_problem.solve(pulp.PULP_CBC_CMD(msg=True))
        selected_players = [
            p
            for p in self.optimization_problem.variables()
            if p.varValue == 1 and "Captain" not in p.name
        ]
        print("Selected players:", [p.name for p in selected_players])
        captain, vice_captain = self._select_captains(selected_players)
        print(f"Captain: {captain}, Vice-Captain: {vice_captain}")
        return {
            "team": [p.name.replace("Player_", "") for p in selected_players],
            "captain": captain,
            "vice_captain": vice_captain,
        }

    def _select_captains(self, players):
        """Select the captain and vice-captain based on predicted points"""
        print("Selecting captain and vice-captain...")
        points = self.data["roster"].set_index("Player")["Predicted_Points"]
        sorted_players = sorted(
            [p.name.replace("Player_", "") for p in players],
            key=lambda x: points.get(x, 0),
            reverse=True,
        )
        if len(sorted_players) < 2:
            return sorted_players[0], None
        return sorted_players[0], sorted_players[1]

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
            econ = row.get("Bowling_Econ", 0)
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
    with open("config.yaml") as f:
        config = yaml.safe_load(f)["model"]["optimization_config"]

    try:
        optimizer = FantasyTeamOptimizer(config)
        optimizer.load_and_preprocess_data()
        optimizer.train_performance_model()
        result = optimizer.optimize_team_selection()

        print("\nOptimal Playing 11:")
        for player in result["team"]:
            print(f"- {player}")
        print(f"\nCaptain: {result['captain']}")
        print(f"Vice-Captain: {result['vice_captain']}")
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)
