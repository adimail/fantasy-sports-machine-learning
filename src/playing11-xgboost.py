#!/usr/bin/env python3
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import pulp
import yaml

def compute_batting_points(row):
    points = row.get('Runs', 0)
    points += row.get('4s', 0) * 4
    points += row.get('6s', 0) * 6
    runs = row.get('Runs', 0)
    if runs >= 25: points += 4
    if runs >= 50: points += 8
    if runs >= 75: points += 12
    if runs >= 100: points += 16
    if runs >= 125: points += 20
    if runs >= 150: points += 24
    if runs == 0 and row.get('Inns', 1) > 0:
        points -= 3
    if row.get('BF', 0) >= 20:
        sr = row.get('SR', 0)
        if sr > 140: points += 6
        elif sr > 120: points += 4
        elif sr >= 100: points += 2
        elif 40 <= sr <= 50: points -= 2
        elif 30 <= sr < 40: points -= 4
        elif sr < 30: points -= 6
    return points

def compute_bowling_points(row):
    points = row.get('Wkts', 0) * 25
    wkts = row.get('Wkts', 0)
    if wkts >= 4: points += 4
    if wkts >= 5: points += 8
    if wkts >= 6: points += 12
    points += row.get('Mdns', 0) * 4
    if row.get('Overs', 0) >= 5:
        econ = row.get('Bowling_Econ', 0)
        if econ < 2.5: points += 6
        elif econ < 3.5: points += 4
        elif econ < 4.5: points += 2
        elif 7 <= econ <= 8: points -= 2
        elif 8 < econ <= 9: points -= 4
        elif econ > 9: points -= 6
    return points

def compute_fielding_points(row):
    points = row.get('Ct', 0) * 8
    if row.get('Ct', 0) >= 3: points += 4
    points += row.get('St', 0) * 12
    points += row.get('Ct Wk', 0) * 6
    return points

class BestPlaying11Optimizer:
    """Calculates the best playing 11 using performance prediction and optimization.
       Uses only the provided CSV files.
    """

    def __init__(self, config):
        self.config = config
        self.data = {}
        self.model = None
        self.feature_processor = None
        self.optimization_problem = None

        self.xgb_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            **config.get('xgb_params', {})
        }
        print("Initialized BestPlaying11Optimizer with config:", config)

    def load_raw_data(self):
        """Load CSV files from the provided data paths."""
        data_paths = self.config['data_paths']
        self.data['batting'] = pd.read_csv(data_paths['batting'])
        self.data['bowling'] = pd.read_csv(data_paths['bowling'])
        self.data['fielding'] = pd.read_csv(data_paths['fielding'])
        self.data['form'] = pd.read_csv(data_paths['form'])
        self.data['roster'] = pd.read_csv(data_paths['roster'])
        # Rename "Player Name" to "Player" if necessary
        if 'Player Name' in self.data['roster'].columns:
            self.data['roster'].rename(columns={"Player Name": "Player"}, inplace=True)
        print("Loaded raw data.")

    def compute_fantasy_points(self):
        """Compute fantasy points for each discipline."""
        print("Computing fantasy points...")
        self.data['batting']['Fantasy_bat'] = self.data['batting'].apply(compute_batting_points, axis=1)
        self.data['bowling']['Fantasy_bowl'] = self.data['bowling'].apply(compute_bowling_points, axis=1)
        self.data['fielding']['Fantasy_field'] = self.data['fielding'].apply(compute_fielding_points, axis=1)
        print("Fantasy points computed.")

    def merge_data(self):
        """Merge batting, bowling, fielding, and form data on 'Player'."""
        print("Merging data...")
        merged = (self.data['batting']
                  .merge(self.data['bowling'], on='Player', suffixes=('_bat', '_bowl'))
                  .merge(self.data['fielding'], on='Player'))
        merged = merged.merge(self.data['form'], on='Player', how='left')

        merged.columns = merged.columns.str.strip()
        merged['Total_Fantasy'] = (merged['Fantasy_bat'] +
                                   merged['Fantasy_bowl'] +
                                   merged['Fantasy_field'])
        form_cols = ['Batting Form', 'Bowling Form', 'Fielding Form']
        if all(col in merged.columns for col in form_cols):
            merged['Recent_Form'] = merged[form_cols].mean(axis=1)
            print("Computed 'Recent_Form' as average of form columns.")
        else:
            merged['Recent_Form'] = np.nan

        before = merged.shape[0]
        merged = merged.dropna(subset=['Total_Fantasy'])
        print(f"Dropped {before - merged.shape[0]} rows due to missing Total_Fantasy.")

        for col, default in zip(['Role', 'Venue_Type', 'Opposition_Strength'],
                                  ['Batsman', 'Home', 'Medium']):
            if col not in merged.columns:
                merged[col] = default
                print(f"Added default column '{col}' with value '{default}'.")

        self.data['merged'] = merged
        print("Data merge complete. Merged data shape:", merged.shape)

    def setup_feature_pipeline(self):
        """Set up a feature processing pipeline using the specified feature columns."""
        numeric_features = self.config['feature_columns']['numeric']
        categorical_features = self.config['feature_columns']['categorical']
        print("Setting up feature processor:")
        print("  Numeric:", numeric_features)
        print("  Categorical:", categorical_features)
        self.feature_processor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ]
        )

    def train_performance_model(self):
        """Train an XGBoost regressor to predict Total_Fantasy using grid search."""
        print("Training performance model...")
        df = self.data['merged']
        y = df['Total_Fantasy']
        feature_cols = self.config['feature_columns']['numeric'] + self.config['feature_columns']['categorical']
        print("Feature columns:")
        print(feature_cols)
        X = df[feature_cols]
        print("Training data shape:", X.shape)
        pipeline = Pipeline([
            ('preprocessor', self.feature_processor),
            ('xgbregressor', xgb.XGBRegressor(**self.xgb_params))
        ])
        tscv = TimeSeriesSplit(n_splits=5)
        param_grid = {
            'xgbregressor__n_estimators': [100, 200],
            'xgbregressor__max_depth': [3, 5],
            'xgbregressor__learning_rate': [0.01, 0.1]
        }
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            error_score='raise',
            verbose=2
        )
        grid_search.fit(X, y)
        self.model = grid_search.best_estimator_
        best_rmse = (-grid_search.best_score_) ** 0.5
        print("Model training complete.")
        print("Best parameters:", grid_search.best_params_)
        print(f"Best RMSE: {best_rmse:.2f}")

    def predict_roster_performance(self):
        """Merge the roster with the merged data and predict each player's performance."""
        print("Predicting roster performance...")
        roster = self.data['roster']
        merged = self.data['merged']
        roster_merged = roster.merge(merged, on='Player', how='left')
        feature_cols = self.config['feature_columns']['numeric'] + self.config['feature_columns']['categorical']
        X_roster = roster_merged[feature_cols]
        roster_merged['Predicted_Points'] = self.model.predict(X_roster)
        self.data['roster_merged'] = roster_merged
        print("Roster performance predicted. Shape:", roster_merged.shape)

    def create_optimization_model(self):
        """Build and solve an optimization model to select the best playing 11."""
        print("Creating optimization model for team selection...")
        roster = self.data['roster_merged']
        players = roster['Player'].unique().tolist()
        player_vars = pulp.LpVariable.dicts("Player", players, cat='Binary')
        prob = pulp.LpProblem("BestPlaying11", pulp.LpMaximize)
        prob += pulp.lpSum([roster.loc[roster['Player'] == p, 'Predicted_Points'].values[0] * player_vars[p]
                            for p in players])
        prob += pulp.lpSum([player_vars[p] for p in players]) == 11

        self.optimization_problem = prob
        print("Optimization model created. Solving...")
        prob.solve(pulp.PULP_CBC_CMD(msg=True))
        selected = [p for p in prob.variables() if p.varValue == 1]
        selected_players = [p.name.replace("Player_", "") for p in selected]
        print("Selected players:", selected_players)
        return selected_players

    def run(self):
        """Execute the entire pipeline to produce the best playing 11."""
        self.load_raw_data()
        self.compute_fantasy_points()
        self.merge_data()
        self.setup_feature_pipeline()
        self.train_performance_model()
        self.predict_roster_performance()
        best_team = self.create_optimization_model()
        print("\nOptimal Playing 11:")
        for player in best_team:
            print(f"- {player}")

with open("config.yaml") as stream:
    try:
        config = yaml.safe_load(stream)
        config = config['model']['xgboost']
    except yaml.YAMLError as exc:
        print(exc)
        exit(1)

if __name__ == "__main__":
    optimizer = BestPlaying11Optimizer(config)
    optimizer.run()
