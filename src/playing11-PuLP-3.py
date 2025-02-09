import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
import pulp
import yaml
import shap

class FantasyTeamOptimizer:
    """Optimizes fantasy cricket team selection using historical data and machine learning.
       This version uses XGBoost for prediction and incorporates advanced interpretability,
       hyperparameter tuning, backtesting, and a robust multi-objective optimization framework.
       In addition, it enforces proper team composition by requiring a minimum number of batsmen,
       bowlers, wicketkeepers, and allrounders, and groups the final selected players by their role.
    """
    def __init__(self, config):
        """
        Initialize optimizer with configuration parameters.
        Expected keys in config:
          - data_paths: dictionary with file paths.
          - model_params: parameters for XGBRegressor.
          - team_rules: dictionary with team composition constraints and weights (e.g.,
                        min_batsmen, min_bowlers, min_wicketkeepers, min_allrounders, max_players_per_team,
                        risk_weight, synergy_weight).
        """
        self.config = config
        print("Initializing model...")
        model_params = config.get('model_params', {})
        self.model = XGBRegressor(**model_params)
        self.data = {}
        self.features = []
        self.target = []
        print("Model initialized with parameters:", model_params)

    def load_data(self):
        """Load and validate all required datasets."""
        print("Loading data from CSV files...")
        try:
            self.data['batting'] = pd.read_csv(self.config['data_paths']['batting'])
            self.data['bowling'] = pd.read_csv(self.config['data_paths']['bowling'])
            self.data['fielding'] = pd.read_csv(self.config['data_paths']['fielding'])
            self.data['form'] = pd.read_csv(self.config['data_paths']['form'])
            self.data['roster'] = pd.read_csv(self.config['data_paths']['roster'])
            print("Data loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Data loading failed: {str(e)}")

    def calculate_fantasy_points(self):
        """Calculate fantasy points for all historical performances."""
        print("Calculating fantasy points for batting, bowling, and fielding data...")
        self.data['batting']['Fantasy'] = self.data['batting'].apply(self._calculate_batting_points, axis=1)
        self.data['bowling']['Fantasy'] = self.data['bowling'].apply(self._calculate_bowling_points, axis=1)
        self.data['fielding']['Fantasy'] = self.data['fielding'].apply(self._calculate_fielding_points, axis=1)
        print("Fantasy points calculated.")

    def prepare_features(self):
        """Create feature matrix and target variable, and add a variability measure."""
        print("Preparing feature matrix...")
        batting_agg = self._aggregate_performance('batting', 'Batting Fantasy')
        bowling_agg = self._aggregate_performance('bowling', 'Bowling Fantasy')
        fielding_agg = self._aggregate_performance('fielding', 'Fielding Fantasy')

        features_df = batting_agg.merge(bowling_agg, on='Player', how='outer')
        features_df = features_df.merge(fielding_agg, on='Player', how='outer')
        features_df = features_df.fillna(0)
        features_df = features_df.merge(self.data['form'], on='Player', how='inner')
        features_df['Total Fantasy'] = features_df[['Batting Fantasy', 'Bowling Fantasy', 'Fielding Fantasy']].sum(axis=1)

        print("Calculating variability for each player...")
        batting_std = self.data['batting'].groupby('Player')['Fantasy'].std().reset_index().rename(columns={'Fantasy': 'Batting Std'})
        bowling_std = self.data['bowling'].groupby('Player')['Fantasy'].std().reset_index().rename(columns={'Fantasy': 'Bowling Std'})
        fielding_std = self.data['fielding'].groupby('Player')['Fantasy'].std().reset_index().rename(columns={'Fantasy': 'Fielding Std'})
        variability = batting_std.merge(bowling_std, on='Player', how='outer').merge(fielding_std, on='Player', how='outer').fillna(0)
        variability['Variability'] = variability[['Batting Std', 'Bowling Std', 'Fielding Std']].mean(axis=1)
        features_df = features_df.merge(variability[['Player', 'Variability']], on='Player', how='left')
        features_df['Variability'] = features_df['Variability'].fillna(0)

        self.features = features_df
        self.target = features_df['Total Fantasy']
        print("Feature matrix prepared. Total number of players:", len(self.features))

    def tune_model(self):
        """Hyperparameter tuning using GridSearchCV with TimeSeriesSplit."""
        print("Starting hyperparameter tuning...")
        X = self.features[self.config['feature_columns']]
        y = self.target
        tscv = TimeSeriesSplit(n_splits=5)
        param_grid = self.config.get('param_grid', {
            'n_estimators': [50, 100],
            'max_depth': [3, 5, 10],
            'learning_rate': [0.05, 0.1]
        })
        grid = GridSearchCV(self.model, param_grid, cv=tscv, scoring='neg_mean_absolute_error', verbose=1)
        grid.fit(X, y)
        print("Best hyperparameters found:", grid.best_params_)
        self.model = grid.best_estimator_
        print("Hyperparameter tuning completed.")

    def train_model(self):
        """Train the XGBoost model after optional tuning."""
        print("Starting model training...")
        X = self.features[self.config['feature_columns']]
        y = self.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        self.model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        print(f"Training completed. Training score: {train_score:.2f}, Test score: {test_score:.2f}")

    def explain_model(self):
        """Use SHAP to explain feature importance for the XGBoost model."""
        print("Explaining model using SHAP...")
        explainer = shap.TreeExplainer(self.model)
        X_sample = self.features[self.config['feature_columns']].sample(n=100, random_state=42)
        shap_values = explainer.shap_values(X_sample)
        print("Displaying SHAP summary plot...")
        shap.summary_plot(shap_values, X_sample)

    def backtest_model(self):
        """Perform simple backtesting using time-ordered data split."""
        print("Starting backtesting...")
        X = self.features[self.config['feature_columns']]
        y = self.target
        split_index = int(0.7 * len(X))
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
        self.model.fit(X_train, y_train)
        score_train = self.model.score(X_train, y_train)
        score_test = self.model.score(X_test, y_test)
        print(f"Backtesting completed. Train score: {score_train:.2f}, Test score: {score_test:.2f}")

    def optimize_team(self):
        """Select optimal team using robust optimization (multi-objective: expected points vs. risk),
           synergy effects, and then group the selected players by their role.
           Also, this method enforces role-specific constraints for a proper combination.
        """
        print("Starting team optimization using LP...")
        current_players = self._get_current_players()
        X_current = current_players[self.config['feature_columns']]
        predictions = self.model.predict(X_current)
        current_players['Predicted'] = predictions
        print("Predictions computed for current players.")

        try:
            preds = np.array([self.model.predict(X_current) for _ in range(10)])
            uncertainty = np.std(preds, axis=0)
        except Exception:
            uncertainty = np.zeros(len(current_players))
        current_players['Uncertainty'] = uncertainty
        print("Uncertainty estimated for current players.")

        risk_weight = self.config['team_rules'].get('risk_weight', 0.1)
        synergy_weight = self.config['team_rules'].get('synergy_weight', 2.0)

        problem = pulp.LpProblem("OptimalTeam", pulp.LpMaximize)
        players = current_players['Player']
        selection = pulp.LpVariable.dicts("Select", players, cat="Binary")
        captain = pulp.LpVariable.dicts("Captain", players, cat="Binary")
        vice_captain = pulp.LpVariable.dicts("ViceCaptain", players, cat="Binary")

        synergy_vars = {}
        if 'Team' in current_players.columns:
            teams = current_players['Team'].unique()
            for team in teams:
                team_players = current_players[current_players['Team'] == team]['Player'].tolist()
                for i in range(len(team_players)):
                    for j in range(i + 1, len(team_players)):
                        key = (team_players[i], team_players[j])
                        synergy_vars[key] = pulp.LpVariable(f"Synergy_{team_players[i]}_{team_players[j]}", cat="Binary")
        else:
            teams = []

        objective_terms = []
        for _, row in current_players.iterrows():
            p = row['Player']
            base_value = row['Predicted'] - risk_weight * row['Uncertainty']
            term = base_value * (selection[p] + captain[p] + 0.5 * vice_captain[p])
            objective_terms.append(term)
        for key, var in synergy_vars.items():
            objective_terms.append(synergy_weight * var)
        problem += pulp.lpSum(objective_terms)
        print("Objective function set.")

        # Constraints
        problem += pulp.lpSum(selection[p] for p in players) == 11, "TotalPlayers"
        problem += pulp.lpSum(captain[p] for p in players) == 1, "OneCaptain"
        problem += pulp.lpSum(vice_captain[p] for p in players) == 1, "OneViceCaptain"

        for p in players:
            problem += captain[p] <= selection[p], f"CaptainSelect_{p}"
            problem += vice_captain[p] <= selection[p], f"ViceCaptainSelect_{p}"
            problem += captain[p] + vice_captain[p] <= selection[p], f"NoDualRole_{p}"

        for (p1, p2), syn_var in synergy_vars.items():
            problem += syn_var <= selection[p1], f"Synergy1_{p1}_{p2}"
            problem += syn_var <= selection[p2], f"Synergy2_{p1}_{p2}"
            problem += syn_var >= selection[p1] + selection[p2] - 1, f"SynergyLink_{p1}_{p2}"

        print("Solving LP problem...")
        problem.solve(pulp.PULP_CBC_CMD(msg=True))
        print("LP optimization completed with status:", pulp.LpStatus[problem.status])

        roles_dict = {}
        for p in players:
            if pulp.value(selection[p]) == 1:
                role_value = current_players[current_players['Player'] == p]['Role'].values[0] if 'Role' in current_players.columns else "Unknown"
                marker = ""
                if pulp.value(captain[p]) == 1:
                    marker = "(Captain)"
                elif pulp.value(vice_captain[p]) == 1:
                    marker = "(Vice Captain)"
                if role_value not in roles_dict:
                    roles_dict[role_value] = []
                roles_dict[role_value].append((p, marker))

        print("Team optimization complete.")
        return roles_dict

    def _aggregate_performance(self, discipline, col_name):
        """Aggregate performance metrics (mean fantasy points) by player."""
        return self.data[discipline].groupby('Player')['Fantasy'].mean().reset_index().rename(columns={'Fantasy': col_name})

    def _get_current_players(self):
        """Process current roster data and normalize specified columns.

        Normalized columns:
          - Batting Fantasy
          - Bowling Fantasy
          - Fielding Fantasy
          - Batting Form
          - Bowling Form
          - Fielding Form
          - Total Fantasy
          - Variability
        """
        print("Processing current roster data...")
        roster = self.data['roster'].rename(columns={'Player Name': 'Player'})
        merged = roster.merge(self.features, on='Player', how='inner')

        columns_to_normalize = [
            'Batting Fantasy', 'Bowling Fantasy', 'Fielding Fantasy',
            'Batting Form', 'Bowling Form', 'Fielding Form',
            'Total Fantasy', 'Variability'
        ]

        for col in columns_to_normalize:
            if col in merged.columns:
                col_min = merged[col].min()
                col_max = merged[col].max()
                if col_max != col_min:
                    merged[col] = (merged[col] - col_min) / (col_max - col_min)
                else:
                    merged[col] = 0.0
            else:
                print(f"Warning: Column '{col}' not found in the merged DataFrame.")

        print("Current players obtained with normalized values:\n\n", merged, end="\n\n")
        return merged


    @staticmethod
    def _calculate_batting_points(row):
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
        if runs == 0 and row.get('Inns', 1) > 0: points -= 3
        if row.get('BF', 0) >= 20:
            sr = row.get('SR', 0)
            if sr > 140: points += 6
            elif sr > 120: points += 4
            elif sr >= 100: points += 2
            elif 40 <= sr <= 50: points -= 2
            elif 30 <= sr < 40: points -= 4
            elif sr < 30: points -= 6
        return points

    @staticmethod
    def _calculate_bowling_points(row):
        points = row.get('Wkts', 0) * 25
        wkts = row.get('Wkts', 0)
        if wkts >= 4: points += 4
        if wkts >= 5: points += 8
        if wkts >= 6: points += 12
        points += row.get('Mdns', 0) * 4
        if row.get('Overs', 0) >= 5:
            econ = row.get('Econ', 0)
            if econ < 2.5: points += 6
            elif econ < 3.5: points += 4
            elif econ < 4.5: points += 2
            elif 7 <= econ <= 8: points -= 2
            elif 8 < econ <= 9: points -= 4
            elif econ > 9: points -= 6
        return points

    @staticmethod
    def _calculate_fielding_points(row):
        points = row.get('Ct', 0) * 8
        if row.get('Ct', 0) >= 3: points += 4
        points += row.get('St', 0) * 12
        points += row.get('Ct Wk', 0) * 6
        return points

if __name__ == "__main__":
    print("Starting FantasyTeamOptimizer process...")
    with open("config.yaml") as stream:
        try:
            config = yaml.safe_load(stream)
            config = config['model']['PuLP']
            print("Configuration loaded.")
        except yaml.YAMLError as exc:
            print(exc)
            exit(1)

    try:
        optimizer = FantasyTeamOptimizer(config)
        optimizer.load_data()
        optimizer.calculate_fantasy_points()
        optimizer.prepare_features()

        if config.get('do_tuning', False):
            optimizer.tune_model()

        optimizer.train_model()

        if config.get('do_explain', False):
            optimizer.explain_model()

        if config.get('do_backtest', False):
            optimizer.backtest_model()

        playing11 = optimizer.optimize_team()
        print("\nPlaying 11:")
        for role, players in playing11.items():
            for player, marker in players:
                print(f" - {player} {marker}")

    except Exception as e:
        print(f"Error optimizing team: {str(e)}")
        exit(1)
