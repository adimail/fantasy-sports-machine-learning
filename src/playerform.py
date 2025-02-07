# Updating th eplayer form
#
# Author: Aditya Godse
#
# This script cleans the player stats data, updates the players from the YAML file,
# filters only players that exist in the squad list, and calculates the recent form
# scores for batting, bowling, and fielding based on the recent matches.
#
# =================================================================================

import sys
import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm
from colorama import Fore, init

init(autoreset=True)


class PlayerForm:
    def __init__(self,
                 bowling_file="output/bowling_data.csv",
                 batting_file="output/batting_data.csv",
                 fielding_file="output/fielding_data.csv",
                 config_file="config.yaml",
                 previous_months=500,
                 decay_rate=0.1):
        """
        Initializes the PlayerForm with file paths and parameters.

        Parameters:
            bowling_file (str): Path to the bowling CSV file.
            batting_file (str): Path to the batting CSV file.
            fielding_file (str): Path to the fielding CSV file.
            config_file (str): Path to the YAML config file containing squad info.
            previous_months (int): Time window in months for recent matches.
            decay_rate (float): Decay rate for weighting recent match performances.
        """
        self.bowling_file = bowling_file
        self.batting_file = batting_file
        self.fielding_file = fielding_file
        self.config_file = config_file
        self.previous_months = previous_months
        self.decay_rate = decay_rate
        self.key_cols = ['Player', 'Team', 'Start Date', 'End Date', "Mat"]

    def load_data(self):
        """
        Loads and merges the bowling, batting, and fielding CSV data.

        Returns:
            pd.DataFrame: The merged DataFrame with cleaned and renamed columns.
        """
        try:
            bowling = pd.read_csv(self.bowling_file)
            batting = pd.read_csv(self.batting_file)
            fielding = pd.read_csv(self.fielding_file)
        except Exception as e:
            print(Fore.RED + f"Error reading CSV files: {e}")
            sys.exit(1)

        # Drop columns with all missing values.
        bowling = bowling.dropna(axis=1, how="all")
        batting = batting.dropna(axis=1, how="all")
        fielding = fielding.dropna(axis=1, how="all")

        # Rename columns for each dataset (except for key columns).
        bowling_renamed = bowling.rename(
            columns=lambda x: f"bowl {x}".lower() if x not in self.key_cols else x)
        batting_renamed = batting.rename(
            columns=lambda x: f"bat {x}".lower() if x not in self.key_cols else x)
        fielding_renamed = fielding.rename(
            columns=lambda x: f"field {x}".lower() if x not in self.key_cols else x)

        # Merge DataFrames on key columns using outer joins.
        df = bowling_renamed.merge(batting_renamed, on=self.key_cols, how='outer')
        df = df.merge(fielding_renamed, on=self.key_cols, how='outer')

        # Convert date columns to datetime objects.
        try:
            df['Start Date'] = pd.to_datetime(df['Start Date'])
            df['End Date'] = pd.to_datetime(df['End Date'])
        except Exception as e:
            print(Fore.RED + f"Error converting date columns: {e}")
            sys.exit(1)

        return df

    def filter_players_by_squad(self, df):
        """
        Filters the DataFrame to retain only rows for players present in the squad
        list defined in the YAML configuration file. It also reports players from the
        YAML file that are missing in the DataFrame.

        Parameters:
            df (pd.DataFrame): The input DataFrame containing player data.

        Returns:
            pd.DataFrame: The filtered DataFrame containing only valid players.
        """
        try:
            with open(self.config_file, 'r') as stream:
                data = yaml.safe_load(stream)
        except Exception as e:
            print(Fore.RED + f"Error reading YAML config file: {e}")
            sys.exit(1)

        player_to_team = {}
        valid_players = []
        squad = data.get('squad', {})

        for team, players in squad.items():
            if players:
                for player in players:
                    valid_players.append(player)
                    player_to_team[player] = team

        filtered_df = df[df['Player'].isin(valid_players)].copy()
        yaml_players = set(valid_players)
        df_players = set(filtered_df['Player'])
        missing_players = yaml_players - df_players

        if missing_players:
            print(Fore.YELLOW + "Missing players from data:")
            for player in sorted(missing_players):
                team = player_to_team.get(player, "Unknown Team")
                print(f"- {player} ({team})")
        else:
            print(Fore.GREEN + "All players from the YAML file are present in the DataFrame.")

        print(f"\nExtracted players: {len(df_players)} / {len(yaml_players)}")
        print(f"Missing players: {len(missing_players)}")

        return filtered_df

    def calculate_form(self, player_df):
        """
        Calculates recent form scores for batting, bowling, and fielding for each player
        based on their matches in the past `previous_months` months using exponential decay.
        This updated version gives additional weight to runs and wickets and also factors in
        the number of fours and sixes for batting, along with more robust metrics for all disciplines.

        The metrics are normalized against expected benchmark values (which can be tuned)
        and then aggregated with weighted contributions. This heuristic approach attempts
        to reflect the player's recent performance in batting, bowling, and fielding.

        Parameters:
            player_df (pd.DataFrame): DataFrame containing player match performance.

        Returns:
            pd.DataFrame: Aggregated form scores per player.
        """
        # Ensure End Date is in datetime format and filter by the cutoff date.
        player_df['End Date'] = pd.to_datetime(player_df['End Date'])
        cutoff_date = pd.to_datetime('today') - pd.DateOffset(months=self.previous_months)
        recent_data = player_df[player_df['End Date'] >= cutoff_date].copy()

        # Sort matches for each player by End Date descending (most recent first) and assign match indices.
        recent_data.sort_values(by=['Player', 'End Date'], ascending=[True, False], inplace=True)
        recent_data['match_index'] = recent_data.groupby('Player').cumcount()

        # Compute exponential decay weights (more weight to recent matches).
        recent_data['weight'] = np.exp(-self.decay_rate * recent_data['match_index'])

        # Benchmark values for normalization (these can be adjusted based on the format and level of play).
        expected_runs = 50.0          # Typical runs per innings.
        expected_avg = 50.0           # Typical batting average.
        expected_strike_rate = 100.0    # Typical strike rate.
        expected_boundaries = 5.0      # Expected combined fours and sixes.
        expected_batting_std = 20.0    # Benchmark for variability in runs.

        expected_wickets = 3.0         # Expected wickets per match.
        expected_economy = 6.0         # Typical economy rate.
        expected_bowl_avg = 30.0       # Expected bowling average.
        expected_bowling_std = 1.0     # Benchmark for variability in wickets.

        expected_fielding = 3.0        # Expected fielding contributions (catches, stumpings, run-outs).
        expected_fielding_std = 1.0    # Benchmark for fielding consistency.

        player_form_list = []
        # Process each player's recent matches with a progress bar.
        grouped = list(recent_data.groupby('Player'))
        for player, group in tqdm(grouped, desc="Calculating form scores", unit="player"):
            total_weight = group['weight'].sum()

            # ------------------
            # Batting Metrics
            # ------------------
            # Weighted aggregates
            weighted_runs = np.average(group['bat runs'].fillna(0), weights=group['weight']) if total_weight > 0 else 0
            weighted_bf = np.average(group['bat bf'].fillna(0), weights=group['weight']) if total_weight > 0 else 0
            # Batting average (if available)
            if group['bat ave'].notna().any():
                batting_average = np.average(
                    group['bat ave'].dropna(),
                    weights=group.loc[group['bat ave'].notna(), 'weight']
                )
            else:
                batting_average = 0
            # Strike rate derived from weighted runs and balls faced.
            strike_rate = (weighted_runs / weighted_bf * 100) if weighted_bf > 0 else 0
            # Boundaries: consider fours and sixes if columns exist, else default to 0.
            weighted_fours = np.average(group['bat 4s'].fillna(0), weights=group['weight']) if 'bat 4s' in group.columns else 0
            weighted_sixes = np.average(group['bat 6s'].fillna(0), weights=group['weight']) if 'bat 6s' in group.columns else 0
            # Consistency: lower standard deviation is better.
            batting_std = group['bat runs'].std(skipna=True) if len(group) > 1 else 0

            # Normalize metrics to a 0-100 scale.
            norm_runs = min((weighted_runs / expected_runs) * 100, 100)
            norm_avg = min((batting_average / expected_avg) * 100, 100)
            norm_sr = min((strike_rate / expected_strike_rate) * 100, 100)
            norm_boundaries = min(((weighted_fours + weighted_sixes) / expected_boundaries) * 100, 100)
            norm_consistency = max(0, 1 - (batting_std / expected_batting_std)) * 100

            # Aggregate batting form score with higher weight to runs.
            batting_form_score = (
                0.4 * norm_runs +
                0.2 * norm_avg +
                0.2 * norm_sr +
                0.1 * norm_boundaries +
                0.1 * norm_consistency
            )
            batting_form_score = np.clip(batting_form_score, 0, 100)

            # ------------------
            # Bowling Metrics
            # ------------------
            weighted_wickets = np.average(group['bowl wkts'].fillna(0), weights=group['weight']) if total_weight > 0 else 0
            weighted_bowl_runs = np.average(group['bowl runs'].fillna(0), weights=group['weight']) if total_weight > 0 else 0
            weighted_overs = np.average(group['bowl overs'].fillna(0), weights=group['weight']) if total_weight > 0 else 0
            if group['bowl ave'].notna().any():
                bowling_average = np.average(
                    group['bowl ave'].dropna(),
                    weights=group.loc[group['bowl ave'].notna(), 'weight']
                )
            else:
                bowling_average = 0
            economy_rate = (weighted_bowl_runs / weighted_overs) if weighted_overs > 0 else 0
            bowling_std = group['bowl wkts'].std(skipna=True) if len(group) > 1 else 0

            norm_wickets = min((weighted_wickets / expected_wickets) * 100, 100)
            norm_economy = max(0, 1 - (economy_rate / expected_economy)) * 100
            norm_bowl_avg = max(0, 1 - (bowling_average / expected_bowl_avg)) * 100
            norm_bowling_consistency = max(0, 1 - (bowling_std / expected_bowling_std)) * 100

            # Emphasize wickets more in the bowling score.
            bowling_form_score = (
                0.5 * norm_wickets +
                0.2 * norm_economy +
                0.2 * norm_bowl_avg +
                0.1 * norm_bowling_consistency
            )
            bowling_form_score = np.clip(bowling_form_score, 0, 100)

            # ------------------
            # Fielding Metrics
            # ------------------
            # Sum fielding contributions from catches, stumpings, and run-outs.
            fielding_contrib = group[['field ct', 'field st', 'field ct wk']].fillna(0).sum(axis=1)
            fielding_average = np.average(fielding_contrib, weights=group['weight']) if total_weight > 0 else 0
            fielding_std = fielding_contrib.std() if len(fielding_contrib) > 1 else 0

            norm_fielding = min((fielding_average / expected_fielding) * 100, 100)
            norm_fielding_consistency = max(0, 1 - (fielding_std / expected_fielding_std)) * 100

            fielding_form_score = (0.7 * norm_fielding + 0.3 * norm_fielding_consistency)
            fielding_form_score = np.clip(fielding_form_score, 0, 100)

            player_form_list.append({
                'Player': player,
                'Batting Form': batting_form_score,
                'Bowling Form': bowling_form_score,
                'Fielding Form': fielding_form_score
            })

        return pd.DataFrame(player_form_list)

    def run(self):
        """
        Executes the full data preprocessing workflow:
          1. Load and merge CSV data.
          2. Filter players based on the YAML squad.
          3. Calculate recent form scores for each player.
        """
        print(Fore.CYAN + "Starting data preprocessing...")
        df = self.load_data()
        filtered_df = self.filter_players_by_squad(df)
        form_scores = self.calculate_form(filtered_df)
        print(Fore.GREEN + "Form scores calculated successfully:")
        form_scores.to_csv("output/recent_player_form.csv", index=False)


def UpdatePlayerForm():
    try:
        preprocessor = PlayerForm()
        preprocessor.run()
    except Exception as e:
        print(Fore.RED + f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        UpdatePlayerForm()
    except KeyboardInterrupt:
        sys.exit(1)
