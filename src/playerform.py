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

        Parameters:
            player_df (pd.DataFrame): DataFrame containing player match performance.

        Returns:
            pd.DataFrame: Aggregated form scores per player.
        """
        player_df['End Date'] = pd.to_datetime(player_df['End Date'])
        cutoff_date = pd.to_datetime('today') - pd.DateOffset(months=self.previous_months)
        recent_data = player_df[player_df['End Date'] >= cutoff_date].copy()

        # Sort matches for each player by End Date descending (most recent first).
        recent_data.sort_values(by=['Player', 'End Date'], ascending=[True, False], inplace=True)
        recent_data['match_index'] = recent_data.groupby('Player').cumcount()

        # Compute exponential decay weights (more weight to recent matches).
        recent_data['weight'] = np.exp(-self.decay_rate * recent_data['match_index'])

        player_form_list = []
        # Convert groupby object to list for tqdm progress display.
        grouped = list(recent_data.groupby('Player'))
        for player, group in tqdm(grouped, desc="Calculating form scores", unit="player"):
            # -----------
            # Batting Metrics
            # -----------
            if group['bat ave'].notna().any():
                batting_average = np.average(
                    group['bat ave'].dropna(),
                    weights=group.loc[group['bat ave'].notna(), 'weight']
                )
            else:
                batting_average = 0

            weighted_runs = (np.average(group['bat runs'].fillna(0), weights=group['weight'])
                             if group['weight'].sum() > 0 else 0)
            weighted_bf = (np.average(group['bat bf'].fillna(0), weights=group['weight'])
                           if group['weight'].sum() > 0 else 0)
            strike_rate = (weighted_runs / weighted_bf * 100) if weighted_bf > 0 else 0

            if len(group) > 1:
                batting_std = group['bat runs'].std(skipna=True)
            else:
                batting_std = 0

            batting_form_score = (0.5 * batting_average) + (0.3 * strike_rate) + (0.2 * (1 - batting_std))
            batting_form_score = np.clip(batting_form_score, 0, 100)

            # -----------
            # Bowling Metrics
            # -----------
            if group['bowl ave'].notna().any():
                bowling_average = np.average(
                    group['bowl ave'].dropna(),
                    weights=group.loc[group['bowl ave'].notna(), 'weight']
                )
            else:
                bowling_average = 0

            total_wickets = (np.average(group['bowl wkts'].fillna(0), weights=group['weight'])
                             if group['weight'].sum() > 0 else 0)
            total_bowl_runs = (np.average(group['bowl runs'].fillna(0), weights=group['weight'])
                               if group['weight'].sum() > 0 else 0)
            total_overs = (np.average(group['bowl overs'].fillna(0), weights=group['weight'])
                           if group['weight'].sum() > 0 else 0)
            economy_rate = (total_bowl_runs / total_overs) if total_overs > 0 else 0

            if len(group) > 1:
                bowling_std = group['bowl wkts'].std(skipna=True)
            else:
                bowling_std = 0

            bowling_form_score = (0.5 * bowling_average) + (0.3 * (1 - economy_rate)) + (0.2 * (1 - bowling_std))
            bowling_form_score = np.clip(bowling_form_score, 0, 100)

            # -----------
            # Fielding Metrics
            # -----------
            fielding_contrib = group[['field ct', 'field st', 'field ct wk']].fillna(0).sum(axis=1)
            fielding_average = (np.average(fielding_contrib, weights=group['weight'])
                                if group['weight'].sum() > 0 else 0)

            if len(fielding_contrib) > 1:
                fielding_std = fielding_contrib.std()
            else:
                fielding_std = 0

            fielding_form_score = (0.6 * fielding_average) + (0.4 * (1 - fielding_std))
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
