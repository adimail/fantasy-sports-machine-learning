# ==============================================================================
# Updating the Player Form
#
# Author: Aditya Godse
#
# This script cleans the player stats data, updates the players from the YAML file,
# filters only players that exist in the squad list, and calculates the recent form
# scores for batting, bowling, and fielding based on the recent matches.
#
# Columns expected:
#   Batting: Player,Mat,Inns,NO,Runs,HS,Ave,BF,SR,100,50,0,4s,6s,Team,Start Date,End Date
#   Bowling: Player,Mat,Inns,Overs,Mdns,Runs,Wkts,BBI,Ave,Econ,SR,4,5,Team,Start Date,End Date
#   Fielding: Player,Mat,Inns,Dis,Ct,St,Ct Wk,Ct Fi,MD,D/I,Team,Start Date,End Date
#
# Usage:
#   python3 -m src.playerform
#
# ==============================================================================

import sys
import os
import pandas as pd
import numpy as np
import yaml
from scipy.stats import percentileofscore
from colorama import Fore, init

init(autoreset=True)


class PlayerForm:
    def __init__(self):
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

        try:
            with open("config.yaml", "r") as stream:
                config = yaml.safe_load(stream)
        except Exception as e:
            print(Fore.RED + f"Error reading YAML config file: {e}")
            sys.exit(1)

        self.config = config

        self.bowling_file = config["data"]["bowling_file"]
        self.batting_file = config["data"]["batting_file"]
        self.fielding_file = config["data"]["fielding_file"]
        self.output_file = config["data"]["output_file"]
        self.previous_months = config["data"]["previous_months"]
        self.decay_rate = config["data"]["decay_rate"]
        self.key_cols = ["Player", "Team", "Start Date", "End Date", "Mat"]

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
            columns=lambda x: f"bowl {x}".lower() if x not in self.key_cols else x
        )
        batting_renamed = batting.rename(
            columns=lambda x: f"bat {x}".lower() if x not in self.key_cols else x
        )
        fielding_renamed = fielding.rename(
            columns=lambda x: f"field {x}".lower() if x not in self.key_cols else x
        )

        # Merge DataFrames on key columns using outer joins.
        df = bowling_renamed.merge(batting_renamed, on=self.key_cols, how="outer")
        df = df.merge(fielding_renamed, on=self.key_cols, how="outer")

        try:
            df["Start Date"] = pd.to_datetime(df["Start Date"])
            df["End Date"] = pd.to_datetime(df["End Date"])
            batting.to_csv(self.batting_file, index=False)
            bowling.to_csv(self.bowling_file, index=False)
            fielding.to_csv(self.fielding_file, index=False)
            print("Updated player files")
        except Exception as e:
            print(Fore.RED + f"Error converting date columns: {e}")
            sys.exit(1)

        return df

    def filter_players_by_squad(self, df):
        """
        Filters the DataFrame to retain only rows for players present in the squad CSV file.
        It also reports players from the squad CSV file that are missing in the DataFrame.

        The CSV file (specified by self.config["data"]["squad_file"]) is expected to have the following columns:
            Credits, Player Type, Player Name, Team, ESPN player name

        After filtering, the "Player" column in the filtered DataFrame is updated with the full name
        from the "Player Name" column of the squad CSV, and the columns "Credits" and "Player Type" are merged in.

        Parameters:
            df (pd.DataFrame): The input DataFrame containing player data.

        Returns:
            pd.DataFrame: The filtered DataFrame containing only valid players with updated names and additional columns.
        """
        try:
            squad_df = pd.read_csv(self.config["data"]["squad_file"])
        except Exception as e:
            print(Fore.RED + f"Error reading squad CSV file: {e}")
            sys.exit(1)

        valid_players = squad_df["ESPN player name"].dropna().tolist()
        player_to_team = squad_df.set_index("ESPN player name")["Team"].to_dict()
        player_abbrev_to_full = squad_df.set_index("ESPN player name")[
            "Player Name"
        ].to_dict()

        filtered_df = df[df["Player"].isin(valid_players)].copy()

        squad_players_set = set(valid_players)
        df_players_set = set(filtered_df["Player"])
        missing_players = squad_players_set - df_players_set

        if missing_players:
            print(Fore.YELLOW + "Missing players from data:")
            for player in sorted(missing_players):
                team = player_to_team.get(player, "Unknown Team")
                print(f"- {player} ({team})")
        else:
            print(
                Fore.GREEN
                + "All players from the squad CSV file are present in the DataFrame."
            )

        print(f"\nExtracted players: {len(df_players_set)} / {len(squad_players_set)}")
        print(f"Missing players: {len(missing_players)}\n")

        filtered_df = filtered_df.merge(
            squad_df[["Credits", "Player Type", "Player Name", "ESPN player name"]],
            left_on="Player",
            right_on="ESPN player name",
            how="left",
        )

        filtered_df["Player"] = filtered_df["Player Name"]

        filtered_df.drop(["ESPN player name", "Player Name"], axis=1, inplace=True)

        return filtered_df

    def calculate_form(self, player_df):
        """
        Calculates recent form scores for batting, bowling, and fielding for each player
        based on their matches in the past `previous_months` months using exponential decay weights
        and normalization based on the relative ranking (percentile) of performance among players.

        The EWMA for each metric is computed per player with an exponential decay weight.
        Then, each metric is normalized via percentile ranking (0-100). Finally, composite form scores
        are computed using format-specific weights for batting and bowling (fielding remains unchanged).

        Parameters:
            player_df (pd.DataFrame): DataFrame containing player match performance.

        Returns:
            pd.DataFrame: Aggregated form scores per player with columns 'Player', 'Batting Form',
                          'Bowling Form', 'Fielding Form', and metadata columns.
        """
        player_df["End Date"] = pd.to_datetime(player_df["End Date"])
        cutoff_date = pd.to_datetime("today") - pd.DateOffset(
            months=self.previous_months
        )
        recent_data = player_df[player_df["End Date"] >= cutoff_date].copy()
        recent_data.sort_values(
            by=["Player", "End Date"], ascending=[True, False], inplace=True
        )
        recent_data["match_index"] = recent_data.groupby("Player").cumcount()
        recent_data["weight"] = np.exp(-self.decay_rate * recent_data["match_index"])

        def compute_ewma(g, col):
            return np.average(g[col].fillna(0), weights=g["weight"])

        def normalize_series(series):
            return series.apply(lambda x: percentileofscore(series.dropna(), x))

        format_weights = {
            "T20": {
                "batting": {
                    "bat runs": 0.3,
                    "bat ave": 0.1,
                    "bat sr": 0.4,
                    "bat 4s": 0.1,
                    "bat 6s": 0.1,
                },
                "bowling": {"bowl wkts": 0.5, "bowl ave": 0.2, "bowl econ": 0.3},
            },
            "ODI": {
                "batting": {
                    "bat runs": 0.35,
                    "bat ave": 0.25,
                    "bat sr": 0.2,
                    "bat 4s": 0.1,
                    "bat 6s": 0.1,
                },
                "bowling": {"bowl wkts": 0.6, "bowl ave": 0.2, "bowl econ": 0.2},
            },
            "Test": {
                "batting": {
                    "bat runs": 0.3,
                    "bat ave": 0.4,
                    "bat sr": 0.1,
                    "bat 4s": 0.1,
                    "bat 6s": 0.1,
                },
                "bowling": {"bowl wkts": 0.5, "bowl ave": 0.3, "bowl econ": 0.2},
            },
        }

        format_type = "ODI"

        batting_weights = format_weights[format_type]["batting"]
        bowling_weights = format_weights[format_type]["bowling"]

        # ------------------
        # Batting Form
        # ------------------
        batting_metrics = {}
        for metric in ["bat runs", "bat bf", "bat sr", "bat ave", "bat 4s", "bat 6s"]:
            batting_metrics[metric] = recent_data.groupby(
                "Player", group_keys=False
            ).apply(lambda g: compute_ewma(g, metric), include_groups=False)
        batting_df = pd.DataFrame(batting_metrics).reset_index()

        batting_norm = {}
        for col in ["bat runs", "bat ave", "bat sr", "bat 4s", "bat 6s"]:
            batting_norm[col] = normalize_series(batting_df[col])

        batting_df["Batting Form"] = (
            batting_weights["bat runs"] * batting_norm["bat runs"]
            + batting_weights["bat ave"] * batting_norm["bat ave"]
            + batting_weights["bat sr"] * batting_norm["bat sr"]
            + batting_weights["bat 4s"] * batting_norm["bat 4s"]
            + batting_weights["bat 6s"] * batting_norm["bat 6s"]
        )

        # ------------------
        # Bowling Form
        # ------------------
        bowling_metrics = {}
        for metric in ["bowl wkts", "bowl runs", "bowl econ", "bowl overs", "bowl ave"]:
            bowling_metrics[metric] = recent_data.groupby(
                "Player", group_keys=False
            ).apply(lambda g: compute_ewma(g, metric), include_groups=False)
        bowling_df = pd.DataFrame(bowling_metrics).reset_index()

        bowling_norm = {}
        bowling_norm["bowl wkts"] = normalize_series(bowling_df["bowl wkts"])
        bowling_norm["bowl ave"] = 100 - normalize_series(bowling_df["bowl ave"])
        bowling_norm["bowl econ"] = 100 - normalize_series(bowling_df["bowl econ"])

        bowling_df["Bowling Form"] = (
            bowling_weights["bowl wkts"] * bowling_norm["bowl wkts"]
            + bowling_weights["bowl ave"] * bowling_norm["bowl ave"]
            + bowling_weights["bowl econ"] * bowling_norm["bowl econ"]
        )

        # ------------------
        # Fielding Form
        # ------------------
        fielding_metrics = {}
        for metric in ["field ct", "field st", "field ct wk"]:
            fielding_metrics[metric] = recent_data.groupby(
                "Player", group_keys=False
            ).apply(lambda g: compute_ewma(g, metric), include_groups=False)
        fielding_df = pd.DataFrame(fielding_metrics).reset_index()

        fielding_norm = {}
        for col in ["field ct", "field st", "field ct wk"]:
            fielding_norm[col] = normalize_series(fielding_df[col])

        fielding_df["Fielding Form"] = (
            0.5 * fielding_norm["field ct"]
            + 0.3 * fielding_norm["field st"]
            + 0.2 * fielding_norm["field ct wk"]
        )

        form_df = (
            batting_df[["Player", "Batting Form"]]
            .merge(bowling_df[["Player", "Bowling Form"]], on="Player", how="outer")
            .merge(fielding_df[["Player", "Fielding Form"]], on="Player", how="outer")
        )
        metadata_df = player_df[
            ["Player", "Credits", "Player Type", "Team"]
        ].drop_duplicates("Player")
        form_df = form_df.merge(metadata_df, on="Player", how="left")

        player_months = (
            recent_data.groupby(["Player", "Team"])["End Date"]
            .agg(lambda x: ((x.max() - x.min()).days // 30, x.max(), x.min()))
            .reset_index()
        )
        player_months.rename(columns={"End Date": "Months of Data"}, inplace=True)
        player_months[["Months of Data", "Latest Date", "Oldest Date"]] = pd.DataFrame(
            player_months["Months of Data"].tolist(), index=player_months.index
        )
        player_months = player_months.sort_values(by="Months of Data", ascending=True)
        for _, row in player_months.iterrows():
            if row["Months of Data"] < 3:
                print(
                    f"{Fore.YELLOW}{row['Months of Data'] + 1}\t"
                    f"{row['Oldest Date'].strftime('%b %y')} - "
                    f"{row['Latest Date'].strftime('%b %y')} \t"
                    f"{row['Player']} ({row['Team']})"
                )
            else:
                print(
                    f"{row['Months of Data']}\t"
                    f"{row['Oldest Date'].strftime('%b %y')} - "
                    f"{row['Latest Date'].strftime('%b %y')} \t"
                    f"{row['Player']} ({row['Team']})"
                )

        return form_df

    def run(self):
        """
        Executes the full data preprocessing workflow:
          1. Load and merge CSV data.
          2. Filter players based on the YAML squad.
          3. Calculate recent form scores for each player.
        """

        if not os.path.exists(self.output_file):
            os.makedirs(self.output_file)

        print(Fore.CYAN + "Starting data preprocessing...")
        df = self.load_data()
        filtered_df = self.filter_players_by_squad(df)
        form_scores = self.calculate_form(filtered_df)
        print(Fore.GREEN + "\n\nForm scores calculated successfully")
        form_scores.to_csv(self.output_file, index=False)


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
