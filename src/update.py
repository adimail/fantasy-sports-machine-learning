# ==============================================================================
# Update Player Data Script
#
# Author: Aditya Godse
# Inspired by: Kaggle Notebook - https://www.kaggle.com/code/decentralized/webscrapping-espncricinfo-cricket-data
#
# Description:
# This Python script updates previously scraped cricket data from ESPNcricinfo.
# It scrapes the latest data for batting, bowling, and fielding for a specified number
# of recent months, cleans the data, and updates existing CSV files by removing rows
# corresponding to the updated time periods before merging in the new data.
#
# Instructions:
# - To update data for a given number of months, run the script with an integer argument.
#   For example, to update data for the last 1 month, run:
#       python3 -m src.update 1
#
# - When integrated with a main application (e.g., via main.py), the update routine
#   can be triggered via command-line flags, such as:
#       python3 main.py --build --update 1
#
# Credentials & Acknowledgements:
# - This script builds on the web scraping logic and methods originally shared in the
#   Kaggle Notebook linked above.
#
# ==============================================================================

import sys
import os
import datetime
import pandas as pd
from .scrapper import Scrapper, BASE_URL, HEADERS
from .playerform import UpdatePlayerForm


def update_existing_data_files(data_dir="data", update_dir="output"):
    """
    For each data type (batting, bowling, fielding), update the existing data file
    in `data_dir` using the new data file in `update_dir`. Rows with matching
    'Start Date' and 'End Date' in the existing data are removed before appending
    the new data.
    """
    data_types = ["batting", "bowling", "fielding"]

    for data_type in data_types:
        old_file = os.path.join(data_dir, f"{data_type}_data.csv")
        new_file = os.path.join(update_dir, f"{data_type}_data.csv")

        # Check if the new update file exists
        if not os.path.exists(new_file):
            print(
                f"Update file {new_file} does not exist. Skipping {data_type} update."
            )
            continue

        try:
            new_df = pd.read_csv(new_file, parse_dates=["Start Date", "End Date"])
        except Exception as e:
            print(f"Error reading {new_file}: {e}")
            continue

        if not os.path.exists(old_file):
            try:
                new_df.to_csv(old_file, index=False)
                print(
                    f"No existing {data_type} data found. Created new file at {old_file}."
                )
            except Exception as e:
                print(f"Error writing to {old_file}: {e}")
            continue

        try:
            old_df = pd.read_csv(old_file, parse_dates=["Start Date", "End Date"])
        except Exception as e:
            print(f"Error reading {old_file}: {e}")
            continue

        try:
            new_df["time_tuple"] = new_df.apply(
                lambda row: (row["Start Date"], row["End Date"]), axis=1
            )
            new_time_tuples = set(new_df["time_tuple"])
            old_df["time_tuple"] = old_df.apply(
                lambda row: (row["Start Date"], row["End Date"]), axis=1
            )
        except Exception as e:
            print(f"Error processing date columns for {data_type}: {e}")
            continue

        updated_old_df = old_df[~old_df["time_tuple"].isin(new_time_tuples)].copy()
        updated_old_df.drop(columns=["time_tuple"], inplace=True)
        new_df.drop(columns=["time_tuple"], inplace=True)

        updated_df = pd.concat([updated_old_df, new_df], ignore_index=True)
        try:
            updated_df.sort_values(by=["Start Date", "End Date"], inplace=True)
        except Exception as e:
            print(f"Error sorting updated data for {data_type}: {e}")

        try:
            updated_df.to_csv(old_file, index=False)
            print(f"Updated {data_type} data successfully in {old_file}.")
        except Exception as e:
            print(f"Error writing updated data to {old_file}: {e}")


def updatePlayerData(months_back):
    """
    Scrapes new data for the last `months_back` months.
    """
    current_date = datetime.datetime.now()
    end_year = current_date.year
    end_month = current_date.month
    start_year = end_year

    if months_back > end_month:
        start_year -= 1
        start_month = 12 + end_month - months_back + 1
    else:
        start_month = end_month - months_back + 1

    scrapper = Scrapper()
    time_spans = scrapper.generate_time_spans(start_year, end_year)
    if months_back <= len(time_spans):
        time_spans = time_spans[-months_back:]
    else:
        print(
            f"Warning: Requested {months_back} months, but only {len(time_spans)} are available."
        )
        time_spans = time_spans[-months_back:]

    print(f"Scraping data for the following time spans: {time_spans}")

    data_frames = {}
    for data_type in ["batting", "bowling", "fielding"]:
        print(f"\n=== Processing {data_type} data ===")
        try:
            df = scrapper.scrape_player_data(data_type, time_spans)
            if df is not None and not df.empty:
                df = scrapper.clean_data(df, data_type)
                print(f"Collected {len(df)} {data_type} records")
                data_frames[data_type] = df
            else:
                print(f"No {data_type} data collected")
        except Exception as e:
            print(f"Error processing {data_type} data: {e}")

    output_dir = "output"
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
    except Exception as e:
        print(f"Error creating output directory '{output_dir}': {e}")
        sys.exit(1)

    for data_type, df in data_frames.items():
        csv_path = os.path.join(output_dir, f"{data_type}_data.csv")
        try:
            df = df.replace("", pd.NA)
            df = df.dropna(axis=1, how="all")
            df.to_csv(csv_path, index=False)
            print(f"Saved {data_type} data to {csv_path}")
        except Exception as e:
            print(f"Error saving {data_type} data to {csv_path}: {e}")


def update_player_data_main(months_back):
    """
    Update player data for the last `months_back` months, update the data files,
    and update the player form.
    """
    if months_back < 1:
        raise ValueError("Months back must be a positive integer.")
    updatePlayerData(months_back)
    update_existing_data_files()
    UpdatePlayerForm()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 update.py <months_back>")
        sys.exit(1)
    try:
        months_back = int(sys.argv[1])
        update_player_data_main(months_back)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
