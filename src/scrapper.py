# Webscrapping ESPNcricinfo Cricket Data
#
# Author: Aditya Godse
# Kaggle Notebook: https://www.kaggle.com/code/decentralized/webscrapping-espncricinfo-cricket-data
#
# This Python script scrapes cricket data from ESPNcricinfo, collecting stats
# for batting, bowling, and fielding over a user-specified range of years.
# It processes and cleans the data before saving it into CSV files. The script
# is built with a simple command-line interface, real-time progress updates,
# and clear reporting of data usage.
#
# For more details on how it works, check out the Kaggle notebook!
#
# ==============================================================================

import click
import requests
import datetime
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from tqdm import tqdm
from colorama import Fore, Style, init
import yaml
import sys
import os

init(autoreset=True)
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# This variable is a subset of the team codes found in the query menu’s
# HTML source, specifically mapping referenced cricket teams to their
# respective numeric IDs
TEAM_CODES = {
    "India": "6",
    "Pakistan": "7",
    "Bangladesh": "25",
    "New Zealand": "5",
    "England": "1",
    "Australia": "2",
    "Afghanistan": "40",
    "South Africa": "3",
}

# Base URL for web scrapping
#
# Without the User-Agent header, some websites may assume that the request
# is coming from a bot or automated scraper. This could lead to the request
# being blocked or flagged as suspicious. By passing a valid User-Agent
# (e.g., one from a common browser like Chrome)
# the website will treat the request as coming from a regular browser,
# which reduces the likelihood of blocking it.

BASE_URL = "https://stats.espncricinfo.com/ci/engine/stats/index.html"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/91.0.4472.124 Safari/537.36"
    )
}


class Scrapper:
    def __init__(self, team_codes=TEAM_CODES, base_url=BASE_URL, headers=HEADERS):
        self.team_codes = team_codes
        self.base_url = base_url
        self.headers = headers
        self.total_downloaded_bytes = 0
        self.downloaded_bytes = {
            "batting": 0,
            "bowling": 0,
            "fielding": 0,
        }

    def generate_time_spans(self, start_year, end_year):
        """Generate a list of (start_date, end_date) tuples for each month between start_year and end_year.
        For the current month, the end date is set to yesterday's date."""
        current_date = datetime.datetime.now()
        today = current_date.date()
        current_year = current_date.year
        current_month = current_date.month
        time_spans = []

        for year in range(start_year, end_year + 1):
            if year == end_year:
                if end_year == current_year:
                    months = current_month
                else:
                    months = 12
            else:
                months = 12

            for month in range(1, months + 1):
                try:
                    start_date = datetime.date(year, month, 1)
                    if year == current_year and month == current_month:
                        end_date = today - datetime.timedelta(days=1)
                    else:
                        if month == 12:
                            end_date = datetime.date(year + 1, 1, 1)
                        else:
                            end_date = datetime.date(year, month + 1, 1)
                    time_spans.append(
                        (start_date.strftime("%d+%b+%Y"), end_date.strftime("%d+%b+%Y"))
                    )
                except Exception as e:
                    print(f"Error generating dates for {year}-{month}: {e}")
        return time_spans

    def extract_player_data(self, html):
        """Extract player data from the HTML using BeautifulSoup and return a DataFrame."""
        soup = BeautifulSoup(html, "html.parser")

        def caption_match(tag_text):
            return tag_text and "overall figures" in tag_text.lower()

        target_table = None
        for table in soup.find_all("table", {"class": "engineTable"}):
            caption = table.find("caption")
            if caption and caption_match(caption.get_text(strip=True)):
                target_table = table
                break

        if not target_table:
            return None

        try:
            thead = target_table.find("thead")
            tbody = target_table.find("tbody")
            if not thead or not tbody:
                return None

            headers = [th.get_text(strip=True) for th in thead.find_all("th")]
            rows = []
            for row in tbody.find_all("tr", {"class": "data1"}):
                cells = [td.get_text(strip=True) for td in row.find_all("td")]
                rows.append(cells)

            if not rows:
                return None

            return pd.DataFrame(rows, columns=headers)
        except Exception as e:
            print(f"{Fore.RED}Error parsing table: {e}{Style.RESET_ALL}")
            return None

    def _format_bytes(self, num_bytes):
        """Format bytes into a human-readable string (B, KB, or MB)."""
        if num_bytes < 1024:
            return f"{num_bytes} B"
        elif num_bytes < 1024 * 1024:
            return f"{num_bytes / 1024:.2f} KB"
        else:
            return f"{num_bytes / (1024 * 1024):.2f} MB"

    def scrape_player_data(self, player_type, time_spans):
        """
        Scrape data for the given player type (batting, bowling, or fielding)
        across all time spans and teams. The downloaded bytes are tracked separately
        for each process.
        """
        df = pd.DataFrame()
        total_iterations = len(time_spans) * len(self.team_codes)
        progress_bar_format = (
            "{l_bar}{bar} | "
            f"{Fore.GREEN}Elapsed: "
            + "{elapsed}"
            + f"{Style.RESET_ALL} | "
            + "{postfix}"
        )

        self.downloaded_bytes[player_type] = 0

        with tqdm(
            total=total_iterations,
            desc=f"{Fore.CYAN}Scraping {player_type}{Style.RESET_ALL}",
            unit="req",
            bar_format=progress_bar_format,
            colour="magenta",
        ) as pbar:
            pbar.set_postfix_str(
                f"{Fore.YELLOW}Data Fetched: {self._format_bytes(0)}{Style.RESET_ALL}"
            )
            for start_date, end_date in time_spans:
                for team_name, team_code in self.team_codes.items():
                    try:
                        params = [
                            ("class", "2"),
                            ("filter", "advanced"),
                            (
                                "orderby",
                                (
                                    "runs"
                                    if player_type == "batting"
                                    else (
                                        "wickets" if player_type == "bowling" else "dis"
                                    )
                                ),
                            ),
                            ("spanmin1", start_date),
                            ("spanmax1", end_date),
                            ("spanval1", "span"),
                            ("team", team_code),
                            ("template", "results"),
                            ("type", player_type),
                        ]
                        response = requests.get(
                            self.base_url,
                            params=dict(params),
                            headers=self.headers,
                            timeout=15,
                        )
                        response.raise_for_status()

                        content_length = len(response.content)
                        # Update counters.
                        self.downloaded_bytes[player_type] += content_length
                        self.total_downloaded_bytes += content_length

                        data = self.extract_player_data(response.text)
                        if data is not None and not data.empty:
                            data["Team"] = team_name
                            data["Start Date"] = start_date
                            data["End Date"] = end_date
                            df = pd.concat([df, data], ignore_index=True)
                    except requests.exceptions.RequestException as re:
                        print(
                            f"{Fore.RED}Request error for {team_name} ({start_date} to {end_date}): {re}{Style.RESET_ALL}"
                        )
                    except Exception as e:
                        print(f"{Fore.RED}Unexpected error: {e}{Style.RESET_ALL}")
                    finally:
                        pbar.set_postfix_str(
                            f"{Fore.YELLOW}Data Fetched: {self._format_bytes(self.downloaded_bytes[player_type])}{Style.RESET_ALL}"
                        )
                        pbar.update(1)

        return df

    def clean_data(self, df, data_type):
        """Clean and convert the data types of the DataFrame based on data_type."""
        if df is None or df.empty:
            return df

        try:
            df.replace("-", np.nan, inplace=True)

            if data_type == "batting":
                if "HS" in df.columns:
                    df["HS"] = df["HS"].str.replace("*", "", regex=False)
                int_cols = [
                    "Mat",
                    "Inns",
                    "NO",
                    "Runs",
                    "BF",
                    "100",
                    "50",
                    "0",
                    "4s",
                    "6s",
                ]
                float_cols = ["Ave", "SR"]
            elif data_type == "bowling":
                int_cols = ["Mat", "Inns", "Mdns", "Runs", "Wkts", "4", "5"]
                float_cols = ["Ave", "Econ", "SR"]
            elif data_type == "fielding":
                int_cols = ["Mat", "Inns", "Dis", "Ct", "St", "Wk", "Fi"]
                float_cols = ["D/I"]
            else:
                return df

            for col in int_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

            for col in float_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

            date_cols = ["Start Date", "End Date"]
            for col in date_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(
                        df[col], format="%d+%b+%Y", errors="coerce"
                    )
        except Exception as e:
            print(
                f"{Fore.RED}Error cleaning data for {data_type}: {e}{Style.RESET_ALL}"
            )

        return df


@click.command()
@click.option(
    "--start_year",
    prompt="Start year",
    type=int,
    help="Starting year for data collection",
)
@click.option(
    "--end_year", prompt="End year", type=int, help="Ending year for data collection"
)
def scrapeData(start_year, end_year):
    """Main entry point: generate time spans, scrape data for all player types, and save to CSV."""
    try:
        current_year = datetime.datetime.now().year
        if start_year > end_year:
            raise ValueError("Start year cannot be after end year")
        if start_year < 1970 or end_year > current_year:
            raise ValueError(f"Years must be between 1970 and {current_year}")

        print(
            f"\n{Fore.GREEN}Generating time spans from {start_year} to {end_year}...{Style.RESET_ALL}"
        )
        scrapper = Scrapper()
        time_spans = scrapper.generate_time_spans(start_year, end_year)

        print(time_spans)

        print(f"{Fore.GREEN}Starting scraping process...{Style.RESET_ALL}")

        output_dir = config["data"]["web_scrapper_output"]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        process_totals = {}
        data_frames = {}

        for data_type in ["batting", "bowling", "fielding"]:
            print(
                f"\n{Fore.MAGENTA}=== Processing {data_type} data ==={Style.RESET_ALL}"
            )
            df = scrapper.scrape_player_data(data_type, time_spans)
            process_total = scrapper._format_bytes(scrapper.downloaded_bytes[data_type])
            process_totals[data_type] = process_total

            if df is not None and not df.empty:
                df = scrapper.clean_data(df, data_type)
                print(f"{Fore.GREEN}Collected {len(df)} {data_type} records")
                data_frames[data_type] = df
            else:
                print(f"{Fore.YELLOW}No {data_type} data collected{Style.RESET_ALL}")

            print(
                f"{Fore.CYAN}{data_type.capitalize()} data downloaded: {process_total}{Style.RESET_ALL}"
            )

        for data_type, df in data_frames.items():
            csv_path = os.path.join(output_dir, f"{data_type}_data.csv")
            try:
                df.to_csv(csv_path, index=False)
                print(
                    f"{Fore.GREEN}Saved {data_type} data to {csv_path}{Style.RESET_ALL}"
                )
            except Exception as e:
                print(f"{Fore.RED}Error saving {data_type} data: {e}{Style.RESET_ALL}")

        overall_total = scrapper._format_bytes(scrapper.total_downloaded_bytes)
        print(f"\n{Fore.CYAN}Scraping completed successfully!{Style.RESET_ALL}")
        print(
            f"Total data downloaded overall: {Fore.YELLOW}{overall_total}{Style.RESET_ALL}"
        )

        for key, value in process_totals.items():
            print(
                f" - {key.capitalize()} data downloaded: {Fore.YELLOW}{value}{Style.RESET_ALL}"
            )
    except KeyboardInterrupt:
        print(f"\n{Fore.RED}Operation cancelled by user.{Style.RESET_ALL}")
        sys.exit(1)
    except Exception as e:
        print(f"{Fore.RED}Fatal error: {e}{Style.RESET_ALL}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        scrapeData()
    except KeyboardInterrupt:
        print(f"\n{Fore.RED}Operation cancelled by user{Style.RESET_ALL}")
        sys.exit(1)
