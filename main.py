import sys
import click
import os
import pandas as pd
from colorama import Fore, Style, init
from src.playerform import UpdatePlayerForm
from src.buildteam import BuildTeam
from src.update import update_player_data_main

init(autoreset=True)


def summarize_squad_data(file_path):
    df = pd.read_csv(file_path)
    total_players = len(df)
    playing_players = df[df["IsPlaying"] == "PLAYING"].shape[0]
    player_types = df["Player Type"].value_counts()
    print(f"\nTotal number of players: {total_players}")
    print(f"Playing players: {playing_players}")
    print(f"\n{player_types.to_string()}{Style.RESET_ALL}")
    return df["Team"].unique()


@click.command()
@click.option("--build", is_flag=True, help="Build the team.")
@click.option("--updateplayerform", is_flag=True, help="Update player form.")
@click.option("--update", type=int, help="Update player data for the last n months.")
@click.argument("option", required=False, type=int)
def main(build, updateplayerform, update, option):
    if update is not None:
        try:
            print(
                f"{Fore.CYAN}Updating player data for the last {update} month(s)...{Style.RESET_ALL}"
            )
            update_player_data_main(update)
        except Exception as e:
            print(f"{Fore.RED}Error updating player data: {e}{Style.RESET_ALL}")
            sys.exit(1)

    if os.path.exists("Downloads"):
        files = os.listdir("Downloads")
        print(f"\n{Fore.GREEN}Number of files in the Downloads folder: {len(files)}")
        if "SquadPlayerNames.csv" in files:
            print(f"{Fore.GREEN}SquadPlayerNames.csv found.{Style.RESET_ALL}")
            teams = summarize_squad_data("Downloads/SquadPlayerNames.csv")
            if len(teams) >= 2:
                team_option_text = f"Build Team for {teams[0]} vs {teams[1]}"
            else:
                team_option_text = "Build Team"
        else:
            print(
                f"{Fore.YELLOW}SquadPlayerNames.csv not found in the Downloads folder.{Style.RESET_ALL}"
            )
            sys.exit(1)
    else:
        print(f"{Fore.YELLOW}Downloads folder not found.{Style.RESET_ALL}")
        sys.exit(1)

    if build:
        try:
            BuildTeam()
            return
        except KeyboardInterrupt:
            print(f"\n{Fore.RED}Operation cancelled by user{Style.RESET_ALL}")
            sys.exit(1)

    if updateplayerform:
        try:
            UpdatePlayerForm()
            return
        except KeyboardInterrupt:
            print(f"\n{Fore.RED}Operation cancelled by user{Style.RESET_ALL}")
            sys.exit(1)

    if option is None and not build and not updateplayerform:
        option = click.prompt(
            f"\n{Fore.CYAN}Choose an option:\n1. {team_option_text}\n2. Update player form\n\n{Style.RESET_ALL}\n\nEnter your choice",
            type=int,
        )

    print("\n" + "-" * 50 + "\n")

    if option == 1:
        try:
            BuildTeam()
        except KeyboardInterrupt:
            print(f"\n{Fore.RED}Operation cancelled by user{Style.RESET_ALL}")
            sys.exit(1)
    elif option == 2:
        try:
            UpdatePlayerForm()
        except KeyboardInterrupt:
            print(f"\n{Fore.RED}Operation cancelled by user{Style.RESET_ALL}")
            sys.exit(1)
    else:
        print(f"{Fore.YELLOW}Invalid option{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
