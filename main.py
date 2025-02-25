import sys
import click
import os
import pandas as pd
from colorama import Fore, Style, init
from src.playerform import UpdatePlayerForm
from src.buildteam import BuildTeam
from src.geneticalgorithm import BuildTeam as BuildGeneticTeam
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
@click.option("--build", is_flag=True, help="Build the team using LP.")
@click.option(
    "--genetic", is_flag=True, help="Use genetic algorithm for team selection."
)
@click.option("--updateplayerform", is_flag=True, help="Update player form.")
@click.option("--update", type=int, help="Update player data for the last n months.")
@click.argument("option", required=False, type=int)
def main(build, genetic, updateplayerform, update, option):
    squad_file = (
        "/app/data/SquadPlayerNames.csv"
        if os.path.exists("/app/data/SquadPlayerNames.csv")
        else "data/SquadPlayerNames.csv"
    )

    if update is not None:
        try:
            print(
                f"{Fore.CYAN}Updating player data for the last {update} month(s)...{Style.RESET_ALL}"
            )
            update_player_data_main(update)
        except Exception as e:
            print(f"{Fore.RED}Error updating player data: {e}{Style.RESET_ALL}")
            sys.exit(1)

    if os.path.exists(squad_file):
        print(
            f"{Fore.GREEN}SquadPlayerNames.csv found at {squad_file}{Style.RESET_ALL}"
        )
        teams = summarize_squad_data(squad_file)
        if len(teams) >= 2:
            team_option_text = f"Build Team for {teams[0]} vs {teams[1]}"
        else:
            team_option_text = "Build Team"
    else:
        print(
            f"{Fore.YELLOW}SquadPlayerNames.csv not found at {squad_file}{Style.RESET_ALL}"
        )
        sys.exit(1)

    if build:
        try:
            if genetic:
                BuildGeneticTeam()
            else:
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
        print(f"\n{Fore.CYAN}Choose an option:")
        print(
            f"{Fore.LIGHTGREEN_EX}1. Run Full Pipeline (Update Data, Update Player Form, Build Team using LP)"
        )
        print("2. Build Team using Linear Programming")
        print("3. Build Team using Genetic Programming")
        print("4. Update Player Form")
        print("5. Update Player Data from recent month")
        option = click.prompt(f"\nEnter your choice (1-5):{Style.RESET_ALL}", type=int)

    print("\n" + "-" * 50 + "\n")

    if option == 1:
        try:
            update_player_data_main(1)
            UpdatePlayerForm()
            BuildTeam()
        except KeyboardInterrupt:
            print(f"\n{Fore.RED}Operation cancelled by user{Style.RESET_ALL}")
            sys.exit(1)
    elif option == 2:
        try:
            BuildTeam()
        except KeyboardInterrupt:
            print(f"\n{Fore.RED}Operation cancelled by user{Style.RESET_ALL}")
            sys.exit(1)
    elif option == 3:
        try:
            BuildGeneticTeam()
        except KeyboardInterrupt:
            print(f"\n{Fore.RED}Operation cancelled by user{Style.RESET_ALL}")
            sys.exit(1)
    elif option == 4:
        try:
            UpdatePlayerForm()
        except KeyboardInterrupt:
            print(f"\n{Fore.RED}Operation cancelled by user{Style.RESET_ALL}")
            sys.exit(1)
    elif option == 5:
        try:
            update_player_data_main(1)
        except KeyboardInterrupt:
            print(f"\n{Fore.RED}Operation cancelled by user{Style.RESET_ALL}")
            sys.exit(1)
    else:
        print(f"{Fore.YELLOW}Invalid option{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
