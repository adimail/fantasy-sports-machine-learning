import sys
import click
from colorama import Fore, Style, init
from src.scrapper import scrapeData
from src.playerform import UpdatePlayerForm

init(autoreset=True)

@click.command()
@click.argument('option', required=False, type=int)
def main(option):
    if option is None:
        option = click.prompt(
            f"\n{Fore.CYAN}Choose an option:\n1. Scrape ESPNcricinfo data\n2. Update player form\n3. Build Team from input.csv\n4. Get Recent Match Data\n5. Full data pipeline\n{Style.RESET_ALL}\n\nEnter your choice",
            type=int
        )

    print()
    print("-" * 50)
    print()

    if option == 1:
        try:
            scrapeData()
        except KeyboardInterrupt:
            print(f"\n{Fore.RED}Operation cancelled by user{Style.RESET_ALL}")
            sys.exit(1)

    if option == 2:
        try:
            UpdatePlayerForm()
        except KeyboardInterrupt:
            print(f"\n{Fore.RED}Operation cancelled by user{Style.RESET_ALL}")
            sys.exit(1)
    else:
        print(f"{Fore.YELLOW}Module not completed yet{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
