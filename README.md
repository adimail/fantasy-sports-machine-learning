# Dream11 Fantasy Points System

![Script Preview](script.jpeg)

<video controls>
  <source src="demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

# How to Use

This repository contains scripts for scraping data from ESPN, pre-processing that data, and calculating recent player form. The processed data is saved in the `output` directory. After scraping, copy the contents from `output` to the `data` directory before running the model scripts.

The algorithm for calculating recent player form is located at `src/playerform.py`

> **Note:** All scripts are located in the `src` directory. Currently, there are six variations of model testing scripts (prefixed with `playing11-`). Once the best model is selected, its core will be migrated to `buildteam.py` in the project root.

## Cloning the Repository and Setting Up the Environment

Follow these steps to get started:

1. **Fork and Clone the Repository:**

   If you plan to contribute to this project, first fork the repository into your GitHub account. If you have collaborator access, you can clone the repository directly. Remember to create a new branch for your changes before submitting a pull request.

   Forked repo:

   ```bash
   git clone https://github.com/yourusername/fantasy-sports-machine-learning.git
   cd fantasy-sports-machine-learning
   ```

   Main Repo:

   ```bash
   git clone https://github.com/adimail/fantasy-sports-machine-learning.git
   cd fantasy-sports-machine-learning
   ```

2. **Create a Virtual Environment:**

   It is recommended to use a virtual environment to manage your project dependencies. Create and activate a virtual environment using the following commands:

   - **For macOS/Linux:**

     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

   - **For Windows:**

     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```

3. **Install Dependencies:**

   With the virtual environment activated, install the necessary packages using:

   ```bash
   pip install -r requirements.txt
   ```

4. **Branching and Pull Requests:**

   To keep the project organized and make the review process smoother, follow these guidelines:

   - **Create a New Branch:**
     Before starting work on a new feature or bug fix, create a new branch with a descriptive name (e.g., `feature/user-authentication` or `bugfix/fix-data-scraper`).

     ```bash
     git checkout -b your-feature-branch
     ```

   - **Commit Your Changes:**
     After making changes, commit them with a clear and concise commit message.

     ```bash
     git add .
     git commit -m "Description of your changes"
     ```

   - **Push the Branch and Open a Pull Request:**
     Push your branch to your fork or the main repository (if you have push access), then open a pull request to merge your changes into the main branch. Ensure your pull request includes a description of the changes and references any relevant issues.

     ```bash
     git push origin your-feature-branch
     ```

     Finally, navigate to the repository on GitHub and create a pull request.

## Running the Scripts

Once your environment is set up, you can run the scripts from the project root:

1. **Data Scraping and Pre-processing:**

   Execute the scraping script:

   ```bash
   python src/scrapper.py
   ```

   Then copy the processed data from `output` to `data`:

   ```bash
   cp -r output/* data/
   ```

2. **Model Execution:**

   Run the desired model script (for example, `playing11-PuLP.py`):

   ```bash
   python src/playing11-PuLP.py
   ```

> **Reminder:** The machine learning model configurations (such as the number of previous months to consider, XGBoost parameters, and data source directories) are specified in the `config.yaml` file located in the root directory.

# Scoring system

## Batting

| Action                                  | Points  | Who Receives the Points |
| --------------------------------------- | ------- | ----------------------- |
| **Run**                                 | +1 pt   | Batsman                 |
| **Boundary Bonus**                      | +4 pts  | Batsman                 |
| **Six Bonus**                           | +6 pts  | Batsman                 |
| **25 Run Bonus**                        | +4 pts  | Batsman                 |
| **50 Run Bonus**                        | +8 pts  | Batsman                 |
| **75 Run Bonus**                        | +12 pts | Batsman                 |
| **100 Run Bonus**                       | +16 pts | Batsman                 |
| **125 Run Bonus**                       | +20 pts | Batsman                 |
| **150 Run Bonus**                       | +24 pts | Batsman                 |
| **Dismissal for a duck**                | -3 pts  | Batsman                 |
| **Strike Rate** (min 20 balls)          |         |                         |
| Above 140 runs per 100 balls            | +6 pts  | Batsman                 |
| Between 120.01 - 140 runs per 100 balls | +4 pts  | Batsman                 |
| Between 100 - 120 runs per 100 balls    | +2 pts  | Batsman                 |
| Between 40 - 50 runs per 100 balls      | -2 pts  | Batsman                 |
| Between 30 - 39.99 runs per 100 balls   | -4 pts  | Batsman                 |
| Below 30 runs per 100 balls             | -6 pts  | Batsman                 |

## Bowling

| Action                                | Points  | Who Receives the Points |
| ------------------------------------- | ------- | ----------------------- |
| **Dot Ball** (Every 3 dot balls)      | +1 pt   | Bowler                  |
| **Wicket** (Excluding Run Out)        | +25 pts | Bowler                  |
| **Bonus (LBW/Bowled)**                | +8 pts  | Bowler                  |
| **4 Wicket Bonus**                    | +4 pts  | Bowler                  |
| **5 Wicket Bonus**                    | +8 pts  | Bowler                  |
| **6 Wicket Bonus**                    | +12 pts | Bowler                  |
| **Maiden Over**                       | +4 pts  | Bowler                  |
| **Economy Rate Points** (min 5 overs) |         |                         |
| Below 2.5 runs per over               | +6 pts  | Bowler                  |
| Between 2.5 - 3.49 runs per over      | +4 pts  | Bowler                  |
| Between 3.5 - 4.5 runs per over       | +2 pts  | Bowler                  |
| Between 7 - 8 runs per over           | -2 pts  | Bowler                  |
| Between 8.01 - 9 runs per over        | -4 pts  | Bowler                  |
| Above 9 runs per over                 | -6 pts  | Bowler                  |

## Fielding

| Action                         | Points  | Who Receives the Points |
| ------------------------------ | ------- | ----------------------- |
| **Catch**                      | +8 pts  | Fielder                 |
| **3 Catch Bonus**              | +4 pts  | Fielder                 |
| **Stumping**                   | +12 pts | Wicketkeeper            |
| **Run Out (Direct Hit)**       | +12 pts | Fielder                 |
| **Run Out (Not a Direct Hit)** | +6 pts  | Fielder                 |

## Additional Points

| Action                                                          | Points | Who Receives the Points |
| --------------------------------------------------------------- | ------ | ----------------------- |
| **Captain Points**                                              | 2x     | Captain                 |
| **Vice-Captain Points**                                         | 1.5x   | Vice-Captain            |
| **In Announced Lineups**                                        | +4 pts | All Players             |
| **Playing Substitute** (Concussion, X-Factor, or Impact Player) | +4 pts | Substitute Players      |

## Other Important Notes

- **Warm-up matches**: Fantasy points will follow the final scorecard, even if the chasing team bats more or stops early.
- **Batting**: No bonus points for runs once the player reaches 150 runs. Only their best knock is counted.
- **Fielding**: Players taking more than 3 catches will receive a 3 Catch Bonus (+4 pts).
- **Substitutes**: Only Concussion, X-Factor, and Impact Player substitutes who play will earn 4 additional points.

### Special Conditions:

- **Strike Rate** and **Economy Rate** points are not awarded for players in _The Hundred_.
- **Substitute Players**: If they replace an announced player, they get 0 points for being announced but earn points if they play.
