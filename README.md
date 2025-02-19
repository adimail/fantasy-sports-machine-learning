# Fantasy Team Optimization System

![Script Preview](script.jpeg)

This system is composed of two primary modules:

1. **Player Form Calculation** – Computes recent performance metrics (form scores) for each player based on historical match data.
2. **Fantasy Team Optimizer** – Uses the computed form scores along with additional roster information to select an optimal team under given constraints.

Each module is described in detail below.

---

## 1. Player Form Calculation

This module is responsible for processing raw match data (batting, bowling, and fielding statistics) and producing composite performance scores (form scores) for each player. The main steps are:

### a. Data Loading and Cleaning

- **CSV Inputs:** Three CSV files (one each for batting, bowling, and fielding) are read.
- **Drop Empty Columns:** Any column that is entirely missing is dropped to ensure clean data.
- **Column Renaming:**
  - For clarity and to avoid naming conflicts during merge, columns in each dataset are prefixed with `"bat "`, `"bowl "`, or `"field "` except for key columns (`Player`, `Team`, `Start Date`, `End Date`, `Mat`).

### b. Data Merging

- **Merging on Key Columns:**
  - The three datasets are merged using outer joins on the key columns. This creates a unified dataset that combines all available statistics for each match.
- **Date Conversion:**
  - The date columns (`Start Date` and `End Date`) are converted to datetime objects, which allows filtering by recency.

### c. Filtering by Recent Matches

- **Time Window:**
  - A parameter (`previous_months`) specifies the recency window. Matches older than the cutoff (today’s date minus the number of months) are excluded.
- **Sorting and Indexing:**
  - The remaining (recent) data is sorted by `Player` and `End Date` (most recent first), and a match index is assigned to each player’s matches.

### d. Exponential Decay Weighting

- **Decay Weights:**
  - An exponential decay factor (using `decay_rate`) is applied based on the match index. This gives higher weight to recent performances compared to older ones.

### e. Calculation of Exponentially Weighted Averages (EWMA)

- **Metric-by-Metric Calculation:**
  - For each performance metric (e.g., runs scored, wickets taken, catches), an exponentially weighted moving average is calculated for each player.

### f. Normalization of Metrics

- **Normalization Method:**
  - Each EWMA is normalized to a 0–100 scale using the cumulative distribution function (CDF) of the standard normal distribution. This converts raw values into scores that reflect how a player performs relative to the global distribution.
- **Handling Different Metrics:**
  - For batting: metrics like runs, strike rate, average, and boundaries are combined with specific weights.
  - For bowling: wickets, average, and economy (with adjustments such as subtracting from 100 for metrics where a lower value is better) are combined.
  - For fielding: key fielding metrics are weighted and summed.

### g. Composite Form Score Generation

- **Final Output:**

  - The resulting DataFrame includes for each player:
    - **Player Name**
    - **Batting Form:** Composite score based on batting metrics.
    - **Bowling Form:** Composite score based on bowling metrics.
    - **Fielding Form:** Composite score based on fielding metrics.
    - **Additional Metadata:** Credits, Player Type, and Team.

- **Example Format:**

  ```
  Player,Batting Form,Bowling Form,Fielding Form,Credits,Player Type,Team
  Aaron Hardie,35.93,43.59,40.99,6.5,ALL,Australia
  Abrar Ahmed,31.81,70.61,35.62,7.0,BOWL,Pakistan
  ```

---

## 2. Fantasy Team Optimizer

This module uses the computed form scores and additional roster data to select the best possible fantasy team. The process involves the following steps:

### a. Data Loading and Preprocessing

- **Evaluation Data:**
  - Load the player form data (the output from the Player Form Calculation module).
- **Roster Data:**
  - Load an input roster CSV file that contains details such as Credits, Player Type, Player Name, Team, and whether the player is playing.
- **Filtering:**
  - Only players marked as `"PLAYING"` are considered.
- **Player Type Mapping:**
  - The roster’s player types (e.g., "All Rounder", "Batsmen", "Bowlers", "Wicket Keeper") are mapped to the abbreviated types used in the evaluation file (`ALL`, `BAT`, `BOWL`).
  - For instance, wicket keepers are treated as batsmen.

### b. Merging Evaluation and Roster Data

- **Merge Keys:**
  - The filtered roster is merged with the evaluation data on `Player` and `Player Type` to ensure that the most up-to-date form scores are used for team selection.
- **Result Check:**
  - A warning is printed if the merge results in an empty DataFrame, indicating potential mismatches in player names or types.

### c. Predicted Fantasy Points and Role Assignment

- **Predicted Score Calculation:**
  - For each player, a “Predicted” fantasy point score is determined based on their form scores:
    - **Bowlers:** Use the Bowling Form directly.
    - **Batsmen:** Use the Batting Form directly.
    - **All-Rounders:** The highest value among Batting, Bowling, and Fielding is chosen.
      - The corresponding role is assigned as `"BAT"` if batting or fielding is best (with fielding defaulting to batsman) or `"BOWL"` if bowling is best.
- **Additional ML Training:**
  - An XGBoost regression model is trained on the three form scores. Although the target is already computed from the form scores, this simulates a machine learning process, and the model is later used to update the predicted scores.

### d. Team Selection Optimization using PuLP

- **Optimization Objective:**
  - The goal is to maximize the total predicted fantasy points of the team. Additional multipliers are applied:
    - **Captain:** Selected player’s score is counted fully extra.
    - **Vice-Captain:** Selected player’s score gets a 0.5 extra multiplier.
- **Decision Variables:**
  - Binary decision variables are defined for:
    - **Selection:** Whether a player is included in the team.
    - **Captain:** Whether a player is chosen as captain.
    - **Vice-Captain:** Whether a player is chosen as vice-captain.
- **Constraints:**
  - **Total Players:** Exactly 11 players must be selected.
  - **Role Constraints:**
    - At least 3 players must have an assigned role of `"BOWL"`.
    - At least 3 players must have an assigned role of `"BAT"`.
  - **Captain/Vice-Captain:**
    - Exactly one captain and one vice-captain must be chosen.
    - The captain and vice-captain must be among the selected players.
    - A player cannot hold both roles simultaneously.
- **Solving the Problem:**
  - The PuLP linear programming problem is solved using the CBC solver.
- **Output:**

  - The final team is output with each player's predicted score, role, and designation if they are selected as captain or vice-captain.

- **Example of Output:**

  ```
  Selected Team

  93.66    BAT     Daryl Mitchell (Captain)
  80.18    BAT     Fakhar Zaman (Vice Captain)
  80.17    BAT     Will Young
  76.07    BOWL    Haris Rauf
  74.66    BAT     Babar Azam
  74.27    BOWL    Matt Henry
  72.40    BAT     Kane Williamson
  72.14    BOWL    Shaheen Afridi
  70.75    BAT     Glenn Phillips
  70.61    BOWL    Abrar Ahmed
  68.82    BAT     Tayyab Tahir
  ```

---

## Conclusion

The system operates as follows:

- **Player Form Calculation:**
  - It starts by cleaning and merging match data, calculating recent performance scores using exponential decay and normalization, and finally producing composite scores for batting, bowling, and fielding.
- **Fantasy Team Optimizer:**
  - It takes these scores, merges them with current roster information, computes predicted fantasy points (with role assignments), and then optimizes team selection using a linear programming approach with constraints on team size, roles, and leadership positions (captain and vice-captain).
