# Dream11 Fantasy Points System

![Script Preview](script.jpeg)

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

---

## Algorithm for calculating player form based on recent matches performance

### 1. Exponential Decay Weighting

- **What It Does:**
  Each match is assigned a weight using an exponential decay function:
  \[
  \text{weight} = e^{-\text{decay_rate} \times \text{match_index}}
  \]
  Here, the most recent match (with a match index of 0) gets full weight, and older matches receive progressively less importance.

- **Why It’s Recommended:**
  This method captures the intuition that recent performances are more indicative of a player’s current form. It’s a common statistical technique in time series analysis (often seen in exponential moving averages) that helps to dampen the noise from older data while emphasizing the most current information.

---

### 2. Batting Metrics

- **Key Components:**

  - **Weighted Runs:**
    The average runs scored per match are computed using the decay weights. This reflects the contribution of each match, with recent high scores being more influential.
  - **Batting Average:**
    If available, the batting average is also weighted by match recency. This gives an idea of consistency and overall performance.
  - **Strike Rate:**
    Derived from weighted runs and balls faced, this shows how quickly a player scores.
  - **Boundaries (4s and 6s):**
    These are separately weighted and aggregated because scoring boundaries is a crucial aspect of modern batting.
  - **Consistency:**
    Measured by the standard deviation of runs. A lower standard deviation indicates that a player is consistently performing, which is a valued trait.

- **Normalization:**
  Each metric is normalized against an expected benchmark (for example, a typical run might be set at 50, or a typical strike rate at 100). This converts the raw numbers into a standardized 0–100 scale, making it easier to compare and combine them.

- **Aggregation:**
  The final batting form score is an aggregate of these normalized metrics with predetermined weights (e.g., giving 40% weight to runs, 20% each to average and strike rate, etc.). Emphasizing runs and boundaries makes sense because they are the most direct indicators of batting impact in the game.

---

### 3. Bowling Metrics

- **Key Components:**

  - **Wickets:**
    The number of wickets taken is calculated as a weighted average. Since taking wickets is often the most decisive factor for a bowler, it is given significant weight.
  - **Bowling Average:**
    This provides insight into the runs conceded per wicket, which is normalized inversely (i.e., a lower average is better).
  - **Economy Rate:**
    Reflects how many runs a bowler concedes per over. A lower economy rate contributes positively to the form score.
  - **Consistency:**
    The variability in wickets taken is considered, with lower variability (i.e., more consistent wicket-taking) being preferable.

- **Normalization & Aggregation:**
  Benchmarks (like an expected 3 wickets per match or an economy of 6.0) are used to normalize these metrics. The final bowling form score gives extra emphasis (e.g., 50%) to wickets because they are a direct measure of impact, while economy and average receive slightly lower weights.

---

### 4. Fielding Metrics

- **Key Components:**

  - **Fielding Contributions:**
    This is the sum of catches, stumpings, and run-outs, all of which are important for a team’s defensive performance.
  - **Consistency:**
    Similar to batting and bowling, the variability (standard deviation) of these contributions is taken into account.

- **Normalization & Aggregation:**
  The fielding contributions are normalized against a benchmark value (for instance, an expected 3 contributions per match). The final fielding form score is a weighted sum that might lean more heavily on the average contributions, as consistency in fielding can be critical in close games.

---

### Why Is This Approach Recommended?

- **Holistic Evaluation:**
  By considering multiple metrics (and not just a single statistic), this method provides a well-rounded view of a player's performance in each discipline.

- **Emphasis on Recency:**
  The exponential decay weighting ensures that a player's current form is highlighted, which is especially important in sports where form can fluctuate significantly over time.

- **Normalization:**
  Converting raw numbers into a standardized scale allows different metrics (which might be measured on entirely different scales) to be compared and aggregated fairly.

- **Customizability:**
  The use of benchmark values means that this model can be tuned based on the level of play or specific team expectations, making it adaptable to different contexts.

- **Balanced Weighting:**
  By assigning higher weights to more impactful metrics (e.g., runs for batting and wickets for bowling), the model aligns well with what most analysts and coaches consider important in evaluating performance.
