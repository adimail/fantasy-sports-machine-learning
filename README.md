# Dream11 Fantasy Points System

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

# Fantasy Points Calculation Algorithm for Live Matches

This algorithm calculates the fantasy points for players during a live match based on the Dream11 Fantasy Points System.

### 1. **Initialize Variables**

For each player:

- Batting points: `batting_points = 0`
- Bowling points: `bowling_points = 0`
- Fielding points: `fielding_points = 0`
- Additional points: `additional_points = 0`
- Total points: `total_points = 0`

### 2. **Batting Points Calculation**

For each batsman:

- **Runs**: Add `+1 point` for every run scored.

  - `batting_points += runs`

- **Boundary Bonus**: Add `+4 points` for each boundary.

  - `batting_points += 4 * (number of boundaries)`

- **Six Bonus**: Add `+6 points` for each six.

  - `batting_points += 6 * (number of sixes)`

- **Run Bonuses**:

  - For each milestone (25, 50, 75, 100, 125, 150 runs), add the corresponding bonus points.
  - Example:
    - If runs >= 150: `batting_points += 24` (150 Run Bonus)
    - If runs >= 125 and <150: `batting_points += 20` (125 Run Bonus)
    - Continue similarly for other milestones.

- **Dismissal for Duck**: Subtract `-3 points` if the player is dismissed for a duck.

  - If `runs == 0`, `batting_points -= 3`

- **Strike Rate** (for minimum 20 balls played):
  - Calculate the strike rate: `strike_rate = (runs / balls) * 100`
  - Add points based on the strike rate:
    - Above 140: `batting_points += 6`
    - Between 120.01 and 140: `batting_points += 4`
    - Between 100 and 120: `batting_points += 2`
    - Below 30: `batting_points -= 6`
    - Continue with other conditions as per the system.

### 3. **Bowling Points Calculation**

For each bowler:

- **Dot Balls**: Add `+1 point` for every 3 dot balls bowled.

  - `dot_balls = number_of_dot_balls // 3`
  - `bowling_points += dot_balls`

- **Wickets**: Add `+25 points` for each wicket (excluding run out).

  - `bowling_points += 25 * (number of wickets)`

- **Bonus (LBW/Bowled)**: Add `+8 points` for each LBW or bowled wicket.

  - `bowling_points += 8 * (number of LBW or bowled wickets)`

- **Wicket Bonuses**:

  - Add `+4 points` for 4 wickets, `+8 points` for 5 wickets, `+12 points` for 6 wickets.
  - Example: `if wickets >= 6: bowling_points += 12`

- **Maiden Over**: Add `+4 points` for each maiden over.

  - `bowling_points += 4 * (number of maiden overs)`

- **Economy Rate** (for minimum 5 overs bowled):
  - Calculate economy rate: `economy_rate = (runs_conceded / overs_bowled)`
  - Add points based on the economy rate:
    - Below 2.5: `bowling_points += 6`
    - Between 2.5 and 3.49: `bowling_points += 4`
    - Between 7 and 8: `bowling_points -= 2`
    - Continue with other conditions as per the system.

### 4. **Fielding Points Calculation**

For each fielder:

- **Catch**: Add `+8 points` for each catch.

  - `fielding_points += 8 * (number of catches)`

- **3 Catch Bonus**: Add `+4 points` if 3 or more catches are taken.

  - If `catches >= 3`, `fielding_points += 4`

- **Stumping**: Add `+12 points` for each stumping by a wicketkeeper.

  - `fielding_points += 12 * (number of stumpings)`

- **Run Out (Direct Hit)**: Add `+12 points` for each run out by direct hit.

  - `fielding_points += 12 * (number of direct hit run outs)`

- **Run Out (Not a Direct Hit)**: Add `+6 points` for each run out that is not a direct hit.
  - `fielding_points += 6 * (number of run outs without direct hit)`

### 5. **Additional Points**

For each player:

- **Captain Points**: Double the points for the captain.

  - `total_points *= 2` if player is captain

- **Vice-Captain Points**: Multiply the points by 1.5 for vice-captain.

  - `total_points *= 1.5` if player is vice-captain

- **In Announced Lineups**: Add `+4 points` if the player is in the announced lineups.

  - `additional_points += 4`

- **Substitute Players**: Add `+4 points` if the player is a substitute (Concussion, X-Factor, or Impact Player).
  - `additional_points += 4`

### 6. **Total Points Calculation**

Finally, sum all the points:

- `total_points = batting_points + bowling_points + fielding_points + additional_points`

### 7. **Return Total Points for Each Player**

- Output the `total_points` for each player at the end of the match.

### Pseudo-Code Example

```python
def calculate_fantasy_points(player_data):
    total_points = 0

    # Batting Points
    total_points += player_data['runs'] + (4 * player_data['boundaries']) + (6 * player_data['sixes'])
    if player_data['runs'] >= 150:
        total_points += 24
    elif player_data['runs'] >= 125:
        total_points += 20
    elif player_data['runs'] >= 100:
        total_points += 16
    elif player_data['runs'] >= 75:
        total_points += 12
    elif player_data['runs'] >= 50:
        total_points += 8
    elif player_data['runs'] >= 25:
        total_points += 4
    if player_data['runs'] == 0:
        total_points -= 3

    # Strike Rate Points
    strike_rate = (player_data['runs'] / player_data['balls']) * 100
    if strike_rate > 140:
        total_points += 6
    elif strike_rate >= 120:
        total_points += 4
    elif strike_rate >= 100:
        total_points += 2
    elif strike_rate <= 30:
        total_points -= 6

    # Bowling Points
    total_points += (25 * player_data['wickets']) + (8 * player_data['lbw_bowled'])
    if player_data['wickets'] >= 6:
        total_points += 12
    elif player_data['wickets'] >= 5:
        total_points += 8
    elif player_data['wickets'] >= 4:
        total_points += 4

    # Economy Rate Points
    economy_rate = player_data['runs_conceded'] / player_data['overs_bowled']
    if economy_rate < 2.5:
        total_points += 6
    elif economy_rate < 3.5:
        total_points += 4
    elif economy_rate < 4.5:
        total_points += 2
    elif economy_rate > 9:
        total_points -= 6

    # Fielding Points
    total_points += (8 * player_data['catches']) + (12 * player_data['stumpings']) + (12 * player_data['run_out_direct_hit']) + (6 * player_data['run_out_not_direct_hit'])
    if player_data['catches'] >= 3:
        total_points += 4

    # Additional Points
    if player_data['captain']:
        total_points *= 2
    if player_data['vice_captain']:
        total_points *= 1.5
    if player_data['in_announced_lineups']:
        total_points += 4
    if player_data['substitute']:
        total_points += 4

    return total_points
```
