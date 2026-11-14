Fantasy start/sit model

Personal project: a weekly start/sit decision tool for my ESPN league. 16-team, full PPR, 4-pt passing TDs, private league. Team: Nostrildamus. Playoffs are Weeks 15–17.

Core idea
Don't build point projections from scratch. Use consensus (ESPN, maybe other sources) as the forecast of each player's average.
The edge is in the decision layer:
a distribution for each player,
a correlation structure between players,
an optimizer that maximizes P(win this head-to-head matchup), not expected points.
Head-to-head is a binary payoff, so variance and correlation with the opponent's lineup matter.
Example: a heavy favorite should hedge.
Example: starting a QB in the same game as the opponent's QB reduces the variance of the margin.
Phases
Archiver (urgent, perishable data). Python script using espn-api (cwendt94).
Snapshot each run:
ESPN projections (statSourceId=1) and injury status for all players
all league rosters and lineups, plus my matchup
scoring settings (once)
Schedule: Tue (after waivers), Thu, Sat night, Sun ~11:45 ET (after inactives), Sun ~3:55 ET (before late games).
Store timestamped snapshots in SQLite or Parquet. Schedule with launchd.
Historical research data from nflverse (nflreadpy / nfl_data_py):
weekly stats, snap counts, targets
schedules with closing spread and total
injury reports
No historical ESPN projections exist, so build a proxy projection (recent usage × Vegas implied team total).
Use the proxy only to estimate the shape of projection errors and how they correlate. Recalibrate the size of the errors against the ESPN archive.
Player distributions. Split each projection into a yardage/receptions part (roughly normal) and a TD part (Poisson-like), so TD-heavy players get the right skew. Validate out of sample with PIT histograms.
Correlation model. Factors for each game's environment, each team's passing and rushing, and a negative link between an offense and the opposing DST. Join to the player distributions with a copula. Fit on early seasons, test on later ones.
Optimizer. Enumerate legal lineups; Monte Carlo (~20k draws) of both lineups; rank by P(win).
Late swap: re-run at each lock with realized scores fixed.
Output: best lineup, P(win) gain vs the highest-projection lineup, whether the decision is close, and the main reason.
Backtest and live tracking.
Walk-forward on synthetic matchups built from real player-weeks.
Measure lift only on decisions where the model disagrees with "start the highest projection."
Log each live week: model pick, my pick, outcome. Bench actual points come from ESPN box scores.
Constraints
ESPN cookies (espn_s2, SWID) are login credentials. They live in the macOS Keychain or a gitignored .env. Never print, log, or commit them.
The ESPN API is unofficial and undocumented: keep requests polite and expect it to break.
Watch for lookahead leakage: only use data available at lineup lock.
League facts (from ESPN settings)
League 765545014, season 2026, my team ID 9. Lineup: QB, RB×2, WR×2, TE, FLEX, D/ST, K; 7 bench, 1 IR.
Lock type INDIVIDUAL_GAME, so late swap is possible. Waivers process Mon and Wed–Sun (not Tue).
The Odds API free tier: 500 credits/month (assumed to reset on the 1st). Game lines cost 2 per call (spreads + totals) and run at every slot.
Player props cost 1 credit per market per game. Each game is pulled once, at the last slot before it locks.
Markets in priority order: anytime TD, receptions, receiving yds, rushing yds, passing yds. A planner drops markets from the end so the rest of the month fits.
The archiver stops calling below 40 remaining.

Archiver (built)
`uv run ff-archive status | snapshot [--odds] [--prev-week] | reparse | tick`
launchd runs `tick` every 5 min; it snapshots only when a slot is due (see src/ff_model/archive/slots.py).
Slots: Tue 12:00 (+ last week's finals), Wed 12:00, Sat 22:00; Sun 11:45 and 15:55; international games T-90; all other kickoffs (TNF/SNF/MNF/Sat) T-60.
pmset wakes need deploy/ff-archiver.sudoers installed in /etc/sudoers.d.
Data: data/archive.db. raw_responses (gzipped JSON) is the source of truth; other tables come from archive/parse.py.
Layout: src/ff_model/{config,db,cli}.py are shared. sources/ holds the API clients (espn, odds). archive/ holds slots, runner, wakes, launchd, parse.
Later phases go in sibling packages (e.g. research/, model/, optimize/, web/).

Status
Archiver live as of 2026-11-13; Week 10 is being captured.
Next: nflverse historical pull, then player distributions.
Open: whether to add a second projection source (e.g. FantasyPros consensus).
