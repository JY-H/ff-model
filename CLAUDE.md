# Fantasy start/sit model

A weekly start/sit decision tool for my ESPN league.

- **League:** private, 16 teams, full PPR, 4-pt passing TDs

## Core idea

Don't build point projections from scratch. ESPN's projections are the forecast of each player's average. The edge is in the **decision layer**:

1. a **distribution** for each player, not just a point estimate
2. a **correlation structure** between players
3. an **optimizer** that maximizes P(win this head-to-head matchup), not expected points

Head-to-head is a win/lose payoff, so variance and correlation with the opponent's lineup matter. For example:

- A heavy favorite should hedge, choosing lower-variance options.
- Starting a QB from the same game as the opponent's QB reduces the variance of the margin.

## Rules

- **Credentials:** the ESPN cookies (`espn_s2`, `SWID`) and `ODDS_API_KEY` are secrets. They live in the gitignored `.env`. Never print, log or commit them.
- **League privacy:** league and team IDs, team names and league-mates' details stay out of the repo, including code, docs and commit messages. Read them from `.env` or the archive.
- **ESPN API:** it's unofficial and undocumented. Keep requests polite (seconds apart, stop on 401/403/429) and expect it to break.
- **No lookahead leakage:** the model may only use data that was available at lineup lock.
- **The Odds API budget:** 500 credits/month on the free tier. Never design anything that could exceed it.

## League facts (from ESPN settings)

| | |
|---|---|
| League and team IDs | in `.env` (`ESPN_LEAGUE_ID`, `ESPN_TEAM_ID`), not in the repo |
| Season | 2026 |
| Starting lineup | QB, RB×2, WR×2, TE, FLEX, D/ST, K (plus 7 bench, 1 IR) |
| Lineup lock | Per game (`INDIVIDUAL_GAME`), so late swap is possible |
| Waivers | Process Mon and Wed–Sun, **not Tuesday** |

## Plan and status

| Phase | Status |
|---|---|
| 1. Archiver | ✅ Live since 2026-11-13 |
| 2. Historical data | ✅ nflverse + ESPN projections 2019–2025 + league scoring |
| 3. Player distributions | ✅ Quantile model with context (weather, spread, injuries, OL/DB) |
| 4. Correlation model | ✅ Role-pair copula, validated on 2024–2025 sums |
| 5. Optimizer + late swap | ✅ `recommend` command and `web` page |
| 6. Backtest and live tracking | ✅ League 2026 W1–9 replay; synthetic walk-forward 2022–2025 |

### 1. Archiver ✅

Saves the data that disappears if it isn't captured at the time. launchd runs `tick` every 5 minutes, and it only takes a snapshot when a slot is due.

**Each snapshot saves:**

- ESPN projections and injury status for every player in the pool
- all 16 rosters and lineups, plus all matchups (with ESPN's own win probability)
- the league's transaction log (waiver claims, adds/drops, your own trades) and its trade feed (every team's trades), on Tuesdays
- game lines (spreads and totals) from The Odds API
- player props, for games that lock before the next slot
- **pre-game context:** a kickoff-window weather forecast for every upcoming outdoor game (Open-Meteo), stadium and roof for every game (`game_context`), and that week's nflverse injury report with practice participation, OL and DBs included (`injury_reports`)
- league scoring settings (on Tuesdays)

**Slots (ET)**, generated from the NFL schedule in `archive/slots.py`:

| Slot | When |
|---|---|
| Tue | 12:00. Also saves last week's final lineups and points (bench included) |
| Wed | 12:00, after Wednesday waivers |
| Sat | 22:00 |
| Sun early games | 11:45, after inactives |
| Sun late games | 15:55 |
| International games | 90 min before kickoff |
| Every other kickoff (Thu, Sun night, Mon, Sat) | 60 min before kickoff |

**Odds budget:**

- Game lines cost 2 credits per call (spreads + totals), at every slot.
- Player props cost 1 credit per market per game. Each game is pulled once, at the last slot before it locks.
- Prop markets in priority order: anytime TD, receptions, receiving yards, rushing yards, passing yards. A planner drops markets from the end of that list if the rest of the month can't be covered.
- No calls below 40 credits remaining. The quota is assumed to reset on the 1st.

**Live model inputs:** `model/live.py` `build_inputs(snapshot_id)` assembles each player's inputs using only what was archived at or before that snapshot. That covers projection and stat line, ESPN injury designation, opponent and kickoff, roof and forecast, spread (median across books, this week's game only), nflverse practice status, and games back after an absence. Live decisions and backtests share this code, so neither can see past the lock.

- Injury designation comes from ESPN; nflverse lags on designations but has practice participation.
- Retractable roofs are treated as closed. The stadium table is `sources/stadiums.py`; nflverse's roof labels are blank or wrong for some venues.
- A test checks that every remaining Thursday, Saturday, holiday and international slot finds its games with stadiums.

**Waking the Mac:** each tick schedules `pmset` wakes 3 minutes before upcoming slots. This needs `deploy/ff-archiver.sudoers` installed in `/etc/sudoers.d` (it is).

### 2. Historical data ✅

**ESPN projection history.** ESPN's public endpoint still serves weekly projections with full projected stat lines for **2019–2025**. 2018 returns nothing; earlier seasons return 404. This replaced the original plan of building a proxy projection: error shape, spread and correlation come straight from ESPN's own errors.

- **Verified as pre-game projections, not final scores:**
  - In 2020, projections and actuals correlate at 0.56 and match exactly in only ~1% of player-weeks.
  - Late scratches keep their projection. Tyrod Taylor, 2020 W2, still shows 16.8.
  - Finished 2026 games still show the pre-game numbers.
- **Unknown:** whether they freeze at kickoff or earlier (before inactives). Check this by comparing the archived 2026 W10 snapshots with what the API serves afterwards.
- **Don't use** the injury and status fields in these responses. They're players' *current* status, not the status that week. Use the nflverse injury reports instead.

**nflverse (2012 to now):** play-by-play (trimmed to ~27 columns), weekly player and team stats, snap counts, schedules with closing spread and total, injury reports, `ff_opportunity` (expected fantasy points), and `ff_playerids` (ID crosswalk).

- ESPN player IDs map to nflverse `gsis_id` through `ff_playerids.espn_id`. All rostered skill players and kickers match.
- D/ST joins on NFL team instead.
- **Leakage:** current-season files include games already played this week, and `schedules.spread_line` for unplayed games is a live line, not a closing one.

**League scoring** (`research/scoring.py`) applies this league's rules to both nflverse actuals and ESPN stat lines.

- Validated against ESPN: 99.6% of 2026 W1–9 actuals match exactly, and all 686 W10 projections match.
- ESPN quirks it handles: a blocked FG counts as a miss; points allowed excludes points the opponent's defense or special teams scored; yards allowed is net of sacks (nflverse stores sack yards as negative in recent seasons).

### 3. Player distributions ✅

`model/marginals.py`: one linear quantile regression per position and percentile (1st–99th), on the ESPN projection, the projection's make-up from ESPN's stat line (TD share and receptions per point; pass-TD and rushing share for QBs), and pre-game context.

- **Context factors**, each kept only if it beat ESPN out of sample (`research/factor_tests.py`, evidence in `data/factor_screen.csv`, regenerated by `screen-factors`):
  - weather (dome, wind, cold) for QB/WR/TE/K
  - the spread, for every position
  - injury designation plus final practice status, and first/second game back, for RB/WR/TE
  - OL starters out (QB/RB), opponent OL out (D/ST) and opponent DBs out (QB)
- **Tested and left out** (ESPN already prices them): opponent defense quality, points allowed by position, pace, kicker stall rate, the "bad defense means they throw more" idea, and key teammates out, and play-by-play opportunity (`research/opportunity.py`, evidence in `data/opportunity_screen.csv`): last week's points over expected, expected points vs ESPN, target-share trend, air-yard share and inside-10 share. 3 of 60 cells cleared the noise band, about what chance gives; the most consistent one (goal-line RBs miss ESPN) disappears once the projection's size is controlled for, and reverses in 2026. Teammates out is shown on the page as information only.
- **Live distributions are "ESPN + context"** (`espn_context_quantiles`): the model's shape, with the mean pinned to ESPN's projection plus the context shift. A player card always adds up: model = ESPN + listed effects.
- **Calibration:** on 2024–2025 and the 2026 holdout, about 10% of outcomes fall below the predicted 10th percentile and about 10% above the 90th. Known weak spots: kickers' upside is a little narrow, and 15+ mph wind is slightly optimistic at the low end.

### 4. Correlation model ✅

`model/correlation.py`: a Gaussian copula over each player's range position, with correlations estimated per role pair (QB1, RB1–2, WR1–3, TE1, K, D/ST), same team and across opponents, from 2019–2025 games. Examples: QB–WR1 +0.35, RB1–RB2 −0.08, QB vs opposing QB +0.20, D/ST vs opposing QB −0.57. Stable across eras. On 2024–2025, simulated stack, team and D/ST-plus-opposing-QB sums put about 80% of actual sums inside their 10–90% range (independent draws: 69–95%).

### 5. Optimizer ✅

`model/optimize.py` scores every legal lineup by P(win) on the same 20,000 simulated weeks of both lineups.

- **Late swap:** players whose game has started are locked at their current points.
- **Opponent:** keeps their current lineup; an Out starter is replaced from their bench.
- **Tie-break:** when P(win) is tied within noise (e.g. near 0% or 100%), the lineup with the most expected points wins.
- **The web page** (`web/`, `ff-archive web`) shows:
  - a recommendation card with swaps against the current lineup, and urgent flags for starters who can't score
  - a head-to-head with model-vs-ESPN tinting and Stack / Hedge / Double-risk tags (from the estimated correlations)
  - a player panel with a plain-English summary, distribution, what moved the expectation, evidence, stat line, props, and what-if controls
  - tabs: Weekly Matchup, Weekly Context (grouped by game), Weekly Review, Season Overview (replay record plus each week's margin split into projection, role, luck and K/D/ST), Trade Analyzer, Rankings (the league-free shareable page; the published claude.ai link is updated by Claude from `ff-archive rankings`)
  - player props (information only, never a model input): on the rankings page, a Books column turns each player's props into points on ESPN's scale. Each prop stat (books' median line; expected TDs from the anytime price) is divided by its slate-wide books/ESPN ratio first, because lines are medians and sit ~17–20% under ESPN's yardage averages
  - the Trade Analyzer (`model/trades.py`): every completed trade in the league (from ESPN's activity feed; `mTransactions2` only details your own team's trades), your trades in detail and every trade judged from both sides, what each side scored since, and your highest-projection lineup each week with vs. without the trade, scored on actual results. Assumes you'd have kept the outgoing players; a trade's window ends when its incoming players are traded on.
  - the Weekly Review tab, for finished weeks (`model/review.py`): each result split into ESPN's projection + **role** (what the usage he got is usually worth, from nflverse expected points, minus ESPN) + **luck** (actual minus that), with rule-based guesses for why usage differed (snap share, next week's injury report, teammates out or leaving early, final score vs. the line, team volume, wind). Information only.
- Any week (live, or replay of 2026 W1–9 with final results) and any team can be shown.

### 6. Backtest ✅ and live tracking ⏭️

- **League replay**, 2026 W1–9 (`model/backtest.py`): all 144 team-weeks decided as of Sunday 11:45 ET.
- **Synthetic walk-forward**, 2022–2025 (`model/backtest_synth.py`, `backtest-synth --variant … --seasons …`): random 16-team leagues from real player-weeks, with models fit only on earlier seasons. An earlier run used each player's last-week distribution for every week (a lookup keyed by player only); those results were void and are kept in `data/backtest_synth_invalid/`.
- **Result for the live variant (ESPN + context):** where it differs from the highest-projection lineup (about 34% of decisions), it gains **+0.60 ± 0.48 points**. Win-rate change is +0.2%, within noise. P(win) is well calibrated (predicted 85% won 82–86%).
- **ESPN's own win probability is too cautious.** It behaves like a margin SD of about 56 points; this league's actual margins vary about 27 around the projected margin (2026 W1–9, 72 matchups). Favorites projected 25+ points ahead won 86% of the time.
- **Still to do:** log each live week (model pick, my pick, outcome) once Tuesday finals land, and add live weeks to the season tab.

## Code and data

```
src/ff_model/
  config.py  db.py  cli.py         shared by all phases
  sources/   espn.py  odds.py  weather.py  stadiums.py     API clients and reference data
  archive/   slots.py  runner.py  wakes.py  launchd.py  parse.py  context.py
  research/  nflverse.py  espn_history.py  scoring.py  projection_errors.py
             factor_tests.py  line_injuries.py  vacated.py  opportunity.py
  model/     marginals.py  correlation.py  simulate.py  optimize.py  live.py  report.py
             backtest.py  backtest_synth.py  evaluate.py  rankings.py  review.py  trades.py
  web/       server.py  static/index.html  static/rankings.html
```

**Commands** (`uv run ff-archive …`):

| Command | What it does |
|---|---|
| `status` | upcoming and missed slots, recent snapshots, odds credits, props plan |
| `snapshot [--odds] [--prev-week]` | take a manual snapshot (odds cost credits) |
| `tick` | what launchd runs every 5 min |
| `reparse` | rebuild the derived tables from saved raw responses |
| `transactions WEEKS…` | archive ESPN's transaction log for those weeks (one request each) plus the league trade feed |
| `backfill-finals WEEKS…` | save final lineups and points for past weeks |
| `nflverse` | download or refresh nflverse history (run by hand on Tuesdays) |
| `espn-history [--seasons] [--weeks]` | download ESPN weekly projections (2019–2025; completed weeks of 2026) |
| `screen-factors` | regenerate the factor evidence table shown on the page |
| `fit-marginals` | refit player distributions on 2019–2025 (~4 min) |
| `fit-correlation` | re-estimate role-pair correlations |
| `recommend [--snapshot N]` | best lineup by P(win) for my matchup, as text |
| `web` | the recommendation page at http://127.0.0.1:8765 (local only) |
| `rankings [--snapshot N]` | write a shareable all-player rankings page to `data/share/` (no league details) |
| `backtest-synth --variant V --seasons …` | synthetic walk-forward backtest (run a single season first) |

**Data** (all under `data/`, gitignored, only on this laptop):

| Path | Contents | Replaceable? |
|---|---|---|
| `archive.db` | live snapshots. `raw_responses` (gzipped JSON) is the source of truth; other tables come from `parse.py`. League data (ESPN league endpoints, `player_snapshots`, `lineups`, `matchups`) carries `league_id`, part of the key; shared data (odds, props, weather, injury reports, NFL schedule) has none. Readers filter on `ESPN_LEAGUE_ID` | **No** |
| `espn_history/` | raw ESPN projection responses + `projections.parquet` | Only while ESPN keeps serving them |
| `nflverse/` | Parquet per dataset and season | Yes, re-download in ~20 s |
| `models/` | fitted `marginals.json`, `correlation.json` | Yes, refit (~5 min) |
| `backtest_synth/`, `backtest_league_2026.parquet` | backtest results | Yes, rerun |

## Open questions

- Odds API quota reset date: assumed the 1st of the month. Check the account dashboard.
- Backups: `archive.db` and `espn_history/` exist only on this laptop. A cloud-folder copy has been proposed.
- Whether ESPN's historical projections freeze at kickoff or before inactives: compare the archived W10 snapshots with what the API serves after the week.
- FantasyPros: decided against. Player props from The Odds API are the better second source.
