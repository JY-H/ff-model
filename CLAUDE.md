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
| 6. Backtest and live tracking | 🟡 League 2026 W1–9 done; synthetic 2019–2025 next |

### 1. Archiver ✅

Saves the data that disappears if it isn't captured at the time. launchd runs `tick` every 5 minutes, and it only takes a snapshot when a slot is due.

**Each snapshot saves:**

- ESPN projections and injury status for every player in the pool
- all 16 rosters and lineups, plus all matchups (with ESPN's own win probability)
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

**nflverse (2012 to now):** weekly player and team stats, snap counts, schedules with closing spread and total, injury reports, `ff_opportunity` (expected fantasy points), and `ff_playerids` (ID crosswalk).

- ESPN player IDs map to nflverse `gsis_id` through `ff_playerids.espn_id`. All rostered skill players and kickers match.
- D/ST joins on NFL team instead.
- **Leakage:** current-season files include games already played this week, and `schedules.spread_line` for unplayed games is a live line, not a closing one.

**League scoring** (`research/scoring.py`) applies this league's rules to both nflverse actuals and ESPN stat lines.

- Validated against ESPN: 99.6% of 2026 W1–9 actuals match exactly, and all 686 W10 projections match.
- ESPN quirks it handles: a blocked FG counts as a miss; points allowed excludes points the opponent's defense or special teams scored; yards allowed is net of sacks (nflverse stores sack yards as negative in recent seasons).

### 3. Player distributions ⏭️

Split each projection into a yardage/receptions part (roughly normal) and a TD part (Poisson-like), so TD-heavy players get the right skew. Use ESPN's projected stat lines for the split. Fit on 2019–2023 and test on 2024–2025 with PIT histograms.

### 4. Correlation model

Factors for each game's environment, each team's passing and rushing, and a negative link between an offense and the opposing D/ST. Join them to the player distributions with a copula. Fit on early seasons, test on later ones.

### 5. Optimizer

- Enumerate legal lineups, run a Monte Carlo of both lineups (~20k draws), and rank by P(win).
- **Late swap:** re-run at each lock, with scores already played fixed.
- **Output:** best lineup, P(win) gain over the highest-projection lineup, whether the decision is close, and the main reason.
- **Interface:** a local web page where I trigger the optimizer myself.

### 6. Backtest and live tracking

- Walk-forward on synthetic matchups built from real player-weeks.
- Measure lift only on decisions where the model disagrees with "start the highest projection."
- Log each live week: model pick, my pick, outcome. Bench points come from the Tuesday snapshot.

## Code and data

```
src/ff_model/
  config.py  db.py  cli.py         shared by all phases
  sources/   espn.py  odds.py      API clients
  archive/   slots.py  runner.py  wakes.py  launchd.py  parse.py
  research/  nflverse.py  espn_history.py  scoring.py  projection_errors.py
```

Later phases go in sibling packages (`model/`, `optimize/`, `web/`).

**Commands** (`uv run ff-archive …`):

| Command | What it does |
|---|---|
| `status` | upcoming and missed slots, recent snapshots, odds credits, props plan |
| `snapshot [--odds] [--prev-week]` | take a manual snapshot (odds cost credits) |
| `tick` | what launchd runs every 5 min |
| `reparse` | rebuild the derived tables from saved raw responses |
| `backfill-finals WEEKS…` | save final lineups and points for past weeks |
| `nflverse` | download or refresh nflverse history (run by hand on Tuesdays) |
| `espn-history [--seasons] [--weeks]` | download ESPN weekly projections (2019–2025; completed weeks of 2026) |
| `fit-marginals` | refit player distributions on 2019–2025 (~4 min) |
| `fit-correlation` | re-estimate role-pair correlations |
| `recommend [--snapshot N]` | best lineup by P(win) for my matchup from the latest (or given) snapshot |
| `web` | open the recommendation page at http://127.0.0.1:8765 (local only) |

**Data** (all under `data/`, gitignored, only on this laptop):

| Path | Contents | Replaceable? |
|---|---|---|
| `archive.db` | live snapshots. `raw_responses` (gzipped JSON) is the source of truth; other tables come from `parse.py` | **No** |
| `espn_history/` | raw ESPN projection responses + `projections.parquet` | Only while ESPN keeps serving them |
| `nflverse/` | Parquet per dataset and season | Yes, re-download in ~20 s |

## Open questions

- Odds API quota reset date: assumed the 1st of the month. Check the account dashboard.
- Backups: `archive.db` and `espn_history/` exist only on this laptop. A cloud-folder copy has been proposed.
- FantasyPros: decided against. Player props from The Odds API are the better second source.
