# FFModel

A weekly start/sit tool for a head-to-head ESPN fantasy football league (16 
teams, full PPR, 4-point passing TDs). It archives everything that matters at lineup lock, turns ESPN's projections into full outcome distributions adjusted for pre-game context, and picks the lineup that maximizes the probability of winning this week's matchup.

A local web page (`ff-archive web`) shows the recommendation, the context behind every adjustment, a post-week review of what happened and why, a season overview and a trade analyzer.

---

## Contents

1. [Core hypothesis](#core-hypothesis)
2. [Prerequisites](#prerequisites)
3. [Quick start](#quick-start)
4. [Architecture](#architecture)
5. [Data layer: the archive](#data-layer-the-archive)
6. [Model pipeline](#model-pipeline)
7. [Context signals](#context-signals)
8. [Post-week analysis](#post-week-analysis)
9. [The web page](#the-web-page)
10. [Validation](#validation)
11. [Repository layout and commands](#repository-layout-and-commands)
12. [Rules and constraints](#rules-and-constraints)
13. [Known limitations](#known-limitations)

---

## Core Hypothesis

Don't build point projections from scratch. ESPN's projection is a reasonable forecast of each player's average; the edge is in the decision layer around it:

1. A distribution per player, not just a point estimate. A 12-point receiver whose points are mostly touchdown equity has a very different week-to-week shape from a 12-point possession receiver.
2. A correlation structure between players. A QB and his WR1 rise and fall together; a D/ST falls when the QB it faces booms.
3. An optimizer for P(win), because head-to-head is win/lose. A heavy favourite should take lower-variance options; an underdog should take upside. Starting a player from the same game as an opponent's starter (a "hedge") reduces the variance of the margin.

On top of ESPN's mean, the model only adds pre-game context that ESPN measurably under-prices (weather, the betting line, the injury report, missing linemen). Every signal had to beat ESPN on out-of-sample games to be kept.

---

## Prerequisites

Nothing league-specific ships with this repo. All data lives in `data/` (gitignored) on the machine that runs it, so anyone can point it at their own ESPN league and build their own archive. Scoring rules and lineup slots are read from your league's settings, and the models are refit on ESPN's projection history rescored with your rules.

You need:

- Python 3.12+ and [uv](https://docs.astral.sh/uv/).
- Access to an ESPN fantasy football league (head-to-head).
- A free [The Odds API](https://the-odds-api.com) key (500 credits/month).
- macOS, only for automatic archiving (launchd and `pmset` wakes). Everything else runs anywhere.

Create a `.env` file in the repo root. It's gitignored; never commit or print it.

```
ESPN_LEAGUE_ID=...
ESPN_TEAM_ID=...
espn_s2=...
SWID={...}
ODDS_API_KEY=...
```

| Variable | Where to find it |
|---|---|
| `ESPN_LEAGUE_ID` | `leagueId=` in your league's URL on fantasy.espn.com |
| `ESPN_TEAM_ID` | `teamId=` in the URL of your team page |
| `espn_s2`, `SWID` | ESPN session cookies: sign in to fantasy.espn.com, then open your browser's developer tools → Application (Chrome) or Storage (Firefox) → Cookies → `espn.com`. Needed for private leagues. `SWID` includes the curly braces. |
| `ODDS_API_KEY` | emailed when you sign up for The Odds API |

The cookies expire periodically; when ESPN starts returning 401/403, copy fresh ones into `.env`.

---

## Quick start

```bash
uv sync
uv run ff-archive nflverse          # download nflverse history (~20 s)
uv run ff-archive espn-history      # ESPN weekly projections 2019-2025 (slow, polite)
uv run ff-archive fit-marginals     # player distributions (~4 min)
uv run ff-archive fit-correlation   # role-pair correlations
uv run ff-archive snapshot          # take a snapshot now
uv run ff-archive web               # http://127.0.0.1:8765
```

To archive automatically, `ff-archive install-launchd` writes a macOS LaunchAgent that runs `tick` every 5 minutes. A tick only checks whether a snapshot slot is due; it fetches nothing otherwise, so data is pulled roughly 8–10 times a week (see [Snapshot slots](#snapshot-slots)). `deploy/ff-archiver.sudoers` lets it schedule `pmset` wakes so the machine wakes for each slot.

---

## Architecture

```
                         ┌───────── snapshots at scheduled slots, ~8-10 a week (see Snapshot slots) ──────────┐
                         │                                                                                    │
 ESPN fantasy API ───────┤  player pool, rosters & matchups every slot; settings, transactions, trades Tue    │
 The Odds API ───────────┤  game lines every slot; player props once per game (Sun-Mon games at Sun 11:45)    │
 Open-Meteo ─────────────┤  kickoff-window forecasts for outdoor games, every slot                            │
 nflverse ───────────────┤  this week's injury report with practice participation, every slot                 │
                         └────────────────────────────────────┬───────────────────────────────────────────────┘
                                                              ▼
                                         data/archive.db  (raw_responses = source of truth)
                                         parse.py ──► player_snapshots · lineups · matchups ·
                                                      odds_lines · prop_lines · game_context · injury_reports
                                                              │
        offline research (by hand)                            │  live decision (per snapshot)
 ┌──────────────────────────────────────┐                     ▼
 │ nflverse 2012→  (stats, schedules,   │        model/live.py  build_inputs(snapshot)
 │   injuries, snaps, play-by-play,     │          only what was archived at or before the snapshot
 │   ff_opportunity, ID crosswalk)      │                     │
 │ ESPN projection history 2019-2025    │                     ▼
 │ league scoring (research/scoring.py) │        model/marginals.py   ESPN + context quantiles (per player)
 │        │                             │                     │
 │        ▼                             │                     ▼
 │ factor screen ─► fit-marginals ──────┼──► models/   model/correlation.py  role-pair Gaussian copula
 │ role pairs   ─► fit-correlation ─────┼──► *.json             │
 └──────────────────────────────────────┘                     ▼
                                         model/simulate.py + optimize.py   20,000 joint weeks, P(win) per lineup
                                                              │
                                                              ▼
                       ff-archive recommend (text)   ·   ff-archive web (local page)   ·   ff-archive rankings (shareable)
```

Three properties hold everywhere:

- **Raw first.** Every external response is stored verbatim (gzipped JSON) before anything is derived from it. All other tables can be rebuilt with `ff-archive reparse`.
- **No lookahead.** Live decisions and backtests share `build_inputs(snapshot_id)`, which only reads data archived at or before that snapshot, so a backtest can't see anything a live decision couldn't.
- **Polite to free APIs.** ESPN's API is unofficial: requests are seconds apart and the client stops on 401/403/429. The Odds API free tier (500 credits/month) is budgeted by a planner that can never exceed it.

---

## Data layer: the archive

### Snapshot slots

`archive/slots.py` generates slots from the NFL schedule (all times ET). A slot is only taken when due; `tick` is cheap otherwise.

| Slot | When | Why |
|---|---|---|
| Tue | 12:00 | start of the week; also saves last week's final lineups and points (bench included) and the transaction log |
| Wed | 12:00 | after Wednesday waivers |
| Sat | 22:00 | final injury reports are out |
| Sun early | 11:45 | after inactives, before 1 pm kickoffs |
| Sun late | 15:55 | before 4 pm kickoffs (props already pulled at 11:45) |
| International | 90 min before kickoff | |
| Every other kickoff (Thu, Sun night, Mon, Sat) | 60 min before kickoff | late swap: lineups lock per game |

Each snapshot saves:

- ESPN projections (with full projected stat lines) and injury status for every player in the pool
- all rosters, lineups and matchups, including ESPN's own win probability
- game lines from The Odds API, and player props for games that lock before the next slot
- pre-game context: a kickoff-window weather forecast for every outdoor game, the stadium and roof for every game, and the week's nflverse injury report with practice participation (offensive line and defensive backs included)
- on Tuesdays: league scoring settings, last week's finals, and the transaction log plus the league trade feed; then last week is finished for review: an nflverse refresh and its ESPN projection history (retried Wednesday if nflverse lags)

### Odds budget

- Game lines: 2 credits per call (spreads + totals), every slot.
- Player props: 1 credit per market per game, pulled once per game: at the last slot before it for Thursday, Saturday and international games, and at Sun 11:45 for every game from Sunday 1 pm through Monday night, so start/sit calls made before 1 pm can use them. Pages show when each player's props were pulled and how long before kickoff. Markets in priority order: anytime TD, receptions, receiving yards, rushing yards, passing yards. A planner drops markets from the end of the list if the rest of the month can't be covered.
- No calls below 40 credits remaining. The quota is assumed to reset on the 1st.

### Tables

| Table | Contents | League-specific |
|---|---|---|
| `raw_responses` | every response, gzipped, with kind, params, status and hash | `league_id` set for ESPN league endpoints |
| `snapshots` | one row per slot run: week, slot, status, errors | |
| `player_snapshots` | per player per snapshot: projection, stat line, status, ownership | yes (league-scored) |
| `lineups`, `matchups` | rosters with lineup slots and points; matchups with ESPN win probability | yes |
| `odds_lines`, `prop_lines` | per book, market and outcome | |
| `game_context` | stadium, roof, forecast per game | |
| `injury_reports` | nflverse report per player: status, practice, injury | |
| `pro_games` | NFL schedule with kickoffs | |
| `odds_ledger`, `scheduled_wakes` | credits used/remaining; pmset wakes | |

League data carries `league_id` as part of its key; shared data (odds, weather, injuries, schedule) has none and would serve a second league unchanged. Readers filter on `ESPN_LEAGUE_ID`.

### Historical data (offline)

- **ESPN projection history, 2019–2025.** ESPN's public endpoint still serves weekly projections with full stat lines for these seasons (2018 returns nothing). They were verified to be pre-game projections, not finals: they correlate with actuals at about 0.56 and match exactly in about 1% of player-weeks, and late scratches keep their projection. Their injury/status fields are current status, so they're ignored in favour of nflverse injury reports.
- **nflverse, 2012 onward:** weekly player and team stats, snap counts, schedules with closing lines, injury reports, `ff_opportunity` (expected fantasy points), trimmed play-by-play, and the `ff_playerids` crosswalk (ESPN ID ↔ nflverse `gsis_id`; D/ST joins on team).
- **League scoring** (`research/scoring.py`) applies this league's rules to both nflverse actuals and ESPN stat lines. It matches ESPN exactly for 99.6% of 2026 W1–9 actuals and all W10 projections, including ESPN's quirks (a blocked FG counts as a miss; points allowed excludes points scored by the opponent's defense or special teams; yards allowed is net of sacks).

---

## Model pipeline

### 1. Inputs at lock — `model/live.py`

`build_inputs(snapshot_id)` assembles, for every player, only what was archived at or before that snapshot: ESPN projection and stat line, ESPN injury designation, opponent and kickoff, roof and forecast, the spread (median across books, this week's game only), nflverse practice status, games back after an absence, and starters out on both lines.

### 2. Player distributions — `model/marginals.py`

One linear quantile regression per position and percentile (23 levels, 1st to 99th) predicts league points from:

- the ESPN projection (plus √projection for QB/RB/WR/TE, so spread grows sub-linearly);
- the make-up of the projection from ESPN's stat line: touchdown share and receptions per point for RB/WR/TE; passing-TD share and rushing share for QBs. A projection built on touchdowns is lumpier than one built on catches;
- context signals (next section), each interacted with the projection so effects scale with volume, and fit separately per percentile so a signal can move the floor and the ceiling differently.

K and D/ST stay linear (their projections are compressed; a curved term made kicker ceilings fall as projections rose).

Live distributions are "ESPN + context". The walk-forward backtest found the best variant keeps ESPN's mean and adds only the context's shift: the model's quantile curve is shifted so its mean equals `ESPN projection + (mean with context − mean without context)`. So every player card adds up exactly: model = ESPN + listed effects.

Calibration (2024–2025 and the 2026 holdout): about 10% of outcomes land below the predicted 10th percentile and about 10% above the 90th. Known soft spots: kickers' upside is a little narrow, and 15+ mph wind is slightly optimistic at the low end.

### 3. Correlation — `model/correlation.py`

A Gaussian copula over each player's position within his own distribution. Correlations are estimated per role pair from 2019–2025 games, where roles are assigned per team-week by projection: QB1, RB1–2, WR1–3, TE1, K, D/ST. Two matrices: same team, and across opponents in the same game.

| Pair | Correlation |
|---|---|
| QB – own WR1 | +0.35 |
| QB – own TE1 | +0.24 |
| RB1 – RB2 (same team) | −0.09 |
| QB vs opposing QB | +0.20 |
| D/ST vs the QB it faces | −0.57 |
| D/ST vs the RB1 it faces | −0.32 |

These are stable across eras. On 2024–2025, simulated stack, team and D/ST-plus-opposing-QB sums put about 80% of actual sums inside their 10–90% range (independent draws ranged from 69% to 95%, i.e. badly miscalibrated without correlation).

### 4. Simulation and optimizer — `model/simulate.py`, `model/optimize.py`

- Draw 20,000 joint weeks for every player in the matchup (both rosters), with common random numbers so every candidate lineup faces the same simulated weeks.
- Enumerate every legal lineup (QB, RB×2, WR×2, TE, FLEX, D/ST, K), deduplicated by player set, with empty placeholders when a slot can't be filled.
- Score each by P(my total > opponent's total).
- **Late swap:** players whose game has kicked off are locked at their current points.
- **Opponent:** their highest-projection lineup from players who can still play (anyone whose game has started stays put), since a lineup set early in the week usually changes by kickoff. The page tags players placed differently from their current lineup. Replays use the lineup they actually started.
- **Tie-break:** when P(win) is tied within simulation noise (e.g. near 0% or 100%), the lineup with more expected points wins.

---

## Context signals

### Signals in the model

Each signal is measured before kickoff, enters the quantile regressions as `projection × signal`, and was kept only because it beat ESPN out of sample (`research/factor_tests.py`; the evidence table is `data/factor_screen.csv`, regenerated by `screen-factors`). The "evidence" column is the 2019–2025 shift in `actual − ESPN projection` for that group relative to the position's average, with its 95% interval and how many of the seven seasons agree in sign.

| Signal | How it's measured at lock | Positions | Evidence, 2019–2025 (points vs ESPN) |
|---|---|---|---|
| Dome / closed roof | stadium table (`sources/stadiums.py`); retractable roofs treated as closed | QB, WR, TE, K | QB +0.8 ± 0.5 (7/7 seasons), WR +0.6 ± 0.3 (7/7), K +0.6 ± 0.3 |
| Wind | Open-Meteo forecast for the kickoff window, outdoor games only | QB, WR, TE, K | 15+ mph: QB −1.3 ± 1.1, WR −0.8 ± 0.6, TE −1.1 ± 0.9, K −0.8 ± 0.6 (7/7). 10–14 mph: QB −0.9 ± 0.7 (7/7) |
| Cold | forecast temperature, outdoors | QB, WR, TE, K | 33–45°F: QB −1.0 ± 0.8, WR −0.9 ± 0.5, K −0.6 ± 0.5 (all 7/7) |
| Expected margin (spread) | median spread across books for this week's game, from the archived odds | all | favoured by 7+: TE +1.1 ± 0.6 (7/7), QB +0.9 ± 0.6. 7+ underdog: QB −1.0 ± 0.9, WR −0.6 ± 0.4, K −0.6 ± 0.6 |
| Injury designation × final practice | ESPN designation (Questionable/Doubtful) combined with the last nflverse practice report of the week: DNP / limited / other | RB, WR, TE | Q/D: WR −1.5 ± 0.6 (7/7), TE −1.2 ± 1.0, RB −0.8 ± 0.7. Played after a DNP: RB −1.9 ± 1.2, WR −1.5 ± 1.0. After limited practice: WR −1.0 ± 0.5, TE −1.2 ± 0.9 |
| Games back after an absence | 1st or 2nd game after missing 2+ games | RB, WR, TE | |
| Own offensive-line starters out | starters = 60%+ of snaps in earlier games; out = listed Out/Doubtful or on IR | QB, RB | RB with 2+ out: −0.6 ± 0.4 |
| Opponent's offensive-line starters out | same definition, for the team the D/ST faces | D/ST | 2+ out: +1.1 ± 0.6 (6/7) |
| Opponent's secondary starters out | defensive backs, same definition | QB | |

QB injury terms were dropped: there are only about 80 listed-QB weeks in training and they hurt out of sample. RBs get no weather terms: the run game showed no weather effect.

Two sources were chosen deliberately: injury designations come from ESPN (nflverse lags on designations) but practice participation comes from nflverse (ESPN doesn't carry it). Stadium roofs come from our own table because nflverse's labels are blank or wrong for some venues; a test checks that every remaining Thursday, Saturday, holiday and international slot finds its games with stadiums.

### Tested and left out (ESPN already prices them)

| Idea | Result |
|---|---|
| Opponent defense quality (yards allowed), points allowed to the position | ESPN prices it; a couple of cells clear the noise band but don't beat ESPN out of sample |
| Team pace (plays per game) | no reliable effect |
| Kicker stall rate (FG / (FG + TD)) and implied team total | no reliable effect beyond the spread |
| "Bad defense means they throw more" (opponent implied total, stats-based mismatch) | no reliable effect |
| Key teammates out (vacated targets/carries) | ESPN already raises the next players up; shown on the page as information only |
| Play-by-play opportunity (`research/opportunity.py`): last week's points over expected, expected points vs ESPN, target-share trend, air-yard share, inside-10 share | 3 of 60 position/group cells cleared the noise band, about what chance gives. The most consistent (goal-line RBs miss ESPN) disappears once projection size is controlled for and reverses in 2026. Evidence: `data/opportunity_screen.csv` |

### Information-only signals

Shown on the page to explain a week, never fed to the model:

- **Teammates out:** which key teammates are out and what share of targets/carries they vacate.
- **Props:** implied probabilities from the archived player props (anytime TD, receptions, yards).
- **Role and luck** (post-week; see below).

---

## Post-week analysis

### Weekly Review — `model/review.py`

For finished weeks, each player's result is split into three parts that add up exactly:

```
actual = ESPN projection + role + luck
role   = expected points from the usage he actually got − ESPN projection
luck   = actual − expected points from that usage
```

"Expected points from usage" is nflverse's `ff_opportunity` model (each target, carry and pass valued by where on the field and how far downfield it happened), rescored for this league. Role answers "did ESPN misjudge how much he'd be used?"; luck answers "what did he do with the work?". K and D/ST have no usage model, so their whole miss is shown against ESPN.

For each player the page shows his work vs his usual share (targets, carries, inside-10 plays, with ↑/↓ when the share moved 8+ points), what that work is worth vs ESPN, and the stats behind the luck (yards and TDs vs expected). When usage differs from ESPN's projection by 2+ points, rule-based likely reasons are listed, drawn from data observed in or after the game:

| Rule | Data | Fires when |
|---|---|---|
| Snap share vs his usual | snap counts | 20+ points below (less work) or 15+ above (more work) |
| ↳ hurt or benched? | next week's injury report | listed → "likely hurt in the game" |
| Teammates out before the game | injury report + IR | a key teammate (15%+ of targets or 30%+ of carries) was out |
| Teammate left early | snap counts | a regular (40%+ of snaps) at his position played 30+ points below usual |
| Game script | final score vs the closing spread | the team finished 10+ points worse than the line (trailing: more passing) or better (leading: more running) |
| Team volume | play-by-play | team carries (RBs) or targets (others) 20%+ off the team's usual |
| Wind | schedule | 15+ mph outdoors |

These are post-hoc guesses with judgment-call thresholds, not causes; nothing here predicts the next week (see the opportunity screen above). Matchups are also summarised as a margin waterfall: projected margin + role + luck + K/D/ST = final margin.

### Trade Analyzer — `model/trades.py`

Every completed trade in the league comes from ESPN's league activity feed (ESPN's transaction view only details your own team's trades; players dropped to make room come from it where known). For each trade and each side:

- what each incoming player scored while on the roster, and as a starter;
- what the outgoing players scored over the same weeks, wherever they went;
- the team's highest-projection lineup with vs without the trade (incoming players out, outgoing and dropped players back), both scored on actual results against the actual opponent. Using the best lineup rather than the one actually started isolates the trade from start/sit calls.

A trade's window ends when its incoming players are traded on; the later trade takes over from there. It assumes the outgoing players would have been kept and every other move made the same.

---

## The web page

`ff-archive web` serves a single-page app at `http://127.0.0.1:8765` (local only). Any week (live snapshots, or replay of finished weeks with final results) and any team can be shown.

| Tab | What it shows |
|---|---|
| Weekly Matchup | win chance with the recommended lineup vs the current one, the highest-projection lineup and ESPN's estimate; swaps with urgent flags for starters who can't score; head-to-head with model-vs-ESPN tinting, 10–90% ranges and Stack / Hedge / Double-risk tags from the correlations; margin histogram; runner-up lineups; benches with P(win) if started. Click a player for a panel with his distribution, what moved the expectation and why, the stat line, props, projection history and what-if controls. |
| Weekly Context | every game in the matchup with line, total, forecast and roof; each player's injury report, missing teammates and context effect; the full raw data for the week |
| Weekly Review | the role/luck breakdown above, bench calls (bench players who beat a starter they could have replaced), and league-wide lists of the luckiest/unluckiest players and who was used more or less than ESPN expected, with likely reasons |
| Season Overview | each finished week replayed: what was started vs the model's lineup vs the highest projection, and each week's margin split into projection, role, luck and K/D/ST |
| Trade Analyzer | your trades in detail and every league trade judged from both sides |

`ff-archive rankings` writes a separate, shareable single-file page (`data/share/`) ranking every projected player of the week with distributions, the reasons behind each adjustment and the scoring rules spelled out, with no rosters, team names or IDs.

---

## Validation

- **League replay, 2026 W1–9** (`model/backtest.py`): all 144 team-weeks decided as of Sunday 11:45 ET with the live models (fit on 2019–2025 only). The model's lineup differs from the highest-projection lineup in 23 of 144 team-weeks, gaining +2.1 points over them in total; the sample is far too small to show a win-rate difference.
- **Synthetic walk-forward, 2022–2025** (`model/backtest_synth.py`): random 16-team leagues built from real player-weeks, with models fit only on earlier seasons. For the live variant (ESPN + context), where the model's lineup differs from the highest-projection lineup (about 34% of decisions), it gains +0.60 ± 0.48 points per changed decision. The win-rate change is +0.2%, within noise. P(win) is well calibrated: predicted 85% won 82–86%.
- ESPN's own win probability is too cautious. It behaves like a margin standard deviation of about 56 points; this league's actual margins vary about 27 points around the projected margin (2026 W1–9, 72 matchups). Favourites projected 25+ points ahead won 86% of the time.

The honest summary: ESPN's projections are hard to beat on the mean. The model's value is mostly in knowing the distribution (calibrated win chances, floors and ceilings, correlation-aware hedging) and in pricing the handful of contexts ESPN under-adjusts for.

An earlier synthetic run used each player's last-week distribution for every week (a lookup keyed by player only); those results were void and are kept separately in `data/backtest_synth_invalid/`.

---

## Repository layout and commands

```
src/ff_model/
  config.py  db.py  cli.py                 settings from .env, SQLite schema + migrations, CLI
  sources/   espn.py  odds.py  weather.py  stadiums.py
  archive/   slots.py  runner.py  wakes.py  launchd.py  parse.py  context.py
  research/  nflverse.py  espn_history.py  scoring.py  projection_errors.py
             factor_tests.py  line_injuries.py  vacated.py  opportunity.py
  model/     live.py  marginals.py  correlation.py  simulate.py  optimize.py
             report.py  review.py  trades.py  rankings.py
             backtest.py  backtest_synth.py  evaluate.py
  web/       server.py  static/index.html  static/rankings.html
tests/       pytest suite (archive-dependent tests skip without data/)
deploy/      ff-archiver.sudoers (pmset wake rule)
```

Commands (`uv run ff-archive …`):

| Command | What it does |
|---|---|
| `status` | upcoming and missed slots, recent snapshots, odds credits, props plan |
| `snapshot [--odds] [--prev-week]` | take a manual snapshot (odds cost credits) |
| `tick` | what launchd runs every 5 minutes: takes a snapshot only if a slot is due |
| `install-launchd` | write the LaunchAgent |
| `reparse` | rebuild derived tables from raw responses |
| `backfill-finals WEEKS…` | save final lineups and points for past weeks |
| `transactions WEEKS…` | archive the transaction log for those weeks plus the league trade feed |
| `nflverse` | download or refresh nflverse history (the Tuesday slot refreshes the current season) |
| `finish-week N` | make a finished week reviewable: nflverse refresh + its ESPN projection history |
| `espn-history [--seasons] [--weeks]` | download ESPN weekly projections |
| `screen-factors` | regenerate the factor evidence table |
| `fit-marginals` / `fit-correlation` | refit the models on 2019–2025 |
| `recommend [--snapshot N]` | best lineup by P(win), as text |
| `web` | the local page |
| `rankings [--snapshot N] [--week N]` | the shareable all-player rankings page |
| `backtest-synth --variant V --seasons …` | synthetic walk-forward backtest (run a single season first) |

Data (all under `data/`, gitignored):

| Path | Contents | Replaceable? |
|---|---|---|
| `archive.db` | live snapshots; `raw_responses` is the source of truth | No |
| `espn_history/` | raw ESPN projection responses + `projections.parquet` | only while ESPN keeps serving them |
| `nflverse/` | Parquet per dataset and season | yes, re-download |
| `models/` | fitted `marginals.json`, `correlation.json` | yes, refit (~5 min) |
| `factor_screen.csv`, `opportunity_screen.csv` | factor evidence | yes, rerun |
| `share/` | generated rankings pages | yes |

Run the tests with `uv run pytest`.

---

## Rules and constraints

- **Credentials** (`espn_s2`, `SWID`, `ODDS_API_KEY`) live only in the gitignored `.env`. Never print, log or commit them; request URLs are kept out of logs because the odds key is a query parameter.
- **League privacy:** league and team IDs, team names and league-mates' details stay out of the repo, including code, docs and commit messages. They're read from `.env` or the archive. The shareable rankings page contains none of them.
- **ESPN API:** unofficial and undocumented. Requests are spaced seconds apart, the client stops on 401/403/429, and it's expected to break.
- **No lookahead:** the model only uses data available at lineup lock. nflverse's current-season files include games already played this week, and `schedules.spread_line` for unplayed games is a live line, so features for week W filter to weeks before W.
- **Odds budget:** 500 credits/month; nothing may be designed that could exceed it.

---

## Known limitations

- The W1–9 league replay uses ESPN's projection history, which holds each game's last pre-kickoff projection (checked against the archived W10 snapshots). For 4 pm and Monday games that includes Sunday-afternoon updates made after the 11:45 decision time, so the replay has slight hindsight on late games. Live weeks (W10 on) use the archived 11:45 snapshot.
- The Odds API quota reset date is assumed to be the 1st of the month.
- `archive.db` and `espn_history/` are stored only locally; there's no backup elsewhere yet.
- The post-week reasons are rule-based guesses with hand-picked thresholds.
- The trade analyzer judges trades on finished weeks only, assumes outgoing players would have been kept, and only knows other teams' drop-to-make-room moves for trades involving your team.
- One league per configuration; the archive is ready for more (`league_id`), the fetch loop and page are not.
