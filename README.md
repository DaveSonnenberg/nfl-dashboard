# NFL Dashboard: Props, Matchups, QB Rushing — v1.4

Personal dashboard for fantasy and props. Shows what each defense allows to an
offense's WR1 / WR2 / WR3 / TE / RB, with this week's matchups ranked by edge.

Data comes straight from nflverse (refreshed nightly in season). The app caches
it for 6 hours. Nothing to schedule or maintain.

## Deploy (one time, ~15 min)

1. **Create a repo.** On GitHub: New repository → name it `nfl-dashboard` →
   Private is fine → Create.
2. **Upload the files.** In the new repo: *Add file → Upload files*, drag in
   everything from this folder **including the `.streamlit` folder** → Commit.
3. **Connect Streamlit.** Go to share.streamlit.io → sign in with GitHub →
   *Create app* → *Deploy a public app from GitHub* → pick the repo,
   branch `main`, main file `app.py` → Deploy.
4. First load takes ~1–2 minutes while packages install. Bookmark the URL.

Optional: in the app's Streamlit settings → *Sharing*, restrict viewers to
your email.

## Layout (v1.4)

Three tabs, each with its own filters at the top (no sidebar):
**Props** · **Matchups** (This week / Defense grid / Defense game log / Team roles)
· **QB rushing**.

## Props tab

Projections for **receptions, receiving yards, rush attempts, rushing yards and
passing yards** for every team's next game, compared with sportsbook lines.

Columns: player · team · opponent · season and last-4 averages · usage driver
(target/carry share, attempts, YPA) · what the opponent allows (with rank) ·
**Projection** (average) · **Fair line** (where the model sees 50/50) · line and
odds · **P(over)** · **Book P(over)** (margin removed) · **Edge** (green at ≥5
pts) · **EV** per $1 · flags (injury status, <4 games, new team, teammates out).
**Click a row** to open the player's window: how the projection was built, the
outcome distribution with the line marked, what the opponent allows by role, and
his season game log.

- **Pos** is the player's role this week among teammates expected to play
  (WR1–WR4, TE1–TE2, RB1–RB2, QB). Ruling players out re-ranks roles.
- **Opp allows** (receiving) = receptions or yards per game the opponent has
  allowed to his role this season, with rank among 32 defenses. Toggle
  **Opponent by role** to see WR1–WR4, TE1–TE2 and RB side by side. This is
  context only: in backtests the role funnel didn't improve projections.
- **Rushing markets include starting QBs** (projected 2.5+ carries or with a
  line). QB rushing model: his carries per start (this season blended with his
  last two), × his YPC, adjusted for spread/total; backtested in
  `model/qb_rush.py` (beats his season average by ~6–7%).

### Odds API setup (one time)
1. Get a key at the-odds-api.com (free plan = 500 credits/month).
2. In Streamlit Cloud: your app → **Settings → Secrets**, add
   `ODDS_API_KEY = "your-key"` → Save. The app restarts.
3. In the Props tab press **Pull latest odds**. One pull ≈ 5 credits per game,
   ~70–80 for a full week (free plan ≈ 6 pulls/month). Pulls are reused for 12
   hours so reloading the page doesn't spend credits.

No key, or a market missing from the feed? Use **Enter or override lines by
hand** under the table (kept for your browser session only).

### How the projections work
Backtested on 2024 and 2025 (trained on earlier seasons). Code and write-ups in
`model/`; coefficients in `model/coefs.json`, outcome distributions in
`model/dist_specs.json`.
- Receiving: team targets × target share (recent games weighted more) × catch
  rate / yards per target, adjusted for spread, total and opponent targets allowed.
- Rushing: team RB carries × carry share × yards per carry, adjusted for spread,
  total, opponent carries and YPC allowed.
- Passing: team attempts × QB yards per attempt, adjusted for spread, total,
  opponent attempts and YPA allowed. Starter = nflverse's listed starter.
- **Ruled-out players:** anyone with an official **Out** designation (nflverse
  injury report) is removed automatically. Designations usually reach nflverse
  Friday night/Saturday, so use the **Ruled out this week** picker at the top of
  the Props tab to remove players sooner. Questionable/Doubtful players stay in
  (flagged). A removed player's targets go mainly to teammates at his position
  (half of his share is redistributed; backtested in `model/redistribute.py`);
  carries go to the remaining backs.

## Tabs

| Tab | What it answers |
|---|---|
| **This week** | Every projected WR1/2/3, TE1, RB1 for the upcoming week, sorted by how soft the opponent is vs. league average for that role |
| **Defense grid** | All 32 defenses × role, heat-mapped. Sort by any role |
| **Defense game log** | One defense, game by game, with the names who did the damage |
| **Team roles (audit)** | How the app labeled a team's receivers each week; where to spot mistakes |

Sidebar: season, stat (PPR, yards, receptions, targets, TDs, air yards),
per-game vs. league-relative view, and week range (drag the left handle for
recent form only).

## QB rushing tab (v1.1)

Three sub-tabs, all using the sidebar season and week range:

- **QB profiles:** every starter with designed runs/g, sneaks, YPC ex-sneak,
  dropbacks/g, scramble %, yards/scramble, kneel yards, average and median
  official rush yards, % of games over your line, and last-4 versions. Pick a QB
  for his splits (with sample sizes), next opponent's defense card, and game log.
- **Defense profiles:** QB rushing allowed, plus **vs norm** columns (what QBs did
  against this defense minus what those same QBs did in their other starts),
  blitz %, sack+hit %, pass EPA allowed, and run D vs RBs. Pick a defense to see
  each QB it faced vs. his norm.
- **League-wide splits:** all starters pooled by opponent trait, to test whether
  a trait moves QB rushing at all.

Definitions: starter = most dropbacks in the game. Official yards include kneels
(how props settle); designed/scramble numbers exclude kneels. Sneaks use FTN's
flag, falling back to 3rd/4th-and-1 designed runs when a play isn't charted yet.

### Coverage & play-callers (v1.2)

- `coordinators.csv`: head coach, DC and **defensive play-caller** for every team,
  2022–2026, with sources. Edit it in GitHub to fix a row; the app uses the
  `def_playcaller` column (text before any `;`).
- `coverage_team_seasons.csv` / `coverage_league.csv`: precomputed man/zone rate,
  true pressure rate and scrambles per team-season (2022–2025) from NFL Next Gen
  Stats via nflverse. These seasons don't change, so the app reads the files
  instead of re-downloading. After nflverse publishes a new season's
  participation data (usually after the season), rebuild locally with
  `python coverage.py 2022 2023 2024 2025 2026` and commit the two CSVs.
- In the app: Defense profiles gain Play-caller / Coverage lean / Caller man z;
  the QB detail's next-opponent card shows the play-caller and his past man
  rates; the **Coverage & play-callers** sub-tab lists every defense.
- Coverage lean = average man rate of defenses the current play-caller called in
  prior seasons, vs that season's league: Man-heavy ≥ +0.5 SD, Zone-heavy ≤ −0.5 SD,
  Unknown = no prior play-calling data.

## How roles are assigned

- WRs who played an offensive snap are ranked by **target share entering the
  game** (games they played, this team only). A game's own result never sets
  its role.
- Early in the season that share is blended with last season's
  (last season counts like ~70 team targets ≈ 2 games). Rookies with no
  history fall back to snap share.
- TE and RB are aggregated by position (all TEs, all RBs).
- A slot with nobody in it (e.g. only 2 WRs played) counts as zero.

## Fixing a role: `role_overrides.csv`

Edit it in GitHub (pencil icon → Commit). The app redeploys in about a minute.

```
season,week,team,player,role
2026,,CIN,Tee Higgins,WR1
2026,6,CIN,Andrei Iosivas,WR2
```

- Blank `week` = all season; a number = that game only.
- `role`: WR1, WR2, WR3, or WR4 (pushes someone out of the top 3).
- `player`: full name exactly as the audit tab shows it.

## Known limits (v1.0)

- Receiving production only (no rushing).
- No opponent adjustment: a defense that faced three elite WR1s looks soft.
  Use the game log and the G column to judge.
- "This week" lists only players who played their team's last game, so check
  injury news.
- Team codes follow nflverse (Rams = `LA`).

## Run locally (optional)

```
pip install -r requirements.txt
streamlit run app.py
```
