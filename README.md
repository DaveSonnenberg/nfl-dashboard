# NFL Defense vs Role + QB Rushing — v1.1

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
