"""Hidden page (app URL + ?admin=pull) to run the one-time historical props pull from Streamlit Cloud."""
import pandas as pd
import streamlit as st

import history_pull as hp
from props_tab import _api_key


def render():
    st.subheader("Historical props pull")
    key = _api_key()
    if not key:
        st.error("No ODDS_API_KEY in the app's Secrets.")
        return
    c1, c2 = st.columns([2, 3])
    seasons = c1.multiselect("Seasons", [2024, 2025, 2026], default=[2025, 2026])
    wk = c2.text_input("Weeks (blank = all, e.g. 1 or 1,2,3)", "")
    weeks = [int(x) for x in wk.replace(" ", "").split(",") if x] or None
    if not seasons:
        return
    G = hp.games(seasons, weeks)
    est = hp.estimate(G)
    st.write(f"**{est['games']}** completed games · already saved **{est['already_saved']}** · "
             f"to pull **{est['to_pull']}** · estimated **{est['credits']:,} credits** "
             f"(markets: {', '.join(hp.MARKETS)}; stops if remaining credits fall below {hp.CREDIT_FLOOR:,}).")
    confirm = st.text_input("Type PULL to enable the button", "")
    if st.button("Pull", type="primary", disabled=confirm != "PULL" or est["to_pull"] == 0):
        bar = st.progress(0.0)
        line = st.empty()

        def prog(k, n, gid, s):
            bar.progress(min(k / max(n, 1), 1.0))
            line.caption(f"{k}/{n} · {gid} · credits used {s['used']:.0f} · remaining {s['remaining']}")

        state = hp.pull(key, G, prog)
        d = hp.combine()
        msg = (f"Saved {state['saved']} games ({state['empty']} with no lines at the snapshot) · "
               f"credits used {state['used']:.0f} · remaining {state['remaining']}")
        (st.warning if state["stop_reason"] else st.success)(
            msg + (f" · stopped: {state['stop_reason']}" if state["stop_reason"] else ""))
        if state["unmatched"]:
            st.caption(f"No Odds API event found for: {', '.join(state['unmatched'])}")
        st.session_state["hp_rows"] = len(d)
    if hp.CSV.exists():
        d = pd.read_csv(hp.CSV)
        st.write(f"Saved file: {d['game_id'].nunique()} games, {len(d):,} lines.")
        st.dataframe(d.groupby(["season", "market"]).agg(games=("game_id", "nunique"), lines=("line", "size")),
                     width="content")
        st.download_button("Download props_lines.csv.gz", hp.CSV.read_bytes(), file_name="props_lines.csv.gz",
                           mime="application/gzip")
        st.caption("Upload the downloaded file to a folder named **history** in your GitHub repo. The app's "
                   "storage is wiped on reboot; the repo copy is what stops games being pulled (and paid for) twice.")
