"""Streamlit front end: run every planner on one scenario, watch the chase, compare the numbers.

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import figures as F  # noqa: E402
from i18n import TEXT  # noqa: E402
from uavguide.compare import run_all, summarize  # noqa: E402
from uavguide.config import Config, SimParams  # noqa: E402
from uavguide.metrics import DEFAULT_WEIGHTS, outcome, rank  # noqa: E402
from uavguide.pursuit import METHODS, PursuitPolicy  # noqa: E402
from uavguide.scenarios import make_scenario  # noqa: E402
from uavguide.world import Obstacle, World  # noqa: E402

st.set_page_config(page_title="Akıllı Rota", page_icon="🛩️", layout="wide")

# ----------------------------------------------------------------------------- language
lang = st.sidebar.radio("Dil / Language", ["tr", "en"], format_func={"tr": "Türkçe", "en": "English"}.get,
                        horizontal=True)


def T(key: str, **kw) -> str:
    return TEXT[lang][key].format(**kw) if kw else TEXT[lang][key]


st.title(T("title"))
st.caption(T("subtitle"))

# ----------------------------------------------------------------------------- sidebar
sb = st.sidebar
sb.header(T("sb_scenario"))
seed = int(sb.number_input(T("seed"), 0, 9999, 3, help=T("seed_help")))
n_zones = sb.slider(T("n_zones"), 4, 20, 12)
zone_scale = sb.slider(T("zone_scale"), 0.5, 2.0, 1.0, 0.05, help=T("zone_scale_help"))
target_mode = sb.selectbox(T("target"), ["line", "circle", "fixed"], format_func=lambda m: T("target_" + m))
target_speed = sb.slider(T("target_speed"), 13.0, 24.0, 17.0, 0.5, disabled=target_mode == "fixed")
wind_speed = sb.slider(T("wind_speed"), 0.0, 10.0, 5.0, 0.5)
wind_from = sb.slider(T("wind_from"), 0, 359, 270, help=T("wind_from_help"))
hidden = sb.checkbox(T("hidden"), False, help=T("hidden_help"))
detect = sb.slider(T("detect"), 40, 200, 80, 10, disabled=not hidden)

sb.header(T("sb_planning"))
DEFAULT_METHODS = ["A* + DP", "Theta* + DP", "APF", "APF + vortex"]
planners = sb.multiselect(T("planners"), METHODS, default=DEFAULT_METHODS, help=T("planners_help"))
replan_mode = sb.radio(T("replan_mode"), ["every", "once"], format_func=lambda m: T("replan_" + m), horizontal=True)
interval = sb.slider(T("replan_interval"), 0.5, 5.0, 2.0, 0.5, disabled=replan_mode == "once")
guard = sb.checkbox(T("guard"), True, help=T("guard_help"))
with sb.expander(T("advanced")):
    t_max = st.slider(T("t_max"), 30, 300, 300, 10)
    margin = st.slider(T("margin"), 5, 60, 10, 1)

run_clicked = sb.button(T("run"), type="primary", width="stretch")

# ----------------------------------------------------------------------------- scenario
base_cfg = Config()
cfg = replace(base_cfg, sim=SimParams(dt=base_cfg.sim.dt, t_max=float(t_max)),
              planner=replace(base_cfg.planner, safety_margin=float(margin),
                              dp_clearance=float(min(base_cfg.planner.dp_clearance, margin))))
policy = PursuitPolicy(replan_interval=None if replan_mode == "once" else float(interval), guard=guard)
scenario_args = dict(zone_scale=float(zone_scale), n_zones=int(n_zones), target_mode=target_mode,
                     target_speed=float(target_speed), wind_speed=float(wind_speed),
                     wind_from_deg=float(wind_from), detect_radius=float(detect) if hidden else None)
base = make_scenario(seed, cfg, **scenario_args)

with st.expander(T("zones_title")):
    st.caption(T("zones_cols"))
    zones_df = pd.DataFrame([(o.x, o.y, o.r) for o in base.world.obstacles], columns=["x", "y", "r"]).round(1)
    edited = st.data_editor(zones_df, num_rows="dynamic", hide_index=True, width="stretch",
                            key=f"zones_{seed}_{n_zones}_{zone_scale}")

obstacles = [Obstacle(float(r.x), float(r.y), max(float(r.r), 1.0))
             for r in edited.itertuples() if pd.notna(r.x) and pd.notna(r.y) and pd.notna(r.r)]
scenario = replace(base, world=World(base.world.bounds, obstacles))

signature = (seed, tuple(sorted(scenario_args.items(), key=lambda kv: kv[0])), tuple(planners), replan_mode,
             interval, guard, t_max, margin, tuple((o.x, o.y, o.r) for o in obstacles))

# ----------------------------------------------------------------------------- run
if not planners:
    st.warning(T("pick_planner"))
    st.stop()

if run_clicked or "res" not in st.session_state:
    results = {}
    bar = st.progress(0.0, text=T("running"))
    for i, name in enumerate(planners):
        results.update(run_all(scenario, [name], policy, cfg, seed=seed))
        bar.progress((i + 1) / len(planners), text=T("running"))
    bar.empty()
    st.session_state["res"] = dict(results=results, scenario=scenario, cfg=cfg, signature=signature)

res = st.session_state["res"]
results, sc, rcfg = res["results"], res["scenario"], res["cfg"]
if res["signature"] != signature:
    st.info(T("stale"))

# a menu instead of st.tabs: Streamlit resets st.tabs to the first tab on every rerun, and a menu
# also means only the section being looked at gets drawn
SECTIONS = ["live", "metrics", "detail", "batch", "about"]
section = st.segmented_control("section", SECTIONS, default="live", key="section",
                               format_func=lambda k: T("tab_" + k), label_visibility="collapsed")
section = section or "live"

# ----------------------------------------------------------------------------- live
if section == "live":
    st.caption(T("live_hint"))
    st.plotly_chart(F.animation_figure(results, sc, rcfg, T), width="stretch",
                    config={"displayModeBar": False})

# ----------------------------------------------------------------------------- metrics
if section == "metrics":
    st.subheader(T("weights"))
    st.caption(T("weights_help"))
    cols = st.columns(4)
    keys = ["time_to_capture", "flown_length", "plan_ms_mean", "min_clearance"]
    labels = [T("w_time"), T("w_length"), T("w_plan"), T("w_clear")]
    weights = {k: c.slider(lb, 0.0, 1.0, DEFAULT_WEIGHTS[k], 0.05) for k, lb, c in zip(keys, labels, cols)}
    if sum(weights.values()) <= 0:
        weights = dict(DEFAULT_WEIGHTS)

    ms = [m for _, m in results.values()]
    order = rank(ms, weights)
    score = dict(order)
    winner = next(((n, s) for n, s in order if s is not None), None)
    if winner:
        st.success(T("winner", name=winner[0].replace("*", "\\*"), score=winner[1]))
    else:
        st.warning(T("no_winner"))

    by_name = {m.planner: m for m in ms}
    table = pd.DataFrame([{
        T("col_planner"): n,
        T("col_outcome"): T("out_" + outcome(by_name[n])),
        T("col_score"): s,
        T("m_time"): by_name[n].time_to_capture,
        T("m_length"): by_name[n].flown_length,
        T("m_plan_ms"): by_name[n].plan_ms_mean,
        T("m_clearance"): by_name[n].min_clearance,
        T("m_replans"): by_name[n].n_replans,
        T("m_waypoints"): by_name[n].mean_waypoints,
        T("m_roll"): by_name[n].max_roll_deg,
    } for n, s in order])
    fmt = {T("col_score"): "%.2f", T("m_time"): "%.1f", T("m_length"): "%.0f", T("m_plan_ms"): "%.1f",
           T("m_clearance"): "%.1f", T("m_waypoints"): "%.1f", T("m_roll"): "%.0f"}
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={k: st.column_config.NumberColumn(format=f) for k, f in fmt.items()})
    st.caption(T("metrics_note"))
    st.plotly_chart(F.bars_figure(ms, T), width="stretch", config={"displayModeBar": False})

# ----------------------------------------------------------------------------- detail
if section == "detail":
    name = st.selectbox(T("detail_planner"), list(results))
    log = results[name][0]
    n_re = len(log.replans)
    j = st.slider(T("detail_replan"), 1, n_re, 1) if n_re > 1 else 1
    ev = log.replans[j - 1]
    st.caption(T("detail_info", t=ev.t, reason=T("reason_" + ev.reason), n=len(ev.route), ms=ev.plan_ms))
    st.plotly_chart(F.route_figure(log, sc, rcfg.planner.safety_margin, j - 1, T), width="stretch",
                    config={"displayModeBar": False})
    st.plotly_chart(F.timeseries_figure(log, rcfg, T), width="stretch", config={"displayModeBar": False})

# ----------------------------------------------------------------------------- batch
if section == "batch":
    st.write(T("batch_intro"))
    n_batch = st.slider(T("batch_n"), 5, 40, 10)
    if st.button(T("batch_run")):
        rows = {n: [] for n in planners}
        bar = st.progress(0.0)
        for i in range(n_batch):
            s_i = make_scenario(seed + i, cfg, **scenario_args)
            for nm, (_, m) in run_all(s_i, planners, policy, cfg, seed=seed + i).items():
                rows[nm].append(m)
            bar.progress((i + 1) / n_batch, text=T("batch_progress", i=i + 1, n=n_batch))
        bar.empty()
        st.session_state["batch"] = rows
    rows = st.session_state.get("batch")
    if not rows:
        st.caption(T("batch_none"))
    else:
        summ = summarize(rows)
        st.dataframe(pd.DataFrame([{
            T("col_planner"): n,
            T("batch_table_success"): s["success_rate"],
            T("batch_table_collision"): s["collision_rate"],
            T("batch_table_left"): s["left_area_rate"],
            T("batch_table_time"): s["time_to_capture"],
            T("batch_table_length"): s["flown_length"],
            T("batch_table_plan"): s["plan_ms_mean"],
            T("batch_table_clear"): s["min_clearance"],
        } for n, s in summ.items()]), hide_index=True, width="stretch",
            column_config={T("batch_table_success"): st.column_config.NumberColumn(format="percent"),
                           T("batch_table_collision"): st.column_config.NumberColumn(format="percent"),
                           T("batch_table_left"): st.column_config.NumberColumn(format="percent"),
                           T("batch_table_time"): st.column_config.NumberColumn(format="%.1f"),
                           T("batch_table_length"): st.column_config.NumberColumn(format="%.0f"),
                           T("batch_table_plan"): st.column_config.NumberColumn(format="%.1f"),
                           T("batch_table_clear"): st.column_config.NumberColumn(format="%.1f")})
        st.caption(T("batch_note"))
        st.subheader(T("batch_outcomes"))
        st.plotly_chart(F.outcome_figure(rows, T), width="stretch", config={"displayModeBar": False})
        st.subheader(T("batch_dist"))
        st.plotly_chart(F.distribution_figure(rows, T), width="stretch", config={"displayModeBar": False})

# ----------------------------------------------------------------------------- about
if section == "about":
    v, ap, g = rcfg.vehicle, rcfg.approach, rcfg
    st.markdown(T("about_md", roll=f"{v.roll_max_deg:.0f}", vs=f"{v.v_stall:.0f}", vmax=f"{v.v_max:.0f}",
                  vc=f"{v.v_cruise:.0f}", margin=rcfg.planner.safety_margin, d_follow=ap.d_follow, d0=ap.d0,
                  capture=ap.capture_radius, hold=ap.hold_time))
