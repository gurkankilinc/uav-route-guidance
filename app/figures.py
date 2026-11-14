"""Plotly figures for the Streamlit app. No Streamlit import here, so they are testable on their own."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from uavguide.metrics import Metrics, outcome

AIRCRAFT, TARGET, ROUTE = "#1f77b4", "#d62728", "#2ca02c"
ZONE_FILL, ZONE_RING, TAIL = "rgba(90,90,90,0.55)", "rgba(120,120,120,0.9)", "rgba(31,119,180,0.45)"
OUTCOME_COLOR = {"success": "#2ca02c", "collided": "#d62728", "left_area": "#ff7f0e", "not_caught": "#9e9e9e"}

Tr = Callable[[str], str]
_identity: Tr = lambda k: k


def _circles(world, grow: float = 0.0, n: int = 48) -> tuple[list, list]:
    """All zones as one polyline with None separators (one trace instead of one per zone)."""
    xs: list = []
    ys: list = []
    th = np.linspace(0.0, 2.0 * math.pi, n)
    for o in world.obstacles:
        xs += list(o.x + (o.r + grow) * np.cos(th)) + [None]
        ys += list(o.y + (o.r + grow) * np.sin(th)) + [None]
    return xs, ys


def _zone_traces(world, margin: float, T: Tr, show_legend: bool) -> list[go.Scatter]:
    zx, zy = _circles(world)
    rx, ry = _circles(world, margin)
    return [
        go.Scatter(x=rx, y=ry, mode="lines", line=dict(color=ZONE_RING, width=1, dash="dot"),
                   name=T("legend_margin"), hoverinfo="skip", showlegend=show_legend, legendgroup="margin"),
        go.Scatter(x=zx, y=zy, mode="lines", fill="toself", fillcolor=ZONE_FILL,
                   line=dict(color="rgba(60,60,60,0.9)", width=1), name=T("legend_zone"),
                   hoverinfo="skip", showlegend=show_legend, legendgroup="zone"),
    ]


def status_text(m: Metrics, T: Tr) -> str:
    o = outcome(m)
    if o == "success":
        return f"✓ {m.time_to_capture:.0f} s"
    return {"collided": "✖ " + T("out_collided"), "left_area": "✖ " + T("out_left_area"),
            "not_caught": "✖ " + T("out_not_caught")}[o]


def animation_figure(results: dict, scenario, cfg, T: Tr = _identity, frames: int = 220,
                     tail_s: float = 15.0) -> go.Figure:
    """One panel per planner, all driven by one time slider. `results`: name -> (PursuitLog, Metrics)."""
    names = list(results)
    n = len(names)
    cols = 2 if n > 1 else 1
    rows = math.ceil(n / cols)
    titles = [f"{nm}  {status_text(results[nm][1], T)}" for nm in names]
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles, horizontal_spacing=0.06,
                        vertical_spacing=0.10 if rows > 1 else 0.0)
    x0, y0, x1, y1 = scenario.world.bounds
    dt = cfg.sim.dt
    t_end = max(float(results[nm][0].t[-1]) for nm in names)
    times = np.linspace(0.0, t_end, max(2, frames))

    dyn: list[tuple[int, str, int]] = []      # (trace index, kind, panel)
    for i, nm in enumerate(names):
        log, _ = results[nm]
        r, c = divmod(i, cols)
        for tr in _zone_traces(scenario.world, cfg.planner.safety_margin, T, show_legend=(i == 0)):
            fig.add_trace(tr, row=r + 1, col=c + 1)
        step = max(1, len(log.t) // 600)
        fig.add_trace(go.Scatter(x=log.pos[::step, 0], y=log.pos[::step, 1], mode="lines",
                                 line=dict(color=TAIL, width=1), hoverinfo="skip", showlegend=False),
                      row=r + 1, col=c + 1)
        fig.add_trace(go.Scatter(x=log.target[::step, 0], y=log.target[::step, 1], mode="lines",
                                 line=dict(color="rgba(214,39,40,0.35)", width=1, dash="dot"),
                                 hoverinfo="skip", showlegend=False), row=r + 1, col=c + 1)
        fig.add_trace(go.Scatter(x=[log.pos[0, 0]], y=[log.pos[0, 1]], mode="markers",
                                 marker=dict(symbol="triangle-up", size=9, color="#2e7d32"),
                                 name=T("legend_start"), showlegend=(i == 0), hoverinfo="skip"),
                      row=r + 1, col=c + 1)
        for kind, kw, name in (
            ("tail", dict(mode="lines", line=dict(color=AIRCRAFT, width=3)), None),
            ("route", dict(mode="lines+markers", line=dict(color=ROUTE, width=2),
                           marker=dict(size=5, color=ROUTE)), T("legend_route")),
            ("aircraft", dict(mode="markers", marker=dict(symbol="triangle-up", size=13, color=AIRCRAFT,
                                                         line=dict(color="white", width=1))), T("legend_aircraft")),
            ("target", dict(mode="markers", marker=dict(symbol="star", size=14, color=TARGET,
                                                        line=dict(color="white", width=1))), T("legend_target")),
        ):
            fig.add_trace(go.Scatter(x=[None], y=[None], name=name or "", showlegend=bool(name) and i == 0,
                                     hoverinfo="skip", **kw), row=r + 1, col=c + 1)
            dyn.append((len(fig.data) - 1, kind, i))
        fig.update_xaxes(range=[x0, x1], row=r + 1, col=c + 1, showgrid=False, zeroline=False)
        fig.update_yaxes(range=[y0, y1], row=r + 1, col=c + 1, showgrid=False, zeroline=False,
                         scaleanchor=f"x{i + 1 if i else ''}", scaleratio=1)

    ev_times = {nm: np.array([e.t for e in results[nm][0].replans]) for nm in names}

    def frame_data(tk: float):
        out = []
        for _, kind, i in dyn:
            log = results[names[i]][0]
            k = min(int(round(tk / dt)), len(log.t) - 1)
            if kind == "aircraft":
                captured = log.captured and tk >= log.capture_t
                out.append(go.Scatter(x=[log.pos[k, 0]], y=[log.pos[k, 1]],
                                      marker=dict(symbol="triangle-up", size=13, line=dict(color="white", width=1),
                                                  color="#2e7d32" if captured else AIRCRAFT)))
            elif kind == "target":
                out.append(go.Scatter(x=[log.target[k, 0]], y=[log.target[k, 1]]))
            elif kind == "tail":
                k0 = max(0, k - int(tail_s / dt))
                out.append(go.Scatter(x=log.pos[k0:k + 1:5, 0], y=log.pos[k0:k + 1:5, 1]))
            else:
                j = int(np.searchsorted(ev_times[names[i]], log.t[k] + 1e-9, side="right")) - 1
                rt = log.replans[max(j, 0)].route
                out.append(go.Scatter(x=rt[:, 0], y=rt[:, 1]))
        return out

    idx = [d[0] for d in dyn]
    first = frame_data(0.0)
    for ti, d in zip(idx, first):
        fig.data[ti].update(d)
    fig.frames = [go.Frame(data=frame_data(tk), traces=idx, name=str(j)) for j, tk in enumerate(times)]
    fig.update_layout(
        height=430 * rows + 60, margin=dict(l=10, r=10, t=110, b=70),
        legend=dict(orientation="h", y=1.0 + 55.0 / (430 * rows + 60 - 180), yanchor="bottom", x=0.0),
        updatemenus=[dict(type="buttons", direction="left", x=0.0, y=-0.02 if rows == 1 else -0.01, xanchor="left", yanchor="top",
                          pad=dict(t=40, r=10),
                          buttons=[dict(label="▶", method="animate",
                                        args=[None, dict(frame=dict(duration=40, redraw=True), fromcurrent=True,
                                                         transition=dict(duration=0))]),
                                   dict(label="⏸", method="animate",
                                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate",
                                                           transition=dict(duration=0))])])],
        sliders=[dict(active=0, x=0.08, len=0.9, y=-0.02 if rows == 1 else -0.01, yanchor="top", pad=dict(t=30, b=0),
                      currentvalue=dict(prefix=T("slider_time") + ": ", suffix=" s"),
                      steps=[dict(method="animate", label=f"{tk:.0f}",
                                  args=[[str(j)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                                       transition=dict(duration=0))])
                             for j, tk in enumerate(times)])],
    )
    return fig


def route_figure(log, scenario, margin: float, replan_index: int, T: Tr = _identity) -> go.Figure:
    """Flown path with one chosen planned route highlighted, and the replan points along the path."""
    fig = go.Figure()
    for tr in _zone_traces(scenario.world, margin, T, True):
        fig.add_trace(tr)
    step = max(1, len(log.t) // 800)
    fig.add_trace(go.Scatter(x=log.pos[::step, 0], y=log.pos[::step, 1], mode="lines", name=T("legend_flown"),
                             line=dict(color=AIRCRAFT, width=2)))
    fig.add_trace(go.Scatter(x=log.target[::step, 0], y=log.target[::step, 1], mode="lines",
                             name=T("legend_target_path"), line=dict(color=TARGET, width=1, dash="dot")))
    pts = np.array([log.pos[min(int(round(e.t / (log.t[1] - log.t[0]))), len(log.t) - 1)] for e in log.replans])
    fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="markers", name=T("legend_replans"),
                             marker=dict(size=4, color="rgba(31,119,180,0.6)")))
    e = log.replans[replan_index]
    fig.add_trace(go.Scatter(x=e.route[:, 0], y=e.route[:, 1], mode="lines+markers", name=T("legend_route"),
                             line=dict(color=ROUTE, width=3), marker=dict(size=7)))
    k = min(int(round(e.t / (log.t[1] - log.t[0]))), len(log.t) - 1)
    fig.add_trace(go.Scatter(x=[log.pos[k, 0]], y=[log.pos[k, 1]], mode="markers", name=T("legend_aircraft"),
                             marker=dict(symbol="triangle-up", size=13, color=AIRCRAFT)))
    fig.add_trace(go.Scatter(x=[log.target[k, 0]], y=[log.target[k, 1]], mode="markers", name=T("legend_target"),
                             marker=dict(symbol="star", size=14, color=TARGET)))
    x0, y0, x1, y1 = scenario.world.bounds
    fig.update_xaxes(range=[x0, x1], showgrid=False, zeroline=False)
    fig.update_yaxes(range=[y0, y1], showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=-0.06))
    return fig


def timeseries_figure(log, cfg, T: Tr = _identity) -> go.Figure:
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=[T("ts_distance"), T("ts_speed"), T("ts_roll"), T("ts_clearance")])
    t = log.t
    fig.add_trace(go.Scatter(x=t, y=log.dist, name=T("ts_distance"), line=dict(color=AIRCRAFT)), row=1, col=1)
    fig.add_hline(y=cfg.approach.d_follow, line=dict(color="gray", dash="dash"), row=1, col=1)
    fig.add_hline(y=cfg.approach.capture_radius, line=dict(color=ROUTE, dash="dot"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=log.v, name=T("ts_airspeed"), line=dict(color=AIRCRAFT)), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=log.v_cmd, name=T("ts_command"), line=dict(color=TARGET, dash="dot")), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=np.degrees(log.phi), name=T("ts_roll"), line=dict(color=AIRCRAFT)), row=3, col=1)
    fig.add_trace(go.Scatter(x=t, y=np.degrees(log.phi_cmd), name=T("ts_roll_cmd"),
                             line=dict(color=TARGET, dash="dot", width=1)), row=3, col=1)
    clr = np.where(np.isfinite(log.clearance), log.clearance, np.nan)
    fig.add_trace(go.Scatter(x=t, y=clr, name=T("ts_clearance"), line=dict(color="#555")), row=4, col=1)
    fig.add_hline(y=0.0, line=dict(color=TARGET, dash="dash"), row=4, col=1)
    if log.captured:
        for r in range(1, 5):
            fig.add_vline(x=log.capture_t, line=dict(color=ROUTE, dash="dot"), row=r, col=1)
    fig.update_xaxes(title_text=T("ts_time"), row=4, col=1)
    fig.update_layout(height=760, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.08))
    return fig


def bars_figure(metrics: list[Metrics], T: Tr = _identity) -> go.Figure:
    specs = [("time_to_capture", T("m_time")), ("flown_length", T("m_length")),
             ("plan_ms_mean", T("m_plan_ms")), ("min_clearance", T("m_clearance"))]
    fig = make_subplots(rows=2, cols=2, subplot_titles=[s[1] for s in specs], vertical_spacing=0.18)
    names = [m.planner for m in metrics]
    colors = [OUTCOME_COLOR[outcome(m)] for m in metrics]
    for i, (key, _) in enumerate(specs):
        vals = [getattr(m, key) for m in metrics]
        fig.add_trace(go.Bar(x=names, y=vals, marker_color=colors, showlegend=False,
                             text=[("–" if (isinstance(v, float) and math.isnan(v)) else f"{v:.1f}") for v in vals],
                             textposition="outside"), row=i // 2 + 1, col=i % 2 + 1)
    fig.add_hline(y=0.0, line=dict(color="gray", width=1), row=2, col=2)
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def outcome_figure(rows: dict[str, list[Metrics]], T: Tr = _identity) -> go.Figure:
    """Stacked outcome shares per planner over a Monte Carlo batch."""
    fig = go.Figure()
    names = list(rows)
    for key in ("success", "collided", "left_area", "not_caught"):
        share = [100.0 * sum(outcome(m) == key for m in rows[nm]) / max(len(rows[nm]), 1) for nm in names]
        fig.add_trace(go.Bar(x=names, y=share, name=T("out_" + key), marker_color=OUTCOME_COLOR[key],
                             text=[f"{v:.0f}%" if v else "" for v in share], textposition="inside"))
    fig.update_layout(barmode="stack", height=360, yaxis_title="%", margin=dict(l=10, r=10, t=20, b=10),
                      legend=dict(orientation="h", y=-0.15))
    return fig


def distribution_figure(rows: dict[str, list[Metrics]], T: Tr = _identity) -> go.Figure:
    """Box plots of time to capture and flown length, successful runs only."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=[T("m_time"), T("m_length")])
    for i, key in enumerate(("time_to_capture", "flown_length")):
        for nm, ms in rows.items():
            vals = [getattr(m, key) for m in ms if outcome(m) == "success"]
            fig.add_trace(go.Box(y=vals, name=nm, boxpoints="all", jitter=0.4, pointpos=0, showlegend=False,
                                 marker=dict(size=4)), row=1, col=i + 1)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
    return fig
