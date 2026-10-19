"""Experiments and figures for 8.3 (L1 guidance in wind) and the wind estimator.

    python scripts/demo_guidance.py                  # English figures -> docs/img
    python scripts/demo_guidance.py --lang tr --out some/dir

Writes the figures plus `guidance_results.json` with every number they show.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from uavguide.config import Config, WindParams
from uavguide.guidance import initial_roll_for_offset
from uavguide.planner import plan_route
from uavguide.sensors import AirDataOnly, NoisyGnssVelocity, TrueGroundVelocity
from uavguide.sim import constant_wind, follow_route, straight_line_test
from uavguide.simplify import douglas_peucker
from uavguide.theta_star import plan_route_theta
from uavguide.wind import WindEstimateSource, downwind_turn_radius
from uavguide.world import default_world

ROOT = Path(__file__).resolve().parents[1]
START, GOAL = (30.0, 30.0), (570.0, 570.0)
WIND_DIRS = [None] + list(range(0, 360, 45))   # None = no wind

# palette: categorical slots 1-3 + neutral reference, ink colours for text
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
REF, INK, INK2, GRID, OBST = "#8a8984", "#0b0b0b", "#52514e", "#e6e5e1", "#9b9a95"

LABELS = {
    "en": dict(
        t="time [s]", xte="cross-track error [m]", roll="roll [deg]", e="east [m]", n="north [m]",
        roll_cmd="commanded", roll_act="actual", wind="wind [m/s]",
        est_e="estimate, east", est_n="estimate, north", true_w="true wind",
        src_true="true ground velocity (reference)", src_gnss="noisy 5 Hz GNSS velocity",
        src_air="air data only (wind ignored)", src_est="air data + estimated wind",
        route_planned="planned route", flown="flown, 9 wind conditions",
        astar="A* + Douglas-Peucker", theta="Theta*", zoom="A* + DP, corner near (300, 330)",
        clr="min clearance to obstacle [m]", wind_from="wind from [deg]", none="none",
        collision="collision (< 0 m)", margin="planning margin (10 m)",
        off_title="15 m right of the track, no wind", cw_title="5 m/s crosswind from the left",
        roll_title="roll in crosswind (actual)", est_title="wind estimate, wind changes at t = 60 s",
        worst="worst"),
    "tr": dict(
        t="zaman [s]", xte="çapraz iz hatası [m]", roll="roll [°]", e="doğu [m]", n="kuzey [m]",
        roll_cmd="komut", roll_act="gerçek", wind="rüzgar [m/s]",
        est_e="kestirim, doğu", est_n="kestirim, kuzey", true_w="gerçek rüzgar",
        src_true="gerçek yer hızı (referans)", src_gnss="gürültülü 5 Hz GNSS hızı",
        src_air="yalnızca hava verisi (rüzgar yok sayılıyor)", src_est="hava verisi + kestirilen rüzgar",
        route_planned="planlanan rota", flown="uçulan iz, 9 rüzgar koşulu",
        astar="A* + Douglas-Peucker", theta="Theta*", zoom="A* + DP, (300, 330) yakınındaki köşe",
        clr="engele en küçük mesafe [m]", wind_from="rüzgarın geldiği yön [°]", none="yok",
        collision="çarpışma (< 0 m)", margin="planlama payı (10 m)",
        off_title="izin 15 m sağında, rüzgarsız", cw_title="soldan 5 m/s yan rüzgar",
        roll_title="yan rüzgarda roll (gerçek)", est_title="rüzgar kestirimi, rüzgar t = 60 s'de değişiyor",
        worst="en kötü"),
}


def style():
    plt.rcParams.update({
        "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
        "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "axes.titlesize": 9.5, "axes.titlecolor": INK, "lines.linewidth": 2.0,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    })


# ---------------------------------------------------------------- experiments
def exp_offset(cfg):
    out = {"speeds": {}}
    for v in (12.0, 20.0, 26.0):
        log = straight_line_test(15.0, cfg, v=v, t_max=60.0)
        x = log.xtrack_right
        inside = np.abs(x) < 1.0
        settle = float(log.t[np.argmax(inside & np.flip(np.cumprod(np.flip(inside))).astype(bool))])
        out["speeds"][str(int(v))] = dict(
            first_roll_cmd_deg=math.degrees(log.phi_cmd[0]), settle_1m_s=settle,
            overshoot_m=float(-x.min()), max_actual_roll_deg=math.degrees(np.abs(log.phi).max()))
    out["analytic_15m_deg"] = math.degrees(initial_roll_for_offset(15.0, cfg.guidance))
    out["analytic_5m_deg"] = math.degrees(initial_roll_for_offset(5.0, cfg.guidance))
    return out, straight_line_test(15.0, cfg, t_max=60.0)


def sources(cfg):
    return {
        "est": lambda: WindEstimateSource(cfg.wind_estimator, cfg.sensors, cfg.vehicle.v_cruise),
        "air": lambda: AirDataOnly(cfg.sensors),
        "gnss": lambda: NoisyGnssVelocity(cfg.sensors),
        "true": lambda: TrueGroundVelocity(),
    }


def exp_crosswind(cfg, seeds=range(10)):
    wind = constant_wind(cfg.wind.vector())
    stats, logs = {}, {}
    for key, make in sources(cfg).items():
        rows = []
        for seed in seeds:
            log = straight_line_test(0.0, cfg, wind=wind, source=make(), t_max=120.0, seed=seed)
            tail = log.t > 60.0
            x = log.xtrack_right[tail]
            rows.append([x.mean(), math.sqrt((x ** 2).mean()), np.abs(log.xtrack_right).max(),
                         math.degrees(log.phi[tail].std()), math.degrees(log.phi_cmd[tail].std())])
            if seed == 0:
                logs[key] = log
        r = np.mean(rows, axis=0)
        stats[key] = dict(steady_mean_m=r[0], steady_rms_m=r[1], max_abs_m=r[2],
                          roll_std_deg=r[3], roll_cmd_std_deg=r[4])
    beta = math.asin(cfg.wind.speed / cfg.vehicle.v_cruise)
    stats["predicted_air_only_m"] = cfg.guidance.l1_distance(cfg.vehicle.v_cruise) * math.sin(beta)
    stats["seeds"] = len(list(seeds))
    return stats, logs


def exp_estimator(cfg):
    w1, w2 = np.array([5.0, 0.0]), np.array([0.0, -6.0])
    src = WindEstimateSource(cfg.wind_estimator, cfg.sensors, cfg.vehicle.v_cruise)
    straight_line_test(0.0, cfg, wind=lambda t: w1 if t < 60.0 else w2, source=src, t_max=120.0)
    e = src.estimator
    T, W, S = np.array(e.history_t), np.array(e.history_w), np.array(e.history_sigma)
    truth = np.where((T < 60.0)[:, None], w1, w2)
    err = np.hypot(*(W - truth).T)
    def first_below(t0, lim):
        m = (T >= t0) & (err < lim)
        idx = np.flatnonzero(m)
        return float(T[idx[0]] - t0) if len(idx) else None
    res = dict(err_at_1s=float(err[np.searchsorted(T, 1.0)]), err_mean_10_60s=float(err[(T > 10) & (T < 60)].mean()),
               sigma_steady=float(S[(T > 30) & (T < 60)].mean()),
               recover_to_0_5_after_change_s=first_below(60.0, 0.5),
               err_mean_90_120s=float(err[T > 90].mean()))
    return res, (T, W, S, truth)


def routes(cfg, world):
    pc = cfg.planner
    safe = lambda a, b: world.segment_clear(a, b, pc.dp_clearance)
    a = plan_route(world, START, GOAL, pc.grid_res, pc.safety_margin).path
    dp, _ = douglas_peucker(a, pc.dp_epsilon, segment_ok=safe)
    th = plan_route_theta(world, START, GOAL, pc.grid_res, pc.safety_margin).path
    return {"astar_dp": dp, "theta": th}


def exp_routes(cfg, world):
    out, logs = {}, {}
    for name, route in routes(cfg, world).items():
        rows = []
        for wd in WIND_DIRS:
            wind = constant_wind((0.0, 0.0) if wd is None else WindParams(cfg.wind.speed, float(wd)).vector())
            log = follow_route(route, cfg, wind=wind, t_max=200.0)
            clr = log.obstacle_clearance(world)
            k = int(np.argmin(clr))
            rows.append(dict(wind_from=wd, finished=log.finished, time_s=float(log.t[-1]),
                             min_clearance_m=float(clr.min()), worst_at=[float(v) for v in log.pos[k]],
                             max_route_dev_m=float(log.route_distance(route).max()),
                             max_roll_deg=float(math.degrees(np.abs(log.phi).max()))))
            logs[(name, wd)] = log
        out[name] = dict(points=len(route), planned_clearance_m=float(world.path_clearance(route)),
                         runs=rows, worst_clearance_m=min(r["min_clearance_m"] for r in rows),
                         collisions=sum(r["min_clearance_m"] < 0 for r in rows))
    # the same Theta* route with different velocity sources in a 5 m/s west wind
    th = routes(cfg, world)["theta"]
    wind = constant_wind(cfg.wind.vector())
    src_rows = {}
    for key, make in sources(cfg).items():
        log = follow_route(th, cfg, wind=wind, source=make(), t_max=200.0)
        src_rows[key] = dict(min_clearance_m=float(log.obstacle_clearance(world).min()),
                             max_route_dev_m=float(log.route_distance(th).max()), finished=log.finished)
    out["theta_sources_west_wind"] = src_rows
    out["downwind_turn_radius_m"] = downwind_turn_radius(cfg.vehicle.v_cruise, cfg.wind.speed, cfg.vehicle.roll_max)
    out["still_air_turn_radius_m"] = cfg.vehicle.min_turn_radius()
    return out, logs


# ---------------------------------------------------------------- figures
def fig_offset(log, L, path):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 4.4), sharex=True)
    a1.axhline(0, color=REF, lw=1, ls="--")
    a1.plot(log.t, log.xtrack_right, color=C1)
    a1.set_ylabel(L["xte"]); a1.set_title(L["off_title"], loc="left")
    a2.plot(log.t, np.degrees(log.phi_cmd), color=C1, label=L["roll_cmd"])
    a2.plot(log.t, np.degrees(log.phi), color=C2, label=L["roll_act"])
    first = math.degrees(log.phi_cmd[0])
    a2.annotate(f"{first:.1f}°", (0, first), xytext=(8, -4), textcoords="offset points", color=INK, fontsize=8.5)
    a2.set_ylabel(L["roll"]); a2.set_xlabel(L["t"]); a2.legend(loc="lower right")
    a2.set_xlim(0, 40)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def fig_crosswind(logs, L, path):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 5.0), sharex=True, gridspec_kw=dict(height_ratios=[1.6, 1]))
    spec = [("true", REF, "--", L["src_true"]), ("air", C2, "-", L["src_air"]),
            ("gnss", C3, "-", L["src_gnss"]), ("est", C1, "-", L["src_est"])]
    for key, col, ls, lab in spec:
        lw = 1.2 if key in ("true", "gnss") else 2.0
        a1.plot(logs[key].t, logs[key].xtrack_right, color=col, ls=ls, lw=lw, label=lab)
    a1.set_ylabel(L["xte"]); a1.set_title(L["cw_title"], loc="left"); a1.legend(loc="center right", fontsize=8)
    for key, col in (("gnss", C3), ("est", C1)):
        a2.plot(logs[key].t, np.degrees(logs[key].phi), color=col, lw=1.2)
    a2.set_ylabel(L["roll"]); a2.set_xlabel(L["t"]); a2.set_title(L["roll_title"], loc="left")
    a2.set_ylim(-6, 6); a2.set_xlim(0, 120)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def fig_estimator(data, L, path):
    T, W, S, truth = data
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for i, (col, lab) in enumerate(((C1, L["est_e"]), (C2, L["est_n"]))):
        ax.fill_between(T, W[:, i] - 2 * S, W[:, i] + 2 * S, color=col, alpha=0.15, lw=0)
        ax.plot(T, W[:, i], color=col, lw=1.6, label=lab)
        ax.plot(T, truth[:, i], color=REF, ls="--", lw=1.2, label=L["true_w"] if i == 0 else None)
    ax.set_xlabel(L["t"]); ax.set_ylabel(L["wind"]); ax.set_title(L["est_title"], loc="left")
    ax.set_ylim(-8, 8); ax.legend(loc="lower left", fontsize=8, ncol=3)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def draw_obstacles(ax, world, margin):
    for o in world.obstacles:
        ax.add_patch(plt.Circle((o.x, o.y), o.r, color=OBST, zorder=1, lw=0))
        ax.add_patch(plt.Circle((o.x, o.y), o.r + margin, fill=False, ls=(0, (3, 3)), lw=0.7, color=REF, zorder=1))


def fig_routes(res, logs, rts, world, cfg, L, path):
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.9), gridspec_kw=dict(width_ratios=[1, 1, 1]))
    panels = [("astar_dp", axs[0], (0, 600, 0, 600), L["astar"], C2),
              ("astar_dp", axs[1], (250, 370, 290, 410), L["zoom"], C2),
              ("theta", axs[2], (0, 600, 0, 600), L["theta"], C1)]
    for name, ax, (x0, x1, y0, y1), title, col in panels:
        draw_obstacles(ax, world, cfg.planner.safety_margin)
        for wd in WIND_DIRS:
            p = logs[(name, wd)].pos
            ax.plot(p[:, 0], p[:, 1], color=col, lw=0.9, alpha=0.55, zorder=2)
        r = rts[name]
        ax.plot(r[:, 0], r[:, 1], color=INK, lw=1.2, marker="o", ms=3, zorder=3)
        worst = min(res[name]["runs"], key=lambda q: q["min_clearance_m"])
        wx, wy = worst["worst_at"]
        ax.plot(wx, wy, marker="X", ms=8, color=INK, zorder=4, mec="white", mew=1)
        if x1 - x0 < 300:
            ax.annotate(f"{L['worst']}: {worst['min_clearance_m']:.1f} m", (wx, wy), xytext=(-4, -24),
                        textcoords="offset points", fontsize=8.5, color=INK)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
        ax.set_title(title, loc="left"); ax.set_xlabel(L["e"])
        ax.grid(False)
    axs[0].set_ylabel(L["n"])
    axs[0].plot([], [], color=INK, lw=1.2, marker="o", ms=3, label=L["route_planned"])
    axs[0].plot([], [], color=C2, lw=0.9, label=L["flown"])
    axs[0].legend(loc="upper left", fontsize=7.5)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def fig_clearance(res, L, path):
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    labels = [L["none"] if wd is None else f"{wd}" for wd in WIND_DIRS]
    x = np.arange(len(labels))
    ax.axhspan(-6, 0, color="#f6e3e3", lw=0, zorder=0)
    ax.axhline(0, color=INK2, lw=1)
    ax.axhline(10, color=REF, lw=1, ls="--")
    ax.text(len(labels) - 0.6, 10.4, L["margin"], ha="right", va="bottom", fontsize=8, color=INK2)
    ax.text(len(labels) - 0.6, -5.6, L["collision"], ha="right", va="bottom", fontsize=8, color=INK2)
    for name, col, lab in (("theta", C1, L["theta"]), ("astar_dp", C2, L["astar"])):
        y = [r["min_clearance_m"] for r in res[name]["runs"]]
        ax.plot(x, y, color=col, marker="o", ms=6, lw=1.6, label=lab, mec="white", mew=1.5)
    ax.set_xticks(x, labels); ax.set_xlabel(L["wind_from"]); ax.set_ylabel(L["clr"])
    ax.set_ylim(-6, 16); ax.legend(loc="center left", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["en", "tr"], default="en")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "img")
    args = ap.parse_args()
    L = LABELS[args.lang]
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    style()

    cfg = Config()
    world = default_world()
    offset, off_log = exp_offset(cfg)
    cross, cross_logs = exp_crosswind(cfg)
    est, est_data = exp_estimator(cfg)
    rres, rlogs = exp_routes(cfg, world)

    fig_offset(off_log, L, out / "guidance_offset.png")
    fig_crosswind(cross_logs, L, out / "guidance_crosswind.png")
    fig_estimator(est_data, L, out / "wind_estimate.png")
    fig_routes(rres, rlogs, routes(cfg, world), world, cfg, L, out / "route_flown.png")
    fig_clearance(rres, L, out / "clearance_by_wind.png")

    results = dict(offset=offset, crosswind=cross, estimator=est, routes=rres)
    (out / "guidance_results.json").write_text(json.dumps(results, indent=1, default=float))
    print(json.dumps(results, indent=1, default=float))


if __name__ == "__main__":
    main()
