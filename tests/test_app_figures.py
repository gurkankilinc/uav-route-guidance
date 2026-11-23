from dataclasses import replace

import pytest

pytest.importorskip("plotly")

import figures as F  # noqa: E402  (app/ is on the test path, see pyproject)
from i18n import TEXT  # noqa: E402

from uavguide.compare import run_all  # noqa: E402
from uavguide.config import Config, SimParams  # noqa: E402
from uavguide.metrics import outcome  # noqa: E402
from uavguide.scenarios import make_scenario  # noqa: E402

CFG = replace(Config(), sim=SimParams(t_max=40.0))
T = lambda k: TEXT["en"][k]


@pytest.fixture(scope="module")
def run():
    sc = make_scenario(2, CFG)
    return sc, run_all(sc, ["A* + DP", "Theta*"], cfg=CFG)


def test_translations_have_the_same_keys():
    assert set(TEXT["tr"]) == set(TEXT["en"])
    assert all(v.strip() for lang in TEXT.values() for v in lang.values())


def test_every_translation_key_used_by_the_app_exists():
    import re
    from pathlib import Path
    app = Path(__file__).resolve().parents[1] / "app"
    src = "".join((app / n).read_text(encoding="utf-8") for n in ("streamlit_app.py", "figures.py"))
    used = {k for k in re.findall(r'\bT\("([a-z0-9_]+)"', src) if not k.endswith("_")}   # "target_" + mode are dynamic
    used |= {f"target_{m}" for m in ("line", "circle", "fixed")} \
        | {f"replan_{m}" for m in ("every", "once")} | {f"out_{o}" for o in ("success", "collided", "left_area", "not_caught")} \
        | {f"reason_{r}" for r in ("initial", "periodic", "detected", "field")} \
        | {f"source_{r}" for r in ("true", "gnss", "wind_est", "air_only")}
    assert used <= set(TEXT["en"]), used - set(TEXT["en"])


def test_animation_has_one_frame_per_step_and_a_panel_per_planner(run):
    sc, res = run
    fig = F.animation_figure(res, sc, CFG, T, frames=30)
    assert len(fig.frames) == 30
    assert len(fig.layout.sliders[0].steps) == 30
    titles = [a.text for a in fig.layout.annotations]
    assert len(titles) == 2 and titles[0].startswith("A* + DP")
    # every frame updates the same dynamic traces (4 per panel)
    assert all(len(f.data) == len(fig.frames[0].data) == 8 for f in fig.frames)


def test_animation_frames_move_the_aircraft(run):
    sc, res = run
    fig = F.animation_figure(res, sc, CFG, T, frames=20)
    aircraft_idx = [j for j, tr in enumerate(fig.frames[0].data) if tr.marker is not None and tr.marker.symbol == "triangle-up"]
    first = fig.frames[0].data[aircraft_idx[0]]
    last = fig.frames[-1].data[aircraft_idx[0]]
    assert (first.x[0], first.y[0]) != (last.x[0], last.y[0])


def test_single_planner_animation_has_one_panel(run):
    sc, res = run
    fig = F.animation_figure({"Theta*": res["Theta*"]}, sc, CFG, T, frames=10)
    assert len(fig.layout.annotations) == 1


def test_detail_figures_build(run):
    sc, res = run
    log = res["Theta*"][0]
    assert len(F.route_figure(log, sc, 10.0, 0, T).data) >= 6
    assert len(F.route_figure(log, sc, 10.0, len(log.replans) - 1, T).data) >= 6
    ts = F.timeseries_figure(log, CFG, T)
    assert len(ts.data) == 6


def test_metric_figures_build(run):
    _, res = run
    ms = [m for _, m in res.values()]
    assert len(F.bars_figure(ms, T).data) == 4
    rows = {n: [m, m] for n, (_, m) in res.items()}
    assert len(F.outcome_figure(rows, T).data) == 4
    assert len(F.distribution_figure(rows, T).data) == 4


def test_status_text_covers_every_outcome(run):
    _, res = run
    for _, m in res.values():
        assert F.status_text(m, T)
    assert outcome(res["Theta*"][1]) in {"success", "collided", "left_area", "not_caught"}
