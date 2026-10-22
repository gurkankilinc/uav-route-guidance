import math
from dataclasses import replace

import pytest

from uavguide.compare import monte_carlo, run_all, summarize
from uavguide.config import Config, SimParams
from uavguide.metrics import Metrics, rank
from uavguide.planners import PLANNERS
from uavguide.pursuit import PursuitPolicy
from uavguide.scenarios import make_scenario, scale_zones
from uavguide.world import default_world, random_world


def fake(name, success=True, t=60.0, length=1000.0, ms=10.0, clr=20.0):
    return Metrics(planner=name, captured=success, collided=False, left_area=False, success=success,
                   time_to_capture=t if success else math.nan, flown_length=length, min_clearance=clr,
                   planned_clearance=clr, n_replans=10, plan_ms_mean=ms, plan_ms_max=ms, nodes_total=100,
                   mean_waypoints=5.0, max_roll_deg=20.0, mean_abs_roll_deg=5.0,
                   track_rms_m=10.0 if success else math.nan)


def test_rank_orders_by_weighted_score_and_puts_failures_last():
    a = fake("fast", t=40.0, length=900.0, ms=50.0)
    b = fake("slow", t=90.0, length=1400.0, ms=5.0)
    c = fake("crashed", success=False)
    order = rank([c, b, a])
    assert [n for n, _ in order] == ["fast", "slow", "crashed"]
    assert order[-1][1] is None and order[0][1] < order[1][1]


def test_rank_weights_change_the_winner():
    a = fake("fast", t=40.0, ms=50.0)
    b = fake("light", t=90.0, ms=5.0)
    assert rank([a, b], {"time_to_capture": 1.0})[0][0] == "fast"
    assert rank([a, b], {"plan_ms_mean": 1.0})[0][0] == "light"


def test_rank_rejects_empty_weights_and_handles_a_single_survivor():
    with pytest.raises(ValueError):
        rank([fake("x")], {"time_to_capture": 0.0})
    assert rank([fake("only"), fake("dead", success=False)]) == [("only", 0.0), ("dead", None)]


def test_scale_zones_scales_radii_only():
    w = default_world()
    s = scale_zones(w, 1.5)
    assert [o.r for o in s.obstacles] == pytest.approx([o.r * 1.5 for o in w.obstacles])
    assert [(o.x, o.y) for o in s.obstacles] == [(o.x, o.y) for o in w.obstacles]


def test_scenarios_are_reproducible_and_start_free():
    a, b = make_scenario(7), make_scenario(7)
    assert a.world.obstacles == b.world.obstacles and a.wind == b.wind
    assert (a.target.state(5.0)[0] == b.target.state(5.0)[0]).all()
    big = make_scenario(7, zone_scale=2.0)
    assert big.world.clearance((big.start.x, big.start.y)) > Config().planner.safety_margin
    assert make_scenario(8).wind != a.wind


def test_scenario_wind_speed_and_direction():
    sc = make_scenario(1, wind_speed=8.0, wind_from_deg=270.0)
    assert sc.wind == pytest.approx((8.0, 0.0), abs=1e-9)        # from the west = blowing east


def test_run_all_returns_metrics_for_every_planner_on_the_same_scenario():
    cfg = replace(Config(), sim=SimParams(t_max=40.0))
    res = run_all(make_scenario(2), cfg=cfg)
    assert list(res) == list(PLANNERS)
    for name, (log, m) in res.items():
        assert m.planner == name == log.planner
        assert m.flown_length > 0 and m.n_replans > 0 and m.plan_ms_mean >= 0


def test_metrics_are_identical_between_repeated_runs_except_wall_time():
    cfg = replace(Config(), sim=SimParams(t_max=30.0))
    a = run_all(make_scenario(3), ["Theta*"], cfg=cfg)["Theta*"][1]
    b = run_all(make_scenario(3), ["Theta*"], cfg=cfg)["Theta*"][1]
    keys = set(a.as_dict()) - {"plan_ms_mean", "plan_ms_max"}
    assert {k: a.as_dict()[k] for k in keys} == pytest.approx({k: b.as_dict()[k] for k in keys}, nan_ok=True)


def test_monte_carlo_and_summary_shape():
    cfg = replace(Config(), sim=SimParams(t_max=30.0))
    rows = monte_carlo(range(2), lambda s: make_scenario(s, cfg), ["A* + DP", "Theta*"],
                       PursuitPolicy(2.0), cfg)
    assert set(rows) == {"A* + DP", "Theta*"} and all(len(v) == 2 for v in rows.values())
    s = summarize(rows)
    assert all(0.0 <= v["success_rate"] <= 1.0 and v["runs"] == 2 for v in s.values())
