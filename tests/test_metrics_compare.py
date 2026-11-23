import math
from dataclasses import replace

import pytest

from functools import partial

from uavguide.compare import mcnemar_p, monte_carlo, paired_difference, run_all, summarize, wilson_interval
from uavguide.config import Config, SimParams
from uavguide.metrics import Metrics, outcome, rank
from uavguide.planners import PLANNERS
from uavguide.pursuit import METHODS, PursuitPolicy
from uavguide.scenarios import TEST_SEEDS, TUNE_SEEDS, make_scenario, scale_zones
from uavguide.world import default_world


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
    assert list(res) == list(METHODS)
    for name, (log, m) in res.items():
        assert m.planner == name == log.planner
        assert m.flown_length > 0 and m.plan_ms_mean >= 0
        assert (m.n_replans > 0) == (name in PLANNERS)


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


def test_outcome_names_the_worst_problem_first():
    from dataclasses import replace as rep
    ok = fake("ok")
    assert outcome(ok) == "success"
    assert outcome(rep(ok, success=False, collided=True)) == "collided"
    assert outcome(rep(ok, success=False, collided=True, left_area=True)) == "collided"
    assert outcome(rep(ok, success=False, left_area=True)) == "left_area"
    assert outcome(rep(ok, success=False, captured=False)) == "not_caught"


def test_wilson_interval_matches_known_values():
    lo, hi = wilson_interval(16, 20)
    assert (lo, hi) == pytest.approx((0.584, 0.919), abs=2e-3)
    assert wilson_interval(0, 20)[0] == 0.0 and wilson_interval(20, 20)[1] == 1.0
    wide = wilson_interval(8, 10)
    narrow = wilson_interval(80, 100)
    assert (wide[1] - wide[0]) > 2 * (narrow[1] - narrow[0])      # ten times the runs: much tighter
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_mcnemar_exact_p_values():
    assert mcnemar_p(0, 0) == 1.0
    assert mcnemar_p(5, 0) == pytest.approx(0.0625)
    assert mcnemar_p(10, 0) == pytest.approx(2 / 1024)
    assert mcnemar_p(6, 6) == 1.0
    assert mcnemar_p(3, 9) == mcnemar_p(9, 3)


def test_paired_difference_counts_discordant_scenarios():
    a = [fake("a", success=s) for s in (True, True, True, True, False, True, True, True)]
    b = [fake("b", success=s) for s in (True, False, False, True, False, False, True, False)]
    d = paired_difference({"a": a, "b": b}, "a", "b")
    assert d["only_a"] == 4 and d["only_b"] == 0 and d["difference"] == pytest.approx(0.5)
    assert d["lo"] > 0.0 and d["p_value"] == pytest.approx(0.125)
    same = paired_difference({"a": a, "b": a}, "a", "b")
    assert same["difference"] == 0.0 and same["p_value"] == 1.0
    with pytest.raises(ValueError):
        paired_difference({"a": a, "b": b[:3]}, "a", "b")


def test_summary_reports_an_interval_and_guard_share():
    rows = {"x": [fake("x"), fake("x", success=False), fake("x"), fake("x")]}
    s = summarize(rows)["x"]
    assert s["success_rate"] == 0.75 and s["success_lo"] < 0.75 < s["success_hi"]
    assert s["left_area_rate"] == 0.0 and s["guard_share"] == 0.0


def test_parallel_monte_carlo_gives_the_same_rows():
    cfg = replace(Config(), sim=SimParams(t_max=20.0))
    make = partial(make_scenario, cfg=cfg)
    one = monte_carlo(range(2), make, ["Theta*"], PursuitPolicy(2.0), cfg, processes=1)
    two = monte_carlo(range(2), make, ["Theta*"], PursuitPolicy(2.0), cfg, processes=2)
    strip = lambda m: {k: v for k, v in m.as_dict().items() if not k.startswith("plan_ms")}
    for x, y in zip(one["Theta*"], two["Theta*"]):
        assert strip(x) == pytest.approx(strip(y), nan_ok=True)


def test_tuning_and_test_seeds_do_not_overlap():
    assert not set(TUNE_SEEDS) & set(TEST_SEEDS)
    assert not set(range(20)) & (set(TUNE_SEEDS) | set(TEST_SEEDS))    # the seeds v1 was developed on


def test_default_ranking_does_not_depend_on_wall_clock_time():
    a, b = fake("a", t=50.0, ms=1.0), fake("b", t=60.0, ms=500.0)
    slow_a = fake("a", t=50.0, ms=900.0)
    assert rank([a, b]) == rank([slow_a, b])


def test_guard_share_is_measured():
    cfg = replace(Config(), sim=SimParams(t_max=40.0))
    sc = make_scenario(2, cfg)
    on = run_all(sc, ["A*"], PursuitPolicy(2.0, guard=True), cfg)["A*"][1]
    off = run_all(sc, ["A*"], PursuitPolicy(2.0, guard=False), cfg)["A*"][1]
    assert off.guard_share == 0.0 and 0.0 <= on.guard_share <= 1.0
