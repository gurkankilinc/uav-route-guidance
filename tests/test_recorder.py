import json
from dataclasses import asdict, replace

import pytest

from uavguide.config import Config, GuardParams, PlannerParams, SimParams
from uavguide.pursuit import PursuitPolicy, pursue
from uavguide.recorder import NavRecorder, config_from_dict, load, replay
from uavguide.scenarios import make_scenario

CFG = replace(Config(), sim=SimParams(t_max=20.0))


def record(tmp_path, method, policy=None, scenario=None, cfg=CFG):
    rec = NavRecorder()
    pursue(scenario or make_scenario(2, cfg), method, policy or PursuitPolicy(2.0), cfg, recorder=rec)
    return rec, rec.save(tmp_path / "flight.jsonl")


@pytest.mark.parametrize("method", ["A* + DP", "Theta*", "APF + vortex"])
def test_replay_reproduces_every_command(tmp_path, method):
    rec, path = record(tmp_path, method)
    res = replay(path)
    assert res.ticks == len(rec.ticks) > 500
    assert res.identical and res.max_roll_diff == 0.0 and res.max_speed_diff == 0.0


def test_replay_is_exact_with_discovered_zones_and_plan_latency(tmp_path):
    sc = make_scenario(4, CFG, detect_radius=80.0)
    rec, path = record(tmp_path, "Theta* + DP", PursuitPolicy(1.0, plan_latency=0.4), sc)
    header, zones, ticks = load(path)
    assert len(zones) > 1                               # the known set grew during the flight
    assert header["plan_latency"] == 0.4
    assert replay(path).identical


def test_replay_at_a_lower_control_rate(tmp_path):
    rec, path = record(tmp_path, "Theta*", PursuitPolicy(2.0, control_rate_hz=10.0, command_delay=0.2))
    assert rec.ticks[1]["dt"] == pytest.approx(0.1) and len(rec.ticks) <= 201
    assert replay(path).identical


def test_another_method_on_the_same_inputs_differs(tmp_path):
    _, path = record(tmp_path, "Theta*")
    res = replay(path, method="APF")
    assert not res.identical and res.first_mismatch_t is not None and res.max_roll_diff > 0.01


def test_file_is_json_lines_with_header_first_and_zones_before_use(tmp_path):
    _, path = record(tmp_path, "A*")
    kinds = [json.loads(line)["type"] for line in path.read_text(encoding="utf-8").splitlines()]
    assert kinds[0] == "header" and kinds[1] == "zones" and set(kinds[2:]) == {"tick"}


def test_config_survives_the_trip_through_json():
    cfg = replace(Config(), guard=GuardParams(horizon=4.0), planner=PlannerParams(safety_margin=25.0))
    back = config_from_dict(json.loads(json.dumps(asdict(cfg))))
    assert back == cfg and isinstance(back.planner.area, tuple)


def test_recorded_configuration_is_used_for_the_replay(tmp_path):
    cfg = replace(CFG, guard=GuardParams(margin=20.0, horizon=8.0))
    _, path = record(tmp_path, "A*", cfg=cfg, scenario=make_scenario(2, cfg))
    assert replay(path).identical                       # would differ if the default guard were used
    assert not replay(path, cfg=CFG).identical


def test_missing_header_is_reported(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"type": "tick"}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        load(p)
