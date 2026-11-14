from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def _app():
    return AppTest.from_file(APP, default_timeout=300)


def _find(elements, label):
    return next(e for e in elements if e.label == label)


_slider = lambda at, label: _find(at.slider, label)


def test_app_runs_with_defaults_and_builds_every_tab():
    at = _app().run()
    assert not at.exception
    assert at.session_state["res"]["results"].keys() == {"A* + DP", "Theta* + DP", "APF", "APF + vortex"}
    for section in ("metrics", "detail", "batch", "about"):     # every section draws without errors
        at.session_state["section"] = section
        at.run()
        assert not at.exception, section
    assert not [d for d in at.dataframe if not str(d.key).startswith("zones_")]    # only the "about" page is showing
    assert len(at.markdown) > 0


def test_app_english_circle_target_plan_once_short_run():
    at = _app().run()
    _find(at.radio, "Dil / Language").set_value("en")                       # language
    at.run()
    _slider(at, "Maximum simulation time (s)").set_value(30)
    at.run()
    assert not at.exception
    _find(at.selectbox, "Target").set_value("Circle")
    _find(at.radio, "When is the route computed?").set_value("Plan once")
    _find(at.multiselect, "Algorithms to compare").set_value(["Theta*"])
    _find(at.button, "Run simulation").click()
    at.run()
    assert not at.exception
    res = at.session_state["res"]
    assert list(res["results"]) == ["Theta*"]
    assert len(res["results"]["Theta*"][0].replans) == 1


def test_app_warns_when_no_planner_is_selected():
    at = _app().run()
    _find(at.multiselect, "Karşılaştırılacak algoritmalar").set_value([])
    at.run()
    assert not at.exception
    assert len(at.warning) == 1
