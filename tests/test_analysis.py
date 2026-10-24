import numpy as np
import pandas as pd

from stawarm.analysis import cold_prone_calls, early_signal_table


def _events(seed=0, follow=True):
    """App 'a': function A fires at random; function B fires 5 s after A (or independently)."""
    rng = np.random.default_rng(seed)
    T = 5 * 86400.0
    a = np.sort(rng.uniform(0, T, 400))
    b = a + 5.0 if follow else np.sort(rng.uniform(0, T, 400))
    return pd.DataFrame({"fid": ["A"] * len(a) + ["B"] * len(b), "app": "a", "start": np.concatenate([a, b])})


def test_cold_prone_calls():
    s = np.array([0.0, 10.0, 1000.0, 1005.0, 3000.0])
    assert cold_prone_calls(s, 600).tolist() == [0.0, 1000.0, 3000.0]


def test_lift_is_high_when_b_follows_a():
    ev = _events(follow=True)
    tab = early_signal_table(ev, pd.Series({"B": "rare"}), 86400, 5 * 86400, leads=(2,), windows=(10,), n_random=2000)
    assert tab.loc[0, "p_signal_at_call"] > 0.95
    assert tab.loc[0, "lift"] > 20


def test_lift_is_near_one_when_independent():
    ev = _events(follow=False)
    tab = early_signal_table(ev, pd.Series({"B": "rare"}), 86400, 5 * 86400, leads=(2,), windows=(300,), n_random=5000)
    assert 0.5 < tab.loc[0, "lift"] < 1.6


def test_signal_too_late_is_not_counted():
    ev = _events(follow=True)                 # B comes 5 s after A
    tab = early_signal_table(ev, pd.Series({"B": "rare"}), 86400, 5 * 86400, leads=(10,), windows=(60,), n_random=2000)
    assert tab.loc[0, "p_signal_at_call"] < 0.2  # a 10-second lead cannot use a 5-second hint


def test_controls_separate_early_warning_from_busy_periods():
    from stawarm.analysis import early_signal_controls
    rng = np.random.default_rng(3)
    T = 6 * 86400.0
    # busy hours: both functions only run between 09:00 and 10:00, independently of each other
    days = rng.integers(0, 6, 3000)
    a = np.sort(days * 86400 + 9 * 3600 + rng.uniform(0, 3600, 3000))
    days_b = rng.integers(0, 6, 60)
    b = np.sort(days_b * 86400 + 9 * 3600 + rng.uniform(0, 3600, 60))
    ev = pd.DataFrame({"fid": ["A"] * len(a) + ["B"] * len(b), "app": "x", "start": np.concatenate([a, b])})
    c = early_signal_controls(ev, pd.Series({"B": "rare"}), 86400, T, lead_s=2, window_s=60, gap_s=600)
    r = c.iloc[0]
    assert r.lift_uniform > 5            # naive baseline is fooled by the busy hour
    assert r.lift_app_active < 1.5       # the controlled baseline is not
    assert abs(r.p_after - r.p_before) < 0.15   # activity is symmetric: co-activity, not a warning


def test_controls_confirm_real_early_warning():
    from stawarm.analysis import early_signal_controls
    ev = _events(follow=True)            # B always 5 s after A
    c = early_signal_controls(ev, pd.Series({"B": "rare"}), 86400, 5 * 86400, lead_s=2, window_s=10, gap_s=600)
    r = c.iloc[0]
    assert r.p_before > 0.9 and r.p_after < 0.3
    assert r.lift_app_active > 2
    assert r.precision_any > 0.9         # every A call is followed by B...
    assert r.precision < r.precision_any # ...but only B calls after a long pause would have been cold
