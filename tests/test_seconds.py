import numpy as np
import pandas as pd

from stawarm.seconds import FeatureStream, chunk_targets, events_to_active


def test_features_match_direct_computation_across_chunks():
    rng = np.random.default_rng(0)
    N, T = 4, 20000
    a = rng.random((N, T)) < 0.01
    app = np.array([0, 0, 1, 1])
    fs = FeatureStream(a, app, rate=np.ones(N), use_app=True, chunk=7000, hist=3600)
    got = {ch.lo: ch for ch in fs.chunks(start=7000)}
    ch = got[14000]
    t = 15000
    i = 1
    # own seconds since last call
    last = np.flatnonzero(a[i, : t + 1])[-1]
    assert np.isclose(np.expm1(ch.X[i, t - ch.lo, 0]), t - last, atol=1e-3)
    # own calls in the last 600 s
    k = ch.names.index("own_600s")
    assert np.isclose(np.expm1(ch.X[i, t - ch.lo, k]), a[i, t - 599: t + 1].sum(), atol=1e-3)
    # other functions of the same app in the last 60 s
    k = ch.names.index("app_60s")
    assert np.isclose(np.expm1(ch.X[i, t - ch.lo, k]), a[0, t - 59: t + 1].sum(), atol=1e-3)
    assert 0 not in got and 7000 in got    # chunks wholly before "start" are skipped


def test_targets_and_activity():
    ev = pd.DataFrame({"fid": [0, 1, 1], "app": [0, 0, 0], "start": [2.5, 5.1, 9.9]})
    a = events_to_active(ev)
    assert a.shape == (2, 10) and a[0, 2] and a[1, 5] and a[1, 9]
    y = chunk_targets(a, 0, 10, np.array([2, 2]), H=3)   # window t+2 .. t+4
    assert y[1].tolist() == [0, 1, 1, 1, 0, 1, 1, 1, 0, 0]
    u = chunk_targets(a, 0, 10, np.array([2, 2]), H=3, union=True)   # window t+1 .. t+4
    assert u[1].tolist() == [0, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    # union always contains the lead-shifted window
    assert np.all(u >= y)


def test_lead_feature_is_optional_and_constant_per_function():
    a = np.zeros((3, 8000), bool); a[:, ::97] = True
    base = next(FeatureStream(a, np.array([0, 0, 1]), np.ones(3), use_app=False, chunk=4000).chunks())
    assert "log_lead" not in base.names
    ch = next(FeatureStream(a, np.array([0, 0, 1]), np.ones(3), use_app=False, chunk=4000,
                            lead=np.array([1, 5, 40])).chunks())
    k = ch.names.index("log_lead")
    assert np.allclose(np.expm1(ch.X[:, :, k]), np.array([[1], [5], [40]]), atol=1e-4)
    assert len(ch.names) == len(base.names) + 1
