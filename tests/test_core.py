import numpy as np
import pytest

from stawarm.features import future_window_any, lead_slots, slots_since_last, trailing_sum
from stawarm.policies.baselines import FixedKeepAlive, HybridHistogram
from stawarm.policies.threshold import ThresholdPolicy, cost_threshold
from stawarm.sim.simulator import simulate_gaps, simulate_slots


def random_trace(n=40, T=600, seed=0):
    rng = np.random.default_rng(seed)
    rates = rng.choice([0.002, 0.02, 0.2], size=n)
    return (rng.random((n, T)) < rates[:, None]).astype(np.int32)


def test_hand_example_keepalive():
    x = np.zeros((1, 20), np.int32)
    x[0, [0, 3, 10]] = 1
    r = simulate_slots(x, FixedKeepAlive(5), np.array([1024.0]), 60.0)
    # slot 0 cold, slot 3 warm (within 5), slot 10 cold (gap 7 > 5)
    assert r.invoked[0] == 3 and r.cold[0] == 2
    # idle warm: 1,2 | 4..8 | 11..15  -> 2 + 5 + 5
    assert r.idle_warm[0] == 12


@pytest.mark.parametrize("k", [1, 3, 10, 20])
def test_slot_and_gap_simulators_agree(k):
    x = random_trace()
    mem = np.full(x.shape[0], 256.0)
    pol = FixedKeepAlive(k)
    a = simulate_slots(x, pol, mem, 60.0, warmup=50)
    pol.reset(x.shape[0], x.shape[1])
    b = simulate_gaps(x, pol.windows, mem, 60.0, warmup=50)
    assert np.array_equal(a.invoked, b.invoked)
    assert np.array_equal(a.cold, b.cold)
    assert np.array_equal(a.idle_warm, b.idle_warm)


def test_histogram_learns_periodic_function():
    T = 3000
    x = np.zeros((1, T), np.int32)
    x[0, ::30] = 1                                # every 30 minutes
    r = simulate_gaps(x, HybridHistogram().windows, np.array([256.0]), 60.0, warmup=1000)
    fixed = simulate_slots(x, FixedKeepAlive(10), np.array([256.0]), 60.0, warmup=1000)
    assert r.cold[0] == 0                          # pre-warms just before each call
    assert fixed.cold[0] == fixed.invoked[0]       # 10-minute keep-alive misses every call
    assert r.idle_warm[0] < 0.3 * (T - 1000)       # without keeping it warm all the time


def test_targets_and_features():
    a = np.array([[0, 0, 1, 0, 0, 0, 1, 0]], bool)
    y = future_window_any(a, np.array([2]), H=2)   # window t+2..t+3
    assert y[0].tolist() == [1, 0, 0, 1, 1, 0, 0, 0]
    assert slots_since_last(a, 99)[0].tolist() == [99, 99, 0, 1, 2, 3, 0, 1]
    assert trailing_sum(a, 3)[0].tolist() == [0, 0, 1, 1, 1, 0, 1, 1]
    assert lead_slots(np.array([0.3, 1.0, 2.5]), 1.0).tolist() == [1, 1, 3]


def test_cost_threshold_monotone():
    th = cost_threshold(mem_mb=np.array([128, 2048]), cold_start_s=np.array([2.0, 2.0]), hold_s=60,
                        price_per_gb_s=1e-5, penalty_per_s=1e-3)
    assert 0 < th[0] < th[1] < 1                   # bigger functions need more confidence


@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.2])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_miss_rate_bound_holds_for_adversarial_predictions(alpha, seed):
    """Proposition 1: miss rate <= alpha + (1 + eta) / (eta * P), for any prediction sequence."""
    rng = np.random.default_rng(seed)
    x = random_trace(n=30, T=4000, seed=seed)
    P = rng.random(x.shape) * 0.2                  # deliberately uninformative, badly calibrated
    eta = 0.05
    pol = ThresholdPolicy(P, theta_star=np.full(30, 0.9), lead=1, hold=1, keepalive_min=1,
                          alpha=alpha, eta=eta)
    r = simulate_slots(x, pol, np.full(30, 256.0), 60.0)
    for i in range(30):
        Pn = r.invoked[i]
        if Pn == 0:
            continue
        assert r.cold[i] / Pn <= alpha + (1 + eta) / (eta * Pn) + 1e-12


@pytest.mark.parametrize("seed", [0, 1])
def test_group_shared_threshold_bound_and_equivalence(seed):
    """A shared threshold per group keeps the group-level bound; one group per function = the per-function controller."""
    rng = np.random.default_rng(seed)
    x = random_trace(n=30, T=4000, seed=seed)
    P = rng.random(x.shape) * 0.2
    eta, alpha = 0.05, 0.05
    groups = np.arange(30) // 5
    pol = ThresholdPolicy(P, theta_star=np.full(30, 0.9), lead=1, hold=1, keepalive_min=1,
                          alpha=alpha, eta=eta, groups=groups)
    r = simulate_slots(x, pol, np.full(30, 256.0), 60.0)
    for g in range(6):
        m = groups == g
        Pg, Cg = r.invoked[m].sum(), r.cold[m].sum()
        if Pg:
            assert Cg / Pg <= alpha + (1 + eta * pol.max_step) / (eta * Pg) + 1e-12
    a = simulate_slots(x, ThresholdPolicy(P, 0.9, alpha=alpha, eta=eta), np.full(30, 256.0), 60.0)
    b = simulate_slots(x, ThresholdPolicy(P, 0.9, alpha=alpha, eta=eta, groups=np.arange(30)), np.full(30, 256.0), 60.0)
    assert np.array_equal(a.cold, b.cold) and np.array_equal(a.idle_warm, b.idle_warm)


def test_state_aware_and_realism_options():
    from stawarm.policies.threshold import StateAwarePolicy
    rng = np.random.default_rng(3)
    x = (rng.random((6, 3000)) < 0.01).astype(np.int8)
    P = rng.random((6, 3000)).astype(np.float32)
    kw = dict(lead=np.full(6, 3), hold=60, keepalive_min=60)
    a = simulate_slots(x, ThresholdPolicy(P, 0.5, **kw), np.full(6, 256.0), 1.0)
    b = simulate_slots(x, StateAwarePolicy(P, P, 0.5, **kw), np.full(6, 256.0), 1.0)
    assert np.array_equal(a.cold, b.cold) and np.array_equal(a.idle_warm, b.idle_warm)
    # defaults reproduce the original accounting; realism options can only cost more
    c = simulate_slots(x, ThresholdPolicy(P, 0.5, **kw), np.full(6, 256.0), 1.0, delay=0, charge_loading=False)
    assert np.array_equal(a.cold, c.cold) and np.array_equal(a.idle_warm, c.idle_warm)
    d = simulate_slots(x, ThresholdPolicy(P, 0.5, **kw), np.full(6, 256.0), 1.0, charge_loading=True)
    assert d.idle_warm.sum() >= a.idle_warm.sum()
