"""The attention layer should find the upstream function and its lag on a synthetic chain."""
import numpy as np
import torch

from stawarm.features import future_window_any
from stawarm.models.stawarm import STAWarm, focal_loss, make_batch, slot_inputs, sparsemax


def test_sparsemax_is_a_sparse_distribution():
    z = torch.tensor([[3.0, 1.0, 0.2, -2.0]])
    p = sparsemax(z)
    assert torch.allclose(p.sum(-1), torch.ones(1))
    assert (p == 0).sum() >= 1


def test_attention_recovers_upstream_function_and_lag():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    T, lag = 6000, 4
    up = (rng.random(T) < 0.03).astype(np.int32)        # upstream function A
    down = np.roll(up, lag)                              # B runs 4 slots after A
    down[:lag] = 0
    noise = (rng.random(T) < 0.03).astype(np.int32)      # unrelated function C
    counts = np.stack([down, up, noise])
    neighbours = [[(1, 1.0), (2, 0.5)], [], []]          # candidates for B: A and C
    z = slot_inputs(counts, 60.0)
    y = future_window_any(counts > 0, np.array([2, 2, 2]), H=1)   # target: B active at t+2
    S, K, L = 8, 2, 6
    ts = np.arange(S, T - 10)
    model = STAWarm(n_features=4, hidden=16, key_dim=8, max_lag=L, n_nodes=3, node_dim=4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for step in range(300):
        model.set_warmup(step < 150)       # softmax first, then sparsemax
        t = rng.choice(ts, 128)
        pairs = np.stack([np.zeros_like(t), t], 1)
        x, m, ids = make_batch(z, neighbours, pairs, S, K)
        logit, _ = model(x, m, ids=ids)
        loss = focal_loss(logit, torch.tensor(y[0, t], dtype=torch.float32))
        opt.zero_grad(); loss.backward(); opt.step()
    t = ts[:2000]
    pairs = np.stack([np.zeros_like(t), t], 1)
    x, m, ids = make_batch(z, neighbours, pairs, S, K)
    with torch.no_grad():
        logit, att = model(x, m, ids=ids)
    p = torch.sigmoid(logit).numpy()
    yy = y[0, t]
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(yy, p) > 0.95                       # it predicts B from A
    mass = att.mean(0).numpy()                               # (M, L+1)
    assert mass[1].sum() > 0.8                               # nearly all weight on A
    assert mass[2].sum() < 0.05                              # and almost none on C
    # B at t+2 means A at t+2-4 = t-2: the useful lag is 2
    assert int(mass[1].argmax()) == lag - 2
