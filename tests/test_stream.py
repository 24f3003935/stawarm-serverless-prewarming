import numpy as np
import torch

from stawarm.models.stawarm_stream import STAWarmStream, neighbour_table, train_and_predict_lstm


def test_neighbour_table_same_app_busiest_first():
    app = np.array([0, 0, 0, 1, -1])
    rate = np.array([5.0, 1.0, 9.0, 3.0, 2.0])
    nb, valid = neighbour_table(app, rate, k=2)
    assert nb[0].tolist() == [2, 1] and valid[0].all()
    assert nb[2].tolist() == [0, 1]
    assert not valid[3].any() and not valid[4].any()      # alone in its app / no app


def test_stream_model_is_causal_and_stateful():
    torch.manual_seed(0)
    N, T, F = 6, 40, 5
    nb, valid = neighbour_table(np.array([0, 0, 0, 1, 1, 2]), np.ones(6), k=3)
    nb, valid = torch.from_numpy(nb), torch.from_numpy(valid)
    m = STAWarmStream(F, N, attention=True).eval()
    x = torch.randn(N, T, F)
    full, _ = m(x, None, nb, valid)
    x2 = x.clone(); x2[:, 25:] += 5.0                        # change only the future
    part, _ = m(x2, None, nb, valid)
    assert torch.allclose(full[:, :25], part[:, :25], atol=1e-6)
    # running in two blocks with carried state = one pass
    a, st = m(x[:, :17], None, nb, valid)
    b, _ = m(x[:, 17:], st, nb, valid)
    assert torch.allclose(torch.cat([a, b], 1), full, atol=1e-5)
    # a function attends only to itself and its own app
    _, _, att = m(x, None, nb, valid, return_attention=True)
    assert torch.allclose(att[5, :, 0], torch.ones(T))      # function 5 is alone in app 2


def test_lstm_pipeline_shapes():
    rng = np.random.default_rng(0)
    N, T = 5, 3 * 21600
    active = rng.random((N, T)) < 0.003
    P, model, names = train_and_predict_lstm(active, np.array([0, 0, 1, 1, 1]), np.ones(N), np.ones(N, np.int64),
                                             60, 2 * 21600, 2 * 21600, attention=True, epochs=1,
                                             log=lambda *_: None)
    assert P.shape == (N, T - 2 * 21600) and np.isfinite(P.astype(float)).all()
    assert 0 <= P.min() and P.max() <= 1 and "own_log_since" in names and "app_10s" not in names


def test_full_trainer_shapes_and_early_stopping():
    from stawarm.models.stawarm_stream import train_and_predict_lstm_full
    rng = np.random.default_rng(1)
    N, T = 6, 3 * 21600
    active = rng.random((N, T)) < 0.004
    logs = []
    P, model, names = train_and_predict_lstm_full(active, np.array([0, 0, 1, 1, 1, 2]), np.ones(N), np.ones(N, np.int64),
                                                  60, 2 * 21600, 2 * 21600, attention=True, hidden=16, layers=2,
                                                  max_epochs=2, bptt=600, log=logs.append, device="cpu")
    assert P.shape == (N, T - 2 * 21600) and np.isfinite(P.astype(float)).all()
    assert "app_10s" in names                       # same inputs as the tree models
    assert any("validation" in m for m in logs)
