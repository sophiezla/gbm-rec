"""C5 parts (gbmc.c5). PyTorch tests skip in the main environment, which has no PyTorch."""

from __future__ import annotations

import numpy as np
import pytest

from gbmc import c5

TINY = c5.Settings(patch=(16, 16, 16), batch=2, epochs=1, steps_per_epoch=4,
                   levels=3, base_channels=4)


def _case(seed=0, shape=(24, 28, 20)):
    rng = np.random.default_rng(seed)
    evaluate = np.zeros(shape, bool)
    evaluate[2:-2, 2:-2, 2:-2] = True
    target = np.zeros(shape, bool)
    target[10:14, 12:16, 8:12] = True
    evaluate &= ~target
    label = np.zeros(shape, bool)
    label[9:15, 11:17, 7:13] = True
    label &= evaluate
    image = rng.standard_normal((5, *shape)).astype(np.float32)
    return c5.Case(f"P{seed}", image, evaluate, label, lr_axis=2)


def test_pack_keeps_cache_order():
    shape = (12, 10, 8)
    evaluate = np.zeros(shape, bool)
    evaluate[3:9, 2:8, 1:6] = True
    target = np.zeros(shape, bool)
    target[5:7, 4:6, 2:4] = True
    evaluate &= ~target
    truth = np.arange(evaluate.sum()) % 7 == 0
    zeros = np.zeros(shape, np.float32)
    packed = c5.pack("P", zeros, zeros, target, zeros.astype(bool), zeros, evaluate, truth, 2)
    assert packed["label"][packed["evaluate"]].tolist() == truth.tolist()
    assert packed["image"].shape[0] == len(c5.CHANNELS)


def test_pack_refuses_wrong_label_count():
    shape = (4, 4, 4)
    evaluate = np.ones(shape, bool)
    zeros = np.zeros(shape, np.float32)
    with pytest.raises(ValueError):
        c5.pack("P", zeros, zeros, zeros.astype(bool), zeros.astype(bool), zeros, evaluate,
                np.zeros(3, bool), 2)


def test_lr_axis_from_direction():
    assert c5.lr_axis_from_direction(np.eye(3).ravel()) == 2
    sagittal = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)  # index z runs left-right
    assert c5.lr_axis_from_direction(sagittal.ravel()) == 0


def test_half_the_patches_are_positive():
    cases = [_case(0), _case(1)]
    rng = np.random.default_rng(0)
    for step in range(5):
        _, y, _, _ = c5.sample_batch(cases, rng, TINY, step)
        assert y[0].any()


def test_window_starts_cover_the_volume():
    assert c5.window_starts(100, 96, 0.5) == [0, 4]
    assert c5.window_starts(96, 96, 0.5) == [0]
    assert c5.window_starts(200, 96, 0.5) == [0, 48, 96, 104]


def test_rotation_is_small_and_proper():
    rng = np.random.default_rng(1)
    for _ in range(20):
        r = c5.rotation_matrix(rng, 10.0)
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(r), 1.0)
        assert np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))) <= 10.0 + 1e-6


def test_pos_weight_is_capped():
    case = _case(0)
    ratio = (case.evaluate.sum() - case.label.sum()) / case.label.sum()
    assert c5.pos_weight([case], 1e9) == pytest.approx(ratio)
    assert c5.pos_weight([case], 5.0) == 5.0


def test_train_and_predict_are_deterministic():
    pytest.importorskip("torch")
    cases = [_case(0), _case(1)]
    scores = []
    for _ in range(2):
        model = c5.train(cases, TINY, seed=3, log=lambda m: None)
        scores.append(c5.predict(model, cases[0], TINY))
    assert scores[0].shape == (int(cases[0].evaluate.sum()),)
    assert np.all((scores[0] >= 0) & (scores[0] <= 1))
    assert np.array_equal(scores[0], scores[1])


def test_unet_keeps_the_patch_shape():
    torch = pytest.importorskip("torch")
    model = c5.build_unet(c5.Settings(levels=4, base_channels=4))
    out = model(torch.zeros((1, 5, 32, 32, 32)))
    assert tuple(out.shape) == (1, 32, 32, 32)
