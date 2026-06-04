import numpy as np

from face_detail_transfer import (
    build_feather_mask,
    extract_luminance_detail,
    fuse_luminance_detail,
)


def test_extract_luminance_detail_has_near_zero_mean_on_smooth_gradient():
    x = np.linspace(0, 255, 64, dtype=np.float32)
    gradient = np.tile(x, (64, 1))

    detail = extract_luminance_detail(gradient, radius=7)

    assert abs(float(detail.mean())) < 0.5


def test_fuse_luminance_detail_leaves_zero_mask_region_unchanged():
    target = np.full((32, 32), 128, dtype=np.float32)
    detail = np.full((32, 32), 40, dtype=np.float32)
    mask = np.zeros((32, 32), dtype=np.float32)

    fused = fuse_luminance_detail(target, detail, mask, alpha=0.5)

    np.testing.assert_allclose(fused, target)


def test_fuse_luminance_detail_applies_alpha_inside_mask():
    target = np.full((16, 16), 100, dtype=np.float32)
    detail = np.full((16, 16), 20, dtype=np.float32)
    mask = np.ones((16, 16), dtype=np.float32)

    fused = fuse_luminance_detail(target, detail, mask, alpha=0.25)

    np.testing.assert_allclose(fused, np.full((16, 16), 105, dtype=np.float32))


def test_fuse_luminance_detail_supports_detail_gain():
    target = np.full((16, 16), 100, dtype=np.float32)
    detail = np.full((16, 16), 10, dtype=np.float32)
    mask = np.ones((16, 16), dtype=np.float32)

    fused = fuse_luminance_detail(target, detail, mask, alpha=0.5, detail_gain=3.0, max_detail=40)

    np.testing.assert_allclose(fused, np.full((16, 16), 115, dtype=np.float32))


def test_fuse_luminance_detail_supports_mask_gamma():
    target = np.full((16, 16), 100, dtype=np.float32)
    detail = np.full((16, 16), 20, dtype=np.float32)
    mask = np.full((16, 16), 0.25, dtype=np.float32)

    fused = fuse_luminance_detail(target, detail, mask, alpha=1.0, mask_gamma=0.5)

    np.testing.assert_allclose(fused, np.full((16, 16), 110, dtype=np.float32))


def test_build_feather_mask_has_soft_interior_edge():
    binary = np.zeros((64, 64), dtype=np.uint8)
    binary[16:48, 16:48] = 255

    mask = build_feather_mask(binary, erode_px=4, feather_px=12)

    assert mask.dtype == np.float32
    assert mask.min() >= 0
    assert mask.max() <= 1
    assert mask[32, 32] > 0.95
    assert mask[16, 32] == 0
    assert 0 < mask[22, 32] < 1
