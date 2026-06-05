import numpy as np

from face_detail_transfer import (
    FaceInfo,
    build_feather_mask,
    build_face_hull_mask,
    build_face_region_mask,
    build_plateau_blend_mask,
    crop_face_square,
    detect_largest_face_with_fallback,
    extract_luminance_detail,
    extract_multiband_details,
    fuse_multiband_luminance_detail,
    fuse_luminance_detail,
    paste_face_crop,
    scale_detail_sigmas_for_paste_size,
)


class FakeFace:
    def __init__(self, bbox):
        self.bbox = np.asarray(bbox, dtype=np.float32)
        self.landmark_2d_106 = np.zeros((106, 2), dtype=np.float32)


class FakeApp:
    def __init__(self, faces):
        self.faces = faces

    def get(self, image):
        return self.faces


def test_extract_luminance_detail_has_near_zero_mean_on_smooth_gradient():
    x = np.linspace(0, 255, 64, dtype=np.float32)
    gradient = np.tile(x, (64, 1))

    detail = extract_luminance_detail(gradient, sigma=1.6)

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


def test_fuse_luminance_detail_clamps_detail_without_extra_gain():
    target = np.full((16, 16), 100, dtype=np.float32)
    detail = np.full((16, 16), 50, dtype=np.float32)
    mask = np.ones((16, 16), dtype=np.float32)

    fused = fuse_luminance_detail(target, detail, mask, alpha=0.5, max_detail=12)

    np.testing.assert_allclose(fused, np.full((16, 16), 106, dtype=np.float32))


def test_extract_multiband_details_sum_matches_broad_detail():
    image = np.zeros((64, 64), dtype=np.float32)
    image[:, 32:] = 180
    image[24:40, 24:40] = 255

    fine, mid = extract_multiband_details(image, fine_sigma=1.0, mid_sigma=3.0)
    broad = extract_luminance_detail(image, sigma=3.0)

    np.testing.assert_allclose(fine + mid, broad, atol=1e-4)


def test_fuse_multiband_luminance_detail_weights_and_clamps_bands_independently():
    target = np.full((8, 8), 100, dtype=np.float32)
    fine = np.full((8, 8), 30, dtype=np.float32)
    mid = np.full((8, 8), 20, dtype=np.float32)
    mask = np.ones((8, 8), dtype=np.float32)

    fused = fuse_multiband_luminance_detail(
        target,
        fine,
        mid,
        mask,
        fine_alpha=0.5,
        mid_alpha=0.25,
        fine_max_detail=12,
        mid_max_detail=8,
    )

    np.testing.assert_allclose(fused, np.full((8, 8), 108, dtype=np.float32))


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


def test_build_plateau_blend_mask_keeps_most_interior_high_weight():
    binary = np.zeros((64, 64), dtype=np.uint8)
    binary[16:48, 16:48] = 255

    mask = build_plateau_blend_mask(binary, erode_px=4, feather_px=12)

    assert mask.dtype == np.float32
    assert mask.min() >= 0
    assert mask.max() <= 1
    assert mask[32, 32] > 0.95
    assert mask[24, 32] > 0.75
    assert 0 < mask[15, 32] < 0.5


def test_build_face_region_mask_expands_beyond_landmark_hull():
    landmarks = np.asarray(
        [
            [30, 30],
            [50, 30],
            [55, 50],
            [40, 62],
            [25, 50],
        ],
        dtype=np.float32,
    )
    face = FaceInfo(
        bbox=np.asarray([20, 18, 60, 66], dtype=np.float32),
        landmarks=landmarks,
    )

    hull = build_face_hull_mask(landmarks, (96, 96))
    region = build_face_region_mask(face, (96, 96))

    assert region.sum() > hull.sum()
    assert region[18, 40] > 0
    assert region[90, 90] == 0


def test_crop_face_square_returns_fixed_work_size_and_valid_box():
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    bbox = np.asarray([60, 40, 90, 70], dtype=np.float32)

    crop, box = crop_face_square(image, bbox, work_size=512, scale=2.0)

    assert crop.shape == (512, 512, 3)
    x1, y1, x2, y2 = box
    assert x2 - x1 == y2 - y1
    assert x1 <= 60 <= x2
    assert y1 <= 40 <= y2
    assert x1 >= 0
    assert y1 >= 0
    assert x2 <= image.shape[1]
    assert y2 <= image.shape[0]


def test_paste_face_crop_only_changes_masked_target_box():
    target = np.zeros((32, 32, 3), dtype=np.uint8)
    crop = np.full((8, 8, 3), 100, dtype=np.uint8)
    mask = np.ones((8, 8), dtype=np.float32)

    pasted = paste_face_crop(target, crop, mask, box=(8, 10, 16, 18))

    assert pasted[9, 12, 0] == 0
    assert pasted[10, 8, 0] == 100
    assert pasted[17, 15, 0] == 100
    assert pasted[18, 15, 0] == 0


def test_detect_largest_face_with_fallback_tries_next_detector():
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    face = FakeFace([1, 2, 9, 12])

    detected = detect_largest_face_with_fallback([FakeApp([]), FakeApp([face])], image)

    np.testing.assert_allclose(detected.bbox, face.bbox)


def test_scale_detail_sigmas_keeps_large_paste_size_unchanged():
    fine, mid, scale = scale_detail_sigmas_for_paste_size(
        fine_sigma=1.0,
        mid_sigma=3.5,
        work_size=512,
        target_box=(0, 0, 768, 768),
    )

    assert fine == 1.0
    assert mid == 3.5
    assert scale == 1.0


def test_scale_detail_sigmas_increases_for_small_paste_size():
    fine, mid, scale = scale_detail_sigmas_for_paste_size(
        fine_sigma=1.0,
        mid_sigma=3.5,
        work_size=512,
        target_box=(0, 0, 128, 128),
        power=0.5,
        max_scale=2.0,
    )

    assert scale == 2.0
    assert fine == 2.0
    assert mid == 7.0
