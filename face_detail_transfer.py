from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass
class FaceInfo:
    bbox: np.ndarray
    landmarks: np.ndarray


@dataclass
class BatchItem:
    source: Path
    target: Path


def read_image(path: str | Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def write_image(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise ValueError(f"Could not encode output image as {ext}: {path}")
    encoded.tofile(str(path))


def odd_kernel(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def extract_luminance_detail(l_channel: np.ndarray, sigma: float) -> np.ndarray:
    source = l_channel.astype(np.float32)
    sigma = max(0.1, float(sigma))
    kernel = odd_kernel(int(round(sigma * 6)) + 1)
    base = cv2.GaussianBlur(source, (kernel, kernel), sigmaX=sigma, sigmaY=sigma)
    return source - base


def gaussian_base(l_channel: np.ndarray, sigma: float) -> np.ndarray:
    source = l_channel.astype(np.float32)
    sigma = max(0.1, float(sigma))
    kernel = odd_kernel(int(round(sigma * 6)) + 1)
    return cv2.GaussianBlur(source, (kernel, kernel), sigmaX=sigma, sigmaY=sigma)


def extract_multiband_details(
    l_channel: np.ndarray,
    fine_sigma: float,
    mid_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    source = l_channel.astype(np.float32)
    fine_base = gaussian_base(source, fine_sigma)
    mid_base = gaussian_base(source, mid_sigma)
    fine_detail = source - fine_base
    mid_detail = fine_base - mid_base
    return fine_detail.astype(np.float32), mid_detail.astype(np.float32)


def fuse_luminance_detail(
    target_l: np.ndarray,
    detail: np.ndarray,
    mask: np.ndarray,
    alpha: float,
    max_detail: float = 24.0,
) -> np.ndarray:
    target = target_l.astype(np.float32)
    clipped_detail = np.clip(detail.astype(np.float32), -max_detail, max_detail)
    soft_mask = np.clip(mask.astype(np.float32), 0.0, 1.0)
    fused = target + float(alpha) * soft_mask * clipped_detail
    return np.clip(fused, 0, 255).astype(np.float32)


def fuse_multiband_luminance_detail(
    target_l: np.ndarray,
    fine_detail: np.ndarray,
    mid_detail: np.ndarray,
    mask: np.ndarray,
    fine_alpha: float,
    mid_alpha: float,
    fine_max_detail: float,
    mid_max_detail: float,
    target_fine_detail: np.ndarray | None = None,
    target_mid_detail: np.ndarray | None = None,
    mode: str = "add",
) -> np.ndarray:
    target = target_l.astype(np.float32)
    soft_mask = np.clip(mask.astype(np.float32), 0.0, 1.0)
    fine = np.clip(fine_detail.astype(np.float32), -fine_max_detail, fine_max_detail)
    mid = np.clip(mid_detail.astype(np.float32), -mid_max_detail, mid_max_detail)
    if mode == "replace":
        if target_fine_detail is None or target_mid_detail is None:
            raise ValueError("replace mode requires target fine and mid detail bands")
        target_fine = np.clip(target_fine_detail.astype(np.float32), -fine_max_detail, fine_max_detail)
        target_mid = np.clip(target_mid_detail.astype(np.float32), -mid_max_detail, mid_max_detail)
        fine = fine - target_fine
        mid = mid - target_mid
    elif mode != "add":
        raise ValueError(f"Unsupported detail mode: {mode}")

    fused = target + soft_mask * (float(fine_alpha) * fine + float(mid_alpha) * mid)
    return np.clip(fused, 0, 255).astype(np.float32)


def build_feather_mask(binary_mask: np.ndarray, erode_px: int, feather_px: int) -> np.ndarray:
    binary = (binary_mask > 0).astype(np.uint8)
    if erode_px > 0:
        k = odd_kernel(erode_px * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        binary = cv2.erode(binary, kernel, iterations=1)

    if feather_px <= 0:
        return binary.astype(np.float32)

    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    mask = np.clip(dist / float(feather_px), 0.0, 1.0)
    return mask.astype(np.float32)


def build_plateau_blend_mask(binary_mask: np.ndarray, erode_px: int, feather_px: int) -> np.ndarray:
    binary = (binary_mask > 0).astype(np.uint8)
    if erode_px > 0:
        k = odd_kernel(erode_px * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        binary = cv2.erode(binary, kernel, iterations=1)

    mask = binary.astype(np.float32)
    if feather_px <= 0:
        return mask

    sigma = max(0.1, float(feather_px) / 3.0)
    kernel = odd_kernel(int(round(sigma * 6)) + 1)
    mask = cv2.GaussianBlur(mask, (kernel, kernel), sigmaX=sigma, sigmaY=sigma)
    peak = float(mask.max())
    if peak > 1e-6:
        mask /= peak
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def preload_onnxruntime_cuda_dlls(providers: list[str]) -> None:
    if "CUDAExecutionProvider" not in providers:
        return

    try:
        import onnxruntime as ort
    except ImportError:
        return

    preload = getattr(ort, "preload_dlls", None)
    if callable(preload):
        preload(directory="")


def load_insightface(model_name: str, models_dir: str | Path, det_size: int, providers: list[str]):
    preload_onnxruntime_cuda_dlls(providers)
    try:
        from insightface.app import FaceAnalysis
    except ImportError as exc:
        raise RuntimeError(
            "insightface is not installed. Run: python -m pip install -r requirements.txt"
        ) from exc

    app = FaceAnalysis(name=model_name, root=str(models_dir), providers=providers)
    app.prepare(ctx_id=0 if providers and providers[0] != "CPUExecutionProvider" else -1, det_size=(det_size, det_size))
    return app


def select_largest_face(faces: Iterable[object]) -> object:
    faces = list(faces)
    if not faces:
        raise RuntimeError("No face detected.")

    def area(face: object) -> float:
        bbox = np.asarray(face.bbox, dtype=np.float32)
        return float(max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1]))

    return max(faces, key=area)


def face_to_info(face: object) -> FaceInfo:
    landmarks = None
    if hasattr(face, "landmark_2d_106"):
        landmarks = getattr(face, "landmark_2d_106")
    elif isinstance(face, dict) and "landmark_2d_106" in face:
        landmarks = face["landmark_2d_106"]

    if landmarks is None:
        raise RuntimeError(
            "InsightFace did not return 106-point landmarks. Use model pack 'buffalo_l'."
        )

    return FaceInfo(
        bbox=np.asarray(face.bbox, dtype=np.float32),
        landmarks=np.asarray(landmarks, dtype=np.float32).reshape(-1, 2),
    )


def detect_largest_face(app: object, image: np.ndarray) -> FaceInfo:
    face = select_largest_face(app.get(image))
    return face_to_info(face)


def detect_largest_face_with_fallback(apps: Iterable[object], image: np.ndarray) -> FaceInfo:
    last_error: RuntimeError | None = None
    for app in apps:
        try:
            return detect_largest_face(app, image)
        except RuntimeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError("No face detector was provided.")


def crop_face_square(
    image: np.ndarray,
    bbox: np.ndarray,
    work_size: int = 512,
    scale: float = 2.4,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox.astype(np.float32)
    cx = float((x1 + x2) * 0.5)
    cy = float((y1 + y2) * 0.5)
    face_side = max(float(x2 - x1), float(y2 - y1))
    side = int(round(max(8.0, face_side * float(scale))))
    side = min(side, w, h)

    left = int(round(cx - side * 0.5))
    top = int(round(cy - side * 0.5))
    left = max(0, min(left, w - side))
    top = max(0, min(top, h - side))
    right = left + side
    bottom = top + side

    crop = image[top:bottom, left:right]
    crop = cv2.resize(crop, (work_size, work_size), interpolation=cv2.INTER_CUBIC)
    return crop, (left, top, right, bottom)


def paste_face_crop(
    target: np.ndarray,
    enhanced_crop: np.ndarray,
    mask: np.ndarray,
    box: tuple[int, int, int, int],
) -> np.ndarray:
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        return target.copy()

    resized_crop = cv2.resize(enhanced_crop, (width, height), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    resized_mask = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    resized_mask = np.clip(resized_mask, 0.0, 1.0)[:, :, None]

    output = target.copy().astype(np.float32)
    roi = output[y1:y2, x1:x2]
    roi[:] = roi * (1.0 - resized_mask) + resized_crop * resized_mask
    return np.clip(output, 0, 255).astype(np.uint8)


def paste_mask(
    shape: tuple[int, int],
    mask: np.ndarray,
    box: tuple[int, int, int, int],
) -> np.ndarray:
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    full_mask = np.zeros(shape, dtype=np.float32)
    if width <= 0 or height <= 0:
        return full_mask
    full_mask[y1:y2, x1:x2] = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    return np.clip(full_mask, 0.0, 1.0)


def write_debug_image(path: Path, image: np.ndarray) -> None:
    write_image(path, image)


def absdiff_visual(a: np.ndarray, b: np.ndarray, gain: float = 8.0) -> np.ndarray:
    diff = cv2.absdiff(a, b).astype(np.float32)
    return np.clip(diff * float(gain), 0, 255).astype(np.uint8)


def mean_abs_diff_in_mask(a: np.ndarray, b: np.ndarray, mask: np.ndarray, threshold: float = 0.15) -> float:
    active = mask > threshold
    if not active.any():
        return 0.0
    diff = cv2.absdiff(a, b)
    return float(diff[active].mean())


def scale_detail_sigmas_for_paste_size(
    fine_sigma: float,
    mid_sigma: float,
    work_size: int,
    target_box: tuple[int, int, int, int],
    power: float = 0.5,
    max_scale: float = 2.0,
) -> tuple[float, float, float]:
    x1, y1, x2, y2 = target_box
    paste_side = max(x2 - x1, y2 - y1)
    if paste_side <= 0:
        return fine_sigma, mid_sigma, 1.0

    downsample = float(work_size) / float(paste_side)
    if downsample <= 1.0:
        return fine_sigma, mid_sigma, 1.0

    scale = min(float(max_scale), max(1.0, downsample ** float(power)))
    return fine_sigma * scale, mid_sigma * scale, scale


def expanded_bbox_points(bbox: np.ndarray, image_shape: tuple[int, int], scale: float) -> np.ndarray:
    h, w = image_shape[:2]
    x1, y1, x2, y2 = bbox.astype(np.float32)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    bw = (x2 - x1) * scale
    bh = (y2 - y1) * scale
    x1 = np.clip(cx - bw * 0.5, 0, w - 1)
    x2 = np.clip(cx + bw * 0.5, 0, w - 1)
    y1 = np.clip(cy - bh * 0.5, 0, h - 1)
    y2 = np.clip(cy + bh * 0.5, 0, h - 1)
    return np.asarray(
        [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
            [(x1 + x2) * 0.5, y1],
            [x2, (y1 + y2) * 0.5],
            [(x1 + x2) * 0.5, y2],
            [x1, (y1 + y2) * 0.5],
        ],
        dtype=np.float32,
    )


def build_correspondence_points(
    source_face: FaceInfo,
    target_face: FaceInfo,
    image_shape: tuple[int, int],
    bbox_scale: float = 1.08,
) -> tuple[np.ndarray, np.ndarray]:
    source_extra = expanded_bbox_points(source_face.bbox, image_shape, bbox_scale)
    target_extra = expanded_bbox_points(target_face.bbox, image_shape, bbox_scale)
    source_points = np.vstack([source_face.landmarks, source_extra]).astype(np.float32)
    target_points = np.vstack([target_face.landmarks, target_extra]).astype(np.float32)
    return source_points, target_points


def delaunay_triangles(points: np.ndarray, width: int, height: int) -> list[tuple[int, int, int]]:
    rect = (0, 0, width, height)
    subdiv = cv2.Subdiv2D(rect)
    for x, y in points:
        if 0 <= x < width and 0 <= y < height:
            subdiv.insert((float(x), float(y)))

    triangles: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for tri in subdiv.getTriangleList():
        coords = tri.reshape(3, 2)
        if np.any(coords[:, 0] < 0) or np.any(coords[:, 0] >= width):
            continue
        if np.any(coords[:, 1] < 0) or np.any(coords[:, 1] >= height):
            continue
        indices = []
        for vertex in coords:
            distances = np.linalg.norm(points - vertex, axis=1)
            idx = int(np.argmin(distances))
            if distances[idx] > 2.0:
                break
            indices.append(idx)
        if len(indices) != 3 or len(set(indices)) != 3:
            continue
        key = tuple(sorted(indices))
        if key not in seen:
            seen.add(key)
            triangles.append(tuple(indices))
    return triangles


def warp_piecewise_affine(
    source: np.ndarray,
    source_points: np.ndarray,
    target_points: np.ndarray,
    triangles: list[tuple[int, int, int]],
    output_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    height, width = output_shape[:2]
    warped = np.zeros((height, width, source.shape[2]), dtype=np.float32)
    coverage = np.zeros((height, width), dtype=np.uint8)
    source_f = source.astype(np.float32)

    for tri in triangles:
        src_tri = source_points[list(tri)].astype(np.float32)
        dst_tri = target_points[list(tri)].astype(np.float32)

        src_rect = cv2.boundingRect(src_tri)
        dst_rect = cv2.boundingRect(dst_tri)
        sx, sy, sw, sh = src_rect
        dx, dy, dw, dh = dst_rect
        if sw <= 1 or sh <= 1 or dw <= 1 or dh <= 1:
            continue

        src_rect_tri = src_tri - np.asarray([sx, sy], dtype=np.float32)
        dst_rect_tri = dst_tri - np.asarray([dx, dy], dtype=np.float32)

        src_crop = source_f[sy : sy + sh, sx : sx + sw]
        if src_crop.size == 0:
            continue

        affine = cv2.getAffineTransform(src_rect_tri, dst_rect_tri)
        warped_crop = cv2.warpAffine(
            src_crop,
            affine,
            (dw, dh),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        tri_mask = np.zeros((dh, dw), dtype=np.float32)
        cv2.fillConvexPoly(tri_mask, np.round(dst_rect_tri).astype(np.int32), 1.0, lineType=cv2.LINE_AA)

        x2 = min(dx + dw, width)
        y2 = min(dy + dh, height)
        crop_w = x2 - dx
        crop_h = y2 - dy
        if crop_w <= 0 or crop_h <= 0:
            continue

        roi = warped[dy:y2, dx:x2]
        mask_roi = tri_mask[:crop_h, :crop_w, None]
        roi[:] = roi * (1.0 - mask_roi) + warped_crop[:crop_h, :crop_w] * mask_roi
        coverage[dy:y2, dx:x2] = np.maximum(
            coverage[dy:y2, dx:x2],
            (tri_mask[:crop_h, :crop_w] * 255).astype(np.uint8),
        )

    return np.clip(warped, 0, 255).astype(np.uint8), coverage


def build_face_hull_mask(landmarks: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape[:2]
    hull = cv2.convexHull(landmarks.astype(np.float32)).astype(np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255, lineType=cv2.LINE_AA)
    return mask


def build_face_region_mask(face: FaceInfo, shape: tuple[int, int], bbox_scale: float = 0.85) -> np.ndarray:
    h, w = shape[:2]
    mask = build_face_hull_mask(face.landmarks, shape)
    if bbox_scale <= 0:
        return mask

    x1, y1, x2, y2 = face.bbox.astype(np.float32)
    cx = float((x1 + x2) * 0.5)
    cy = float((y1 + y2) * 0.5)
    bw = float((x2 - x1) * bbox_scale)
    bh = float((y2 - y1) * bbox_scale)
    center = (int(round(np.clip(cx, 0, w - 1))), int(round(np.clip(cy, 0, h - 1))))
    axes = (
        max(1, int(round(bw * 0.50))),
        max(1, int(round(bh * 0.58))),
    )
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, thickness=-1, lineType=cv2.LINE_AA)
    return mask


def enhance_face_detail(
    source_bgr: np.ndarray,
    target_bgr: np.ndarray,
    source_face: FaceInfo,
    target_face: FaceInfo,
    fine_alpha: float = 0.65,
    mid_alpha: float = 0.30,
    fine_sigma: float = 1.0,
    mid_sigma: float = 3.5,
    fine_max_detail: float = 18.0,
    mid_max_detail: float = 18.0,
    mask_erode: int = 8,
    mask_feather: int = 36,
    mask_region_scale: float = 0.85,
    detail_mode: str = "add",
) -> tuple[np.ndarray, np.ndarray]:
    if source_bgr.shape[:2] != target_bgr.shape[:2]:
        source_bgr = cv2.resize(source_bgr, (target_bgr.shape[1], target_bgr.shape[0]), interpolation=cv2.INTER_CUBIC)

    h, w = target_bgr.shape[:2]
    source_points, target_points = build_correspondence_points(source_face, target_face, target_bgr.shape)
    triangles = delaunay_triangles(target_points, w, h)
    if len(triangles) < 20:
        raise RuntimeError(f"Too few Delaunay triangles for reliable warp: {len(triangles)}")

    warped_source, coverage = warp_piecewise_affine(source_bgr, source_points, target_points, triangles, target_bgr.shape[:2])

    target_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    warped_lab = cv2.cvtColor(warped_source, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_l = target_lab[:, :, 0]
    warped_l = warped_lab[:, :, 0]

    face_mask_binary = build_face_region_mask(target_face, target_bgr.shape[:2], bbox_scale=mask_region_scale)
    face_mask_binary = cv2.bitwise_and(face_mask_binary, coverage)
    final_mask = build_plateau_blend_mask(face_mask_binary, erode_px=mask_erode, feather_px=mask_feather)

    fine_detail, mid_detail = extract_multiband_details(
        warped_l,
        fine_sigma=fine_sigma,
        mid_sigma=mid_sigma,
    )
    target_fine_detail = None
    target_mid_detail = None
    if detail_mode == "replace":
        target_fine_detail, target_mid_detail = extract_multiband_details(
            target_l,
            fine_sigma=fine_sigma,
            mid_sigma=mid_sigma,
        )

    target_lab[:, :, 0] = fuse_multiband_luminance_detail(
        target_l,
        fine_detail,
        mid_detail,
        final_mask,
        fine_alpha=fine_alpha,
        mid_alpha=mid_alpha,
        fine_max_detail=fine_max_detail,
        mid_max_detail=mid_max_detail,
        target_fine_detail=target_fine_detail,
        target_mid_detail=target_mid_detail,
        mode=detail_mode,
    )

    enhanced = cv2.cvtColor(np.clip(target_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return enhanced, final_mask


def enhance_image_face_detail(
    app: object,
    source_bgr: np.ndarray,
    target_bgr: np.ndarray,
    crop_app: object | None = None,
    work_size: int = 512,
    crop_scale: float = 2.4,
    fine_alpha: float = 0.65,
    mid_alpha: float = 0.30,
    fine_sigma: float = 1.0,
    mid_sigma: float = 3.5,
    fine_max_detail: float = 18.0,
    mid_max_detail: float = 18.0,
    mask_erode: int = 8,
    mask_feather: int = 36,
    mask_region_scale: float = 0.85,
    scale_aware_sigma: bool = True,
    sigma_scale_power: float = 0.5,
    max_sigma_scale: float = 2.0,
    detail_mode: str = "add",
    min_crop_mean_diff: float = 2.5,
    max_auto_detail_gain: float = 2.0,
    debug_dir: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    crop_app = crop_app or app
    full_detectors = [app] if crop_app is app else [app, crop_app]
    source_full_face = detect_largest_face_with_fallback(full_detectors, source_bgr)
    target_full_face = detect_largest_face_with_fallback(full_detectors, target_bgr)

    source_crop, _ = crop_face_square(source_bgr, source_full_face.bbox, work_size=work_size, scale=crop_scale)
    target_crop, target_box = crop_face_square(target_bgr, target_full_face.bbox, work_size=work_size, scale=crop_scale)

    source_crop_face = detect_largest_face(crop_app, source_crop)
    target_crop_face = detect_largest_face(crop_app, target_crop)
    effective_fine_sigma = fine_sigma
    effective_mid_sigma = mid_sigma
    if scale_aware_sigma:
        effective_fine_sigma, effective_mid_sigma, _ = scale_detail_sigmas_for_paste_size(
            fine_sigma,
            mid_sigma,
            work_size,
            target_box,
            power=sigma_scale_power,
            max_scale=max_sigma_scale,
        )

    def run_crop_enhance(alpha_scale: float) -> tuple[np.ndarray, np.ndarray]:
        return enhance_face_detail(
            source_crop,
            target_crop,
            source_crop_face,
            target_crop_face,
            fine_alpha=fine_alpha * alpha_scale,
            mid_alpha=mid_alpha * alpha_scale,
            fine_sigma=effective_fine_sigma,
            mid_sigma=effective_mid_sigma,
            fine_max_detail=fine_max_detail,
            mid_max_detail=mid_max_detail,
            mask_erode=mask_erode,
            mask_feather=mask_feather,
            mask_region_scale=mask_region_scale,
            detail_mode=detail_mode,
        )

    auto_detail_gain = 1.0
    enhanced_crop, crop_mask = run_crop_enhance(auto_detail_gain)
    initial_crop_mean_diff = mean_abs_diff_in_mask(enhanced_crop, target_crop, crop_mask)
    if min_crop_mean_diff > 0 and initial_crop_mean_diff < min_crop_mean_diff:
        denominator = max(initial_crop_mean_diff, 1e-6)
        auto_detail_gain = min(float(max_auto_detail_gain), float(min_crop_mean_diff) / denominator)
        if auto_detail_gain > 1.001:
            enhanced_crop, crop_mask = run_crop_enhance(auto_detail_gain)

    enhanced = paste_face_crop(target_bgr, enhanced_crop, crop_mask, target_box)
    full_mask = paste_mask(target_bgr.shape[:2], crop_mask, target_box)
    if debug_dir is not None:
        debug_path = Path(debug_dir)
        debug_path.mkdir(parents=True, exist_ok=True)
        write_debug_image(debug_path / "source_crop.png", source_crop)
        write_debug_image(debug_path / "target_crop.png", target_crop)
        write_debug_image(debug_path / "enhanced_crop.png", enhanced_crop)
        write_debug_image(debug_path / "crop_mask.png", (np.clip(crop_mask, 0, 1) * 255).astype(np.uint8))
        write_debug_image(debug_path / "crop_diff_x8.png", absdiff_visual(enhanced_crop, target_crop, gain=8))
        write_debug_image(debug_path / "enhanced.png", enhanced)
        write_debug_image(debug_path / "final_diff_x8.png", absdiff_visual(enhanced, target_bgr, gain=8))

        crop_diff = cv2.absdiff(enhanced_crop, target_crop)
        final_diff = cv2.absdiff(enhanced, target_bgr)
        crop_active = crop_mask > 0.15
        final_active = full_mask > 0.15
        stats = [
            f"detail_mode: {detail_mode}",
            f"target_box: {target_box}",
            f"effective_fine_sigma: {effective_fine_sigma:.4f}",
            f"effective_mid_sigma: {effective_mid_sigma:.4f}",
            f"initial_crop_mean_abs_diff_in_mask: {initial_crop_mean_diff:.4f}",
            f"auto_detail_gain: {auto_detail_gain:.4f}",
            f"crop_mask_pixels_gt_0.15: {int(crop_active.sum())}",
            f"crop_mask_pixels_gt_0.80: {int((crop_mask > 0.8).sum())}",
            f"final_mask_pixels_gt_0.15: {int(final_active.sum())}",
            f"crop_mean_abs_diff_in_mask: {float(crop_diff[crop_active].mean()) if crop_active.any() else 0.0:.4f}",
            f"crop_max_abs_diff: {int(crop_diff.max())}",
            f"final_mean_abs_diff_in_mask: {float(final_diff[final_active].mean()) if final_active.any() else 0.0:.4f}",
            f"final_max_abs_diff: {int(final_diff.max())}",
        ]
        (debug_path / "stats.txt").write_text("\n".join(stats) + "\n", encoding="utf-8")
    return enhanced, full_mask


def parse_providers(value: str) -> list[str]:
    providers = [part.strip() for part in value.split(",") if part.strip()]
    return providers or ["CPUExecutionProvider"]


def resolve_json_path(value: object, base_dir: Path, key: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Batch item key '{key}' must be a non-empty string path.")

    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path


def load_batch_items(json_path: str | Path, source_key: str, target_key: str) -> list[BatchItem]:
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "items" in data:
        data = data["items"]
    if not isinstance(data, list):
        raise ValueError("Batch JSON must be a list, or an object with an 'items' list.")

    base_dir = path.resolve().parent
    items: list[BatchItem] = []
    for index, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise ValueError(f"Batch item {index} must be an object.")
        if source_key not in entry:
            raise ValueError(f"Batch item {index} is missing source key '{source_key}'.")
        if target_key not in entry:
            raise ValueError(f"Batch item {index} is missing target key '{target_key}'.")
        items.append(
            BatchItem(
                source=resolve_json_path(entry[source_key], base_dir, source_key),
                target=resolve_json_path(entry[target_key], base_dir, target_key),
            )
        )

    return items


def output_path_for_source(source: str | Path, out_dir: str | Path) -> Path:
    return Path(out_dir) / f"{Path(source).stem}.png"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transfer native source face detail into a Flux I2I target face.")
    parser.add_argument("--source", default=None, help="Source/input image path for single-image mode.")
    parser.add_argument("--target", default=None, help="Flux output/target image path for single-image mode.")
    parser.add_argument("--out", default=None, help="Enhanced output image path for single-image mode.")
    parser.add_argument("--input-json", default=None, help="Batch JSON path. Use a list or an object with an 'items' list.")
    parser.add_argument("--source-key", default="source", help="Source image key in each batch JSON item.")
    parser.add_argument("--target-key", default="target", help="Target image key in each batch JSON item.")
    parser.add_argument("--out-dir", default=None, help="Batch output directory. Output names use source stem with .png suffix.")
    parser.add_argument("--mask-out", default=None, help="Optional debug mask output path.")
    parser.add_argument("--debug-dir", default=None, help="Optional directory for crop, diff, and stats debug outputs.")
    parser.add_argument("--models-dir", default="models/insightface", help="InsightFace model root.")
    parser.add_argument("--model-name", default="buffalo_l", help="InsightFace model pack name.")
    parser.add_argument("--providers", default="CPUExecutionProvider", help="ONNX Runtime providers.")
    parser.add_argument("--det-size", type=int, default=1024, help="InsightFace detection size.")
    parser.add_argument("--crop-det-size", type=int, default=512, help="InsightFace detection size for fixed face crops.")
    parser.add_argument("--work-size", type=int, default=512, help="Fixed face crop size used for detail transfer.")
    parser.add_argument("--crop-scale", type=float, default=2.4, help="Square crop scale relative to detected face bbox.")
    parser.add_argument("--fine-alpha", type=float, default=0.65, help="Fine texture transfer strength.")
    parser.add_argument("--mid-alpha", type=float, default=0.30, help="Mid-frequency structure transfer strength.")
    parser.add_argument("--detail-mode", choices=["add", "replace"], default="add", help="Add source detail on top, or replace target detail bands with source detail.")
    parser.add_argument("--fine-sigma", type=float, default=1.0, help="Fine detail Gaussian sigma.")
    parser.add_argument("--mid-sigma", type=float, default=3.5, help="Mid-frequency Gaussian sigma.")
    parser.add_argument("--fine-max-detail", type=float, default=18.0, help="Clamp fine detail magnitude in LAB-L units.")
    parser.add_argument("--mid-max-detail", type=float, default=18.0, help="Clamp mid detail magnitude in LAB-L units.")
    parser.add_argument("--no-scale-aware-sigma", action="store_true", help="Disable automatic sigma scaling for small pasted faces.")
    parser.add_argument("--sigma-scale-power", type=float, default=0.5, help="Small-face sigma scaling exponent.")
    parser.add_argument("--max-sigma-scale", type=float, default=2.0, help="Maximum automatic sigma scale for small faces.")
    parser.add_argument("--min-crop-mean-diff", type=float, default=2.5, help="Auto-boost detail until crop mean absolute diff in mask reaches this value. Use 0 to disable.")
    parser.add_argument("--max-auto-detail-gain", type=float, default=2.0, help="Maximum automatic alpha gain used by --min-crop-mean-diff.")
    parser.add_argument("--mask-erode", type=int, default=8, help="Shrink face mask inward before feathering.")
    parser.add_argument("--mask-feather", type=int, default=36, help="Gaussian soft-edge feather size.")
    parser.add_argument("--mask-region-scale", type=float, default=0.85, help="Optional bbox-ellipse face mask expansion. Use 0 for landmark hull only.")
    return parser


def enhance_one_from_paths(
    app: object,
    crop_app: object,
    source_path: str | Path,
    target_path: str | Path,
    output_path: str | Path,
    args: argparse.Namespace,
    mask_output_path: str | Path | None = None,
    debug_dir: str | Path | None = None,
) -> None:
    source = read_image(source_path)
    target = read_image(target_path)
    enhanced, mask = enhance_image_face_detail(
        app,
        source,
        target,
        crop_app=crop_app,
        work_size=args.work_size,
        crop_scale=args.crop_scale,
        fine_alpha=args.fine_alpha,
        mid_alpha=args.mid_alpha,
        fine_sigma=args.fine_sigma,
        mid_sigma=args.mid_sigma,
        fine_max_detail=args.fine_max_detail,
        mid_max_detail=args.mid_max_detail,
        mask_erode=args.mask_erode,
        mask_feather=args.mask_feather,
        mask_region_scale=args.mask_region_scale,
        scale_aware_sigma=not args.no_scale_aware_sigma,
        sigma_scale_power=args.sigma_scale_power,
        max_sigma_scale=args.max_sigma_scale,
        detail_mode=args.detail_mode,
        min_crop_mean_diff=args.min_crop_mean_diff,
        max_auto_detail_gain=args.max_auto_detail_gain,
        debug_dir=debug_dir,
    )
    write_image(output_path, enhanced)
    if mask_output_path:
        write_image(mask_output_path, (np.clip(mask, 0, 1) * 255).astype(np.uint8))


def validate_args(args: argparse.Namespace) -> None:
    if args.input_json:
        missing = [name for name in ["out_dir"] if getattr(args, name) is None]
        if missing:
            raise ValueError("--input-json requires --out-dir.")
        if args.mask_out:
            raise ValueError("--mask-out is only supported in single-image mode.")
        return

    missing = [name for name in ["source", "target", "out"] if getattr(args, name) is None]
    if missing:
        formatted = ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        raise ValueError(f"Single-image mode requires {formatted}.")


def main() -> None:
    args = build_arg_parser().parse_args()
    validate_args(args)

    providers = parse_providers(args.providers)
    app = load_insightface(args.model_name, args.models_dir, args.det_size, providers)
    crop_app = load_insightface(args.model_name, args.models_dir, args.crop_det_size, providers)

    if args.input_json:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        items = load_batch_items(args.input_json, args.source_key, args.target_key)
        for item in items:
            output_path = output_path_for_source(item.source, out_dir)
            debug_dir = None
            if args.debug_dir:
                debug_dir = Path(args.debug_dir) / item.source.stem
            enhance_one_from_paths(app, crop_app, item.source, item.target, output_path, args, debug_dir=debug_dir)
        return

    enhance_one_from_paths(
        app,
        crop_app,
        args.source,
        args.target,
        args.out,
        args,
        mask_output_path=args.mask_out,
        debug_dir=args.debug_dir,
    )


if __name__ == "__main__":
    main()
