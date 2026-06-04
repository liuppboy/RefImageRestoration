from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass
class FaceInfo:
    bbox: np.ndarray
    landmarks: np.ndarray


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


def load_insightface(model_name: str, models_dir: str | Path, det_size: int, providers: list[str]):
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


def enhance_face_detail(
    source_bgr: np.ndarray,
    target_bgr: np.ndarray,
    source_face: FaceInfo,
    target_face: FaceInfo,
    alpha: float = 0.28,
    detail_sigma: float = 1.6,
    max_detail: float = 18.0,
    mask_erode: int = 8,
    mask_feather: int = 36,
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

    face_mask_binary = build_face_hull_mask(target_face.landmarks, target_bgr.shape[:2])
    face_mask_binary = cv2.bitwise_and(face_mask_binary, coverage)
    final_mask = build_feather_mask(face_mask_binary, erode_px=mask_erode, feather_px=mask_feather)

    detail = extract_luminance_detail(warped_l, sigma=detail_sigma)
    target_lab[:, :, 0] = fuse_luminance_detail(
        target_l,
        detail,
        final_mask,
        alpha=alpha,
        max_detail=max_detail,
    )

    enhanced = cv2.cvtColor(np.clip(target_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return enhanced, final_mask


def parse_providers(value: str) -> list[str]:
    providers = [part.strip() for part in value.split(",") if part.strip()]
    return providers or ["CPUExecutionProvider"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transfer native source face detail into a Flux I2I target face.")
    parser.add_argument("--source", required=True, help="Source/input image path.")
    parser.add_argument("--target", required=True, help="Flux output/target image path.")
    parser.add_argument("--out", required=True, help="Enhanced output image path.")
    parser.add_argument("--mask-out", default=None, help="Optional debug mask output path.")
    parser.add_argument("--models-dir", default="models/insightface", help="InsightFace model root.")
    parser.add_argument("--model-name", default="buffalo_l", help="InsightFace model pack name.")
    parser.add_argument("--providers", default="CPUExecutionProvider", help="ONNX Runtime providers.")
    parser.add_argument("--det-size", type=int, default=640, help="InsightFace detection size.")
    parser.add_argument("--alpha", type=float, default=0.28, help="Simple source-detail transfer strength.")
    parser.add_argument("--detail-sigma", type=float, default=1.6, help="Gaussian sigma for fine detail extraction.")
    parser.add_argument("--max-detail", type=float, default=18.0, help="Clamp source detail magnitude in LAB-L units.")
    parser.add_argument("--mask-erode", type=int, default=8, help="Shrink face mask inward before feathering.")
    parser.add_argument("--mask-feather", type=int, default=36, help="Distance-transform feather size.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    source = read_image(args.source)
    target = read_image(args.target)
    if source.shape[:2] != target.shape[:2]:
        source_for_detection = cv2.resize(source, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_CUBIC)
    else:
        source_for_detection = source

    providers = parse_providers(args.providers)
    app = load_insightface(args.model_name, args.models_dir, args.det_size, providers)
    source_face = detect_largest_face(app, source_for_detection)
    target_face = detect_largest_face(app, target)

    enhanced, mask = enhance_face_detail(
        source_for_detection,
        target,
        source_face,
        target_face,
        alpha=args.alpha,
        detail_sigma=args.detail_sigma,
        max_detail=args.max_detail,
        mask_erode=args.mask_erode,
        mask_feather=args.mask_feather,
    )
    write_image(args.out, enhanced)
    if args.mask_out:
        write_image(args.mask_out, (np.clip(mask, 0, 1) * 255).astype(np.uint8))


if __name__ == "__main__":
    main()
