from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests
from skimage.metrics import structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from face_detail_transfer import (
    build_face_hull_mask,
    detect_largest_face,
    enhance_face_detail,
    extract_luminance_detail,
    load_insightface,
    read_image,
    write_image,
)

DEFAULT_SAMPLE_QUERIES = [
    "Eileen Collins portrait",
    "Mae Jemison portrait",
    "Sally Ride portrait",
]


def download_default_samples(samples_dir: Path) -> None:
    session = requests.Session()
    session.headers.update({"User-Agent": "RefImageRestorationEval/0.1"})
    samples_dir.mkdir(parents=True, exist_ok=True)
    for query in DEFAULT_SAMPLE_QUERIES:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", query.lower()) + ".jpg"
        out_path = samples_dir / safe_name
        if out_path.exists():
            continue
        search = session.get(
            "https://images-api.nasa.gov/search",
            params={"q": query, "media_type": "image"},
            timeout=60,
        )
        search.raise_for_status()
        image_url = None
        for item in search.json()["collection"]["items"][:8]:
            collection = session.get(item["href"], timeout=60)
            collection.raise_for_status()
            candidates = [
                url
                for url in collection.json()
                if isinstance(url, str)
                and re.search(r"~(?:orig|large|medium)\.jpg$|\.jpg$", url, re.IGNORECASE)
            ]
            if candidates:
                image_url = candidates[0]
                break
        if image_url is None:
            raise RuntimeError(f"No downloadable NASA image found for query: {query}")
        image = session.get(image_url, timeout=120)
        image.raise_for_status()
        out_path.write_bytes(image.content)
        print(f"downloaded {query}: {image_url}")
        time.sleep(1)


def center_face_crop(image: np.ndarray, bbox: np.ndarray, size: int = 768) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox.astype(np.float32)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    side = max(x2 - x1, y2 - y1) * 2.3
    side = min(side, w, h)
    x1 = int(round(np.clip(cx - side * 0.5, 0, w - side)))
    y1 = int(round(np.clip(cy - side * 0.5, 0, h - side)))
    crop = image[y1 : y1 + int(side), x1 : x1 + int(side)]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def make_target_gt(source: np.ndarray) -> np.ndarray:
    h, w = source.shape[:2]
    center = (w * 0.5, h * 0.5)
    matrix = cv2.getRotationMatrix2D(center, angle=1.8, scale=1.015)
    matrix[0, 2] += 5.0
    matrix[1, 2] -= 3.0
    return cv2.warpAffine(source, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)


def degrade_target(gt: np.ndarray) -> np.ndarray:
    h, w = gt.shape[:2]
    small = cv2.resize(gt, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    up = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(up, (0, 0), sigmaX=1.1, sigmaY=1.1)
    ok, encoded = cv2.imencode(".jpg", blurred, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def masked_values(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return image[mask > 0.15]


def face_bbox_from_mask(mask: np.ndarray) -> tuple[slice, slice]:
    ys, xs = np.where(mask > 0.15)
    y1, y2 = max(0, ys.min() - 16), min(mask.shape[0], ys.max() + 17)
    x1, x2 = max(0, xs.min() - 16), min(mask.shape[1], xs.max() + 17)
    return slice(y1, y2), slice(x1, x2)


def evaluate_pair(gt: np.ndarray, target: np.ndarray, enhanced: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    gt_lab = cv2.cvtColor(gt, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
    enhanced_lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB).astype(np.float32)
    gt_l = gt_lab[:, :, 0]
    target_l = target_lab[:, :, 0]
    enhanced_l = enhanced_lab[:, :, 0]

    roi = face_bbox_from_mask(mask)
    gt_roi = gt_l[roi]
    target_roi = target_l[roi]
    enhanced_roi = enhanced_l[roi]

    gt_detail = extract_luminance_detail(gt_l, sigma=1.6)
    target_detail = extract_luminance_detail(target_l, sigma=1.6)
    enhanced_detail = extract_luminance_detail(enhanced_l, sigma=1.6)
    m = mask > 0.25

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        av = a[m].astype(np.float64)
        bv = b[m].astype(np.float64)
        av -= av.mean()
        bv -= bv.mean()
        denom = np.sqrt(np.sum(av * av) * np.sum(bv * bv))
        return float(np.sum(av * bv) / denom) if denom > 1e-9 else 0.0

    return {
        "target_mae_l": float(np.mean(np.abs(masked_values(target_l - gt_l, mask)))),
        "enhanced_mae_l": float(np.mean(np.abs(masked_values(enhanced_l - gt_l, mask)))),
        "target_ssim_l": float(structural_similarity(gt_roi, target_roi, data_range=255)),
        "enhanced_ssim_l": float(structural_similarity(gt_roi, enhanced_roi, data_range=255)),
        "target_detail_corr": corr(target_detail, gt_detail),
        "enhanced_detail_corr": corr(enhanced_detail, gt_detail),
        "mask_mean": float(mask.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default="tests/samples", help="Directory with downloaded test images.")
    parser.add_argument("--out-dir", default="tests/eval_outputs", help="Output directory for generated cases.")
    parser.add_argument("--alpha", type=float, default=0.28)
    parser.add_argument("--detail-sigma", type=float, default=1.6)
    parser.add_argument("--max-detail", type=float, default=18.0)
    parser.add_argument("--mask-erode", type=int, default=8)
    parser.add_argument("--mask-feather", type=int, default=36)
    parser.add_argument("--no-download", action="store_true", help="Do not download default NASA samples.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = Path(args.samples)
    if not args.no_download and not list(samples_dir.glob("*.jpg")):
        download_default_samples(samples_dir)

    app = load_insightface("buffalo_l", "models/insightface", 640, ["CPUExecutionProvider"])

    rows = []
    for path in sorted(samples_dir.glob("*.jpg")):
        image = read_image(path)
        full_face = detect_largest_face(app, image)
        source = center_face_crop(image, full_face.bbox)
        source_face = detect_largest_face(app, source)
        gt = make_target_gt(source)
        target = degrade_target(gt)
        target_face = detect_largest_face(app, target)
        enhanced, mask = enhance_face_detail(
            source,
            target,
            source_face,
            target_face,
            alpha=args.alpha,
            detail_sigma=args.detail_sigma,
            max_detail=args.max_detail,
            mask_erode=args.mask_erode,
            mask_feather=args.mask_feather,
        )
        metrics = evaluate_pair(gt, target, enhanced, mask)
        stem = path.stem
        write_image(out_dir / f"{stem}_source.png", source)
        write_image(out_dir / f"{stem}_target.png", target)
        write_image(out_dir / f"{stem}_gt.png", gt)
        write_image(out_dir / f"{stem}_enhanced.png", enhanced)
        write_image(out_dir / f"{stem}_mask.png", (np.clip(mask, 0, 1) * 255).astype(np.uint8))
        rows.append({"sample": stem, **metrics})

    if not rows:
        raise RuntimeError(f"No .jpg samples found in {samples_dir}")

    csv_path = out_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            row["sample"],
            f"mae {row['target_mae_l']:.3f}->{row['enhanced_mae_l']:.3f}",
            f"ssim {row['target_ssim_l']:.4f}->{row['enhanced_ssim_l']:.4f}",
            f"detail_corr {row['target_detail_corr']:.4f}->{row['enhanced_detail_corr']:.4f}",
        )
    print(csv_path)


if __name__ == "__main__":
    main()
