# Source-Guided Face Detail Transfer

This tool transfers native high-frequency face detail from a Flux input image into a Flux I2I output image.
It does not use GFPGAN, CodeFormer, or another generative face restorer.

## Install

```powershell
python -m pip install -r requirements.txt
python download_models.py
```

`download_models.py` initializes InsightFace `buffalo_l`, which includes SCRFD face detection and 106-point landmarks.

## Run

```powershell
python face_detail_transfer.py `
  --source source.png `
  --target target.png `
  --out enhanced.png `
  --mask-out mask.png `
  --alpha 0.28 `
  --detail-sigma 1.6 `
  --max-detail 18 `
  --mask-erode 8 `
  --mask-feather 36
```

The default ONNX Runtime provider is CPU. If you install a compatible GPU build, add:

```powershell
--providers CUDAExecutionProvider,CPUExecutionProvider
```

## Important Parameters

- `--alpha`: source detail transfer strength. Start with `0.20-0.35`.
- `--detail-sigma`: fine detail extraction sigma. `1.2-2.0` keeps the transfer focused on texture instead of face structure.
- `--max-detail`: clamps transferred detail to reduce ghosting and halos.
- `--mask-erode`: shrinks the target face mask inward before fusion.
- `--mask-feather`: distance-transform feather width. Larger means softer boundaries.
- `--mask-out`: writes the final soft confidence mask for debugging.

For a little more visible texture without turning into sharpening:

```powershell
python face_detail_transfer.py `
  --source source.png `
  --target target.png `
  --out enhanced_stronger.png `
  --mask-out mask_stronger.png `
  --alpha 0.35 `
  --detail-sigma 1.4 `
  --max-detail 22 `
  --mask-erode 6 `
  --mask-feather 32
```

## What It Does

1. Detects the largest source and target face with InsightFace.
2. Uses 106-point landmarks and Delaunay triangles to locally warp the source face into target coordinates.
3. Extracts only luminance high-frequency detail from the warped source face.
4. Builds a conservative target-face mask and softens it with distance-transform feathering.
5. Adds the source detail into the target LAB luminance channel through the soft face mask.

The target color, lighting, and low-frequency shape are preserved. Only matched face detail is transferred.

## Evaluation

The repository includes a reproducible evaluation helper:

```powershell
python scripts/evaluate_detail_transfer.py
```

If `tests/samples/` is empty, the script downloads default public NASA portrait samples first. It then creates a
ground-truth target from each source image, degrades it into a blurred target, runs the transfer, and writes metrics
to `tests/eval_outputs/metrics.csv`.

For the local NASA portrait samples used during development, the default settings improved all three checks:

| Sample | MAE L | SSIM L | Detail corr |
| --- | ---: | ---: | ---: |
| Eileen Collins portrait | 3.650 -> 2.976 | 0.9126 -> 0.9334 | 0.7630 -> 0.8402 |
| Mae Jemison portrait | 1.818 -> 1.468 | 0.9661 -> 0.9724 | 0.8076 -> 0.8645 |
| Sally Ride portrait | 2.657 -> 2.177 | 0.9460 -> 0.9583 | 0.7827 -> 0.8499 |

The downloaded samples and generated outputs are ignored by git.
