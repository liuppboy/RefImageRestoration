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
  --fine-alpha 0.65 `
  --mid-alpha 0.30 `
  --fine-sigma 1.0 `
  --mid-sigma 3.5 `
  --fine-max-detail 18 `
  --mid-max-detail 18 `
  --mask-erode 8 `
  --mask-feather 36
```

The default ONNX Runtime provider is CPU. If you install a compatible GPU build, add:

```powershell
--providers CUDAExecutionProvider,CPUExecutionProvider
```

## Important Parameters

- `--fine-alpha`: fine texture transfer strength.
- `--mid-alpha`: mid-frequency structure transfer strength for eyes, lips, nose, and inner facial contours.
- `--fine-sigma`: fine detail extraction sigma.
- `--mid-sigma`: mid-frequency extraction sigma.
- `--fine-max-detail`: clamps fine detail magnitude to reduce speckle and halos.
- `--mid-max-detail`: clamps mid detail magnitude to reduce ghosting.
- `--mask-erode`: shrinks the target face mask inward before fusion.
- `--mask-feather`: distance-transform feather width. Larger means softer boundaries.
- `--mask-out`: writes the final soft confidence mask for debugging.

For a lighter pass:

```powershell
python face_detail_transfer.py `
  --source source.png `
  --target target.png `
  --out enhanced_light.png `
  --mask-out mask_light.png `
  --fine-alpha 0.45 `
  --mid-alpha 0.22 `
  --fine-max-detail 16 `
  --mid-max-detail 14
```

## What It Does

1. Detects the largest source and target face with InsightFace.
2. Uses 106-point landmarks and Delaunay triangles to locally warp the source face into target coordinates.
3. Extracts fine and mid-frequency luminance detail from the warped source face.
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
| Eileen Collins portrait | 3.650 -> 2.534 | 0.9126 -> 0.9455 | 0.7630 -> 0.8722 |
| Mae Jemison portrait | 1.818 -> 1.465 | 0.9661 -> 0.9751 | 0.8076 -> 0.8935 |
| Sally Ride portrait | 2.657 -> 1.899 | 0.9460 -> 0.9650 | 0.7827 -> 0.8834 |

The downloaded samples and generated outputs are ignored by git.
