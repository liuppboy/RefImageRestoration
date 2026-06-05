# Source-Guided Face Detail Transfer

This tool transfers native high-frequency face detail from a Flux input image into a Flux I2I output image.
It does not use GFPGAN, CodeFormer, or another generative face restorer.

## Install

```powershell
python -m pip install -r requirements.txt
python download_models.py
```

`download_models.py` initializes InsightFace `buffalo_l`, which includes SCRFD face detection and 106-point landmarks.

## Single Image

```powershell
python face_detail_transfer.py `
  --source source.png `
  --target target.png `
  --out enhanced.png `
  --mask-out mask.png `
  --debug-dir debug_run `
  --det-size 1024 `
  --crop-det-size 512 `
  --work-size 512 `
  --crop-scale 2.4 `
  --fine-alpha 0.65 `
  --mid-alpha 0.30 `
  --detail-mode add `
  --fine-sigma 1.0 `
  --mid-sigma 3.5 `
  --fine-max-detail 18 `
  --mid-max-detail 18 `
  --sigma-scale-power 0.5 `
  --max-sigma-scale 2.0 `
  --min-crop-mean-diff 2.5 `
  --max-auto-detail-gain 2.0 `
  --mask-region-scale 0.85 `
  --mask-erode 8 `
  --mask-feather 36
```

The default ONNX Runtime provider is CPU. If you install a compatible GPU build, add:

```powershell
--providers CUDAExecutionProvider,CPUExecutionProvider
```

When CUDA is requested, the tool calls `onnxruntime.preload_dlls(directory="")` before InsightFace initializes, so
pip-installed CUDA/cuDNN runtime libraries can be found without manually editing `LD_LIBRARY_PATH`.

## Batch JSON

Batch mode reads source image paths from a JSON file, finds the target image with the same file name in `--target-dir`,
and writes PNG outputs to a directory. The output file name is based on the source image name, with the suffix changed
to `.png`.

Example JSON:

```json
[
  {
    "input_image": "source_a.jpg"
  },
  {
    "input_image": "source_b.png"
  }
]
```

Run:

```powershell
python face_detail_transfer.py `
  --input-json cases.json `
  --source-key input_image `
  --target-dir flux_outputs `
  --out-dir enhanced_outputs `
  --providers CUDAExecutionProvider,CPUExecutionProvider
```

Relative source image paths in the JSON are resolved relative to the JSON file. If a source path is
`inputs/source_a.jpg`, the target is read from `flux_outputs/source_a.jpg`, and the output is written to
`enhanced_outputs/source_a.png`.

If no face is detected in the source or target image, the tool writes the original target image to the output path
instead of failing the batch.

The JSON can also be:

```json
{
  "items": [
    {
      "source": "source_a.png"
    }
  ]
}
```

Legacy JSON files with both source and target paths are still supported by using `--target-key` instead of
`--target-dir`.

## Important Parameters

- `--input-json`: batch JSON path. Use with `--out-dir`.
- `--source-key`: source image key in each JSON item. Default: `source`.
- `--target-dir`: target image directory for batch mode. The target file name must match the source file name.
- `--target-key`: optional target image key for legacy batch JSON files.
- `--out-dir`: batch output directory. Outputs are named from source stem with `.png` suffix.
- `--det-size`: InsightFace detection resolution. The default is `1024`, which is more reliable for small faces.
- `--crop-det-size`: InsightFace detection resolution inside the fixed face crop. The default is `512`, which is more stable for close face crops.
- `--work-size`: fixed square face crop size used for transfer. The default is `512`.
- `--crop-scale`: square crop size relative to the detected face box before resizing to `--work-size`.
- `--fine-alpha`: fine texture transfer strength.
- `--mid-alpha`: mid-frequency structure transfer strength for eyes, lips, nose, and inner facial contours.
- `--detail-mode`: `add` transfers source detail on top of the target. `replace` moves target detail bands toward source detail bands.
- `--fine-sigma`: fine detail extraction sigma.
- `--mid-sigma`: mid-frequency extraction sigma.
- `--fine-max-detail`: clamps fine detail magnitude to reduce speckle and halos.
- `--mid-max-detail`: clamps mid detail magnitude to reduce ghosting.
- `--sigma-scale-power`: automatically increases detail extraction sigma when the enhanced 512 crop is pasted back into a smaller face.
- `--max-sigma-scale`: caps automatic small-face sigma scaling. Increase slightly only when small-face results are still too subtle.
- `--no-scale-aware-sigma`: disables automatic small-face sigma scaling.
- `--min-crop-mean-diff`: auto-boosts detail strength when the 512 working crop changes too little inside the mask. Use `0` to disable.
- `--max-auto-detail-gain`: caps the automatic alpha gain used by `--min-crop-mean-diff`.
- `--mask-region-scale`: expands the transfer mask with a small bbox-based face ellipse. Use `0` for landmark hull only.
- `--mask-erode`: shrinks the target face mask inward before fusion.
- `--mask-feather`: Gaussian soft-edge width. Larger means softer boundaries.
- `--mask-out`: writes the final soft confidence mask for debugging.
- `--debug-dir`: writes crop-level and final diff images plus numeric stats.

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

1. Detects the largest source and target face with InsightFace at `--det-size`.
2. Crops each face to a fixed 512 x 512 working image so small faces are processed at a stable scale.
3. Redetects landmarks inside the fixed crop at `--crop-det-size`.
4. Uses 106-point landmarks and Delaunay triangles to locally warp the source face into target coordinates.
5. Extracts fine and mid-frequency luminance detail from the warped source face.
6. Builds a conservative target-face mask with a high-weight interior and Gaussian soft edges.
7. Adds the source detail into the target LAB luminance channel through the soft face mask, then pastes the enhanced crop back.

The target color, lighting, and low-frequency shape are preserved. Only matched face detail is transferred.
If the face is extremely small in the final target image, the visible result is still limited by the final pixel count:
the 512 crop improves detection and transfer stability, but it cannot display pore-level detail inside a 30-pixel face.
For small faces, judge the effect by zooming in or writing `--mask-out`; at native size the change is intentionally subtle
to avoid sharpening halos and texture artifacts.
If the mask looks correct but the result is still too subtle, keep alpha unchanged first and try:

```powershell
--min-crop-mean-diff 4.0
```

This tells the tool to make the 512 working crop measurably different before pasting it back. If the face is small,
`--max-sigma-scale 2.5` can also help transfer details at a frequency that survives the final downsample.

When the mask looks right but the output appears unchanged, run with `--debug-dir debug_run` and inspect:

- `debug_run/enhanced_crop.png` versus `debug_run/target_crop.png`: verifies whether the 512 working face changed.
- `debug_run/crop_diff_x8.png`: visualizes crop-level changes.
- `debug_run/final_diff_x8.png`: visualizes changes after pasting back.
- `debug_run/stats.txt`: reports mean and max absolute differences inside the mask.

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
