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
  --alpha 0.25 `
  --detail-radius 3 `
  --mask-erode 12 `
  --mask-feather 32 `
  --sim-sigma 22
```

The default ONNX Runtime provider is CPU. If you install a compatible GPU build, add:

```powershell
--providers CUDAExecutionProvider,CPUExecutionProvider
```

## Important Parameters

- `--alpha`: detail transfer strength. Start with `0.18-0.30`.
- `--detail-radius`: high-frequency extraction radius. `3-5` is usually useful for face texture.
- `--max-detail`: clamps transferred detail to reduce ghosting and halos.
- `--mask-erode`: shrinks the target face mask inward before fusion.
- `--mask-feather`: distance-transform feather width. Larger means softer boundaries.
- `--sim-sigma`: low-frequency similarity gate. Lower values reject more mismatched regions.
- `--mask-out`: writes the final soft confidence mask for debugging.

## What It Does

1. Detects the largest source and target face with InsightFace.
2. Uses 106-point landmarks and Delaunay triangles to locally warp the source face into target coordinates.
3. Extracts only luminance high-frequency detail from the warped source face.
4. Builds a conservative target-face mask and multiplies it by a low-frequency similarity mask.
5. Adds the source detail into the target LAB luminance channel through the soft confidence mask.

The target color, lighting, and low-frequency shape are preserved. Only matched face detail is transferred.
