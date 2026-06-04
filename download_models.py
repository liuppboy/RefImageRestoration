from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download InsightFace models used by face_detail_transfer.py.")
    parser.add_argument("--models-dir", default="models/insightface", help="InsightFace model root.")
    parser.add_argument("--model-name", default="buffalo_l", help="InsightFace model pack.")
    parser.add_argument("--det-size", type=int, default=640, help="Detection size used for initialization.")
    args = parser.parse_args()

    try:
        from insightface.app import FaceAnalysis
    except ImportError as exc:
        raise SystemExit("Install dependencies first: python -m pip install -r requirements.txt") from exc

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    app = FaceAnalysis(name=args.model_name, root=str(models_dir), providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(args.det_size, args.det_size))

    expected_dir = models_dir / "models" / args.model_name
    print(f"Model pack ready: {expected_dir}")


if __name__ == "__main__":
    main()
