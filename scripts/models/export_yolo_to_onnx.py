#!/usr/bin/env python3
"""
Export YOLO model to ONNX format for C++ deployment.

Requirements (run in build container):
    pip install ultralytics

Usage:
    python scripts/models/export_yolo_to_onnx.py --model yolov10n
"""

from ultralytics import YOLO
import argparse

def main():
    parser = argparse.ArgumentParser(description="Export YOLO to ONNX")
    parser.add_argument("--model", default="yolov10n", help="YOLO model name")
    parser.add_argument("--output", default="data/models/", help="Output directory")
    args = parser.parse_args()

    print(f"Downloading and exporting {args.model} to ONNX...")

    model = YOLO(f"{args.model}.pt")
    model.export(
        format="onnx",
        dynamic=True,
        simplify=True,
        imgsz=640,
    )

    import shutil
    import os
    os.makedirs(args.output, exist_ok=True)
    shutil.move(f"{args.model}.onnx", f"{args.output}{args.model}.onnx")
    print(f"✓ Exported to {args.output}{args.model}.onnx")

if __name__ == "__main__":
    main()
