"""Public full-family annotation entry point; see revision.annotation_entrypoint.

Preserves --run, --layer, --ontology-dir and feature_annotations_Lxx.json. The
underlying scorer requires the full observed gene universe and measures random floors.
"""
from pathlib import Path
import sys

# Also support direct script invocation, whose import path starts inside the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from revision.annotation_entrypoint import main


if __name__ == "__main__":
    main()
