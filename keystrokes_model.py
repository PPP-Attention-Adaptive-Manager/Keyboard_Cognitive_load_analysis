"""Convenience wrapper to run the V2 sklearn baseline benchmark from repo root.

Usage:
  python keystrokes_model.py

This delegates to V2/training/keystrokes_model.py.
"""

from V2.training.keystrokes_model import main


if __name__ == "__main__":
    raise SystemExit(main())
