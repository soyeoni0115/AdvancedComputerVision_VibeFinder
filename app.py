from pathlib import Path
import sys


SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))

import app  # noqa: F401,E402
