from pathlib import Path
import runpy
import sys


SRC_DIR = Path(__file__).resolve().parent / "src"
SRC_APP = Path(__file__).resolve().parent / "src" / "app.py"
sys.path.insert(0, str(SRC_DIR))

runpy.run_path(str(SRC_APP), run_name="__main__")
