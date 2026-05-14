from pathlib import Path
import runpy


ROOT_DIR = Path(__file__).resolve().parent

runpy.run_path(
    ROOT_DIR / "src" / "main.py",
    run_name="__main__"
)
