"""Load the controller without importing the VLM dependency stack."""
import importlib.util
import sys
from pathlib import Path


def load_htm():
    path = Path(__file__).resolve().parents[2] / "qvic/model/htm.py"
    spec = importlib.util.spec_from_file_location("htm_diagnostics", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
