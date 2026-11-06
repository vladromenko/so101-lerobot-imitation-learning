"""Local manipulation runtime using the installed SO-101 hardware stack."""
from pathlib import Path
import sys

# Only this application uses the read-only native hold-reconnect extension.
_native = Path(__file__).resolve().parents[1]/'runtime/native-hold/python'
if _native.is_dir():
    sys.path.insert(0,str(_native))
