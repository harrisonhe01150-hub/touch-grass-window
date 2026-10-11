import os
from pathlib import Path

# All local state (weather cache, your labels, results). TGW_DATA lets you point the app at another folder.
DATA = Path(os.environ.get("TGW_DATA", Path(__file__).resolve().parent.parent / "data"))
