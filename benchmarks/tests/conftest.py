import sys
from pathlib import Path

# The benchmark scripts import each other as top-level modules (they run as scripts), so the tests put
# their folder on the path the same way.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
