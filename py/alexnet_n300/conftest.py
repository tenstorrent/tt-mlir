import sys
from pathlib import Path

# Add the qwe directory to Python path so pytest can find local modules
sys.path.insert(0, str(Path(__file__).parent))
