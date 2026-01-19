
import sys
from pathlib import Path

# Add project root to sys.path if not already present
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from matrix_factorization.cli import main

if __name__ == "__main__":
    main()
