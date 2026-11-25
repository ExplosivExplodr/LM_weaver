# Thin wrapper so the documented command keeps working from the repo root:
#   python3 trainer.py  [options]
# The implementation lives in src/lm_weaver/trainer.py
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from lm_weaver.trainer import main

if __name__ == "__main__":
    main()
