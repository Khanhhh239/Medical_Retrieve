import os
import sys

# Cho phép `from src...` trong test mà không cần cài package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
