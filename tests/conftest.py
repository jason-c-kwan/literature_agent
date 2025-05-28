import sys
import os

# Add the project root directory to sys.path
# This ensures that modules like 'consts' and 'tools' can be imported directly
# when tests are run from any subdirectory or by various test runners.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
