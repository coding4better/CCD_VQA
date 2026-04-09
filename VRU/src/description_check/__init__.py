"""
Description consistency validation module.

Used to evaluate consistency between generated video descriptions and the source QA facts.
"""

__version__ = "1.0.0"
__author__ = "CCD-VQA Team"

from pathlib import Path

# Module paths
MODULE_DIR = Path(__file__).parent
RESULTS_DIR = MODULE_DIR / "results"

# Ensure the results directory exists
RESULTS_DIR.mkdir(exist_ok=True)

__all__ = [
    "MODULE_DIR",
    "RESULTS_DIR",
]
