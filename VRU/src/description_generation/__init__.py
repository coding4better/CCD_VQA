"""
Description Generation Module

This module is responsible for generating video descriptions from QA pairs.

Available modules:
- data_loader_csv: Load and process QA pair data from CSV files
"""

from .data_loader_csv import QADataLoader, load_qa_data

__all__ = ['QADataLoader', 'load_qa_data']
