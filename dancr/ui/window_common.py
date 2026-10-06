"""Constants and the logger shared by the window and its mixins."""
from __future__ import annotations

import logging

FILE_FILTER = "DANCR project (*.json)"
DATA_FILTER = "Data files (*.csv *.tsv *.txt *.dat *.xlsx *.xlsm *.xls *.parquet);;All files (*)"
log = logging.getLogger("dancr.ui")
