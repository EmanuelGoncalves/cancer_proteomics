#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import sys
import logging
import seaborn as sns
from crispy.CrispyPlot import CrispyPlot

__version__ = "0.0.1"

# - SET STYLE

sns.set(
    style="ticks",
    context="paper",
    font_scale=0.75,
    font="sans-serif",
    rc=CrispyPlot.SNS_RC,
)

# - Logging
__name__ = "cancer_proteomics"

logger = logging.getLogger(__name__)

if not logger.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("[%(asctime)s - %(levelname)s]: %(message)s"))
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)
    logger.propagate = False

# - HANDLES
__all__ = [
    "__version__",
    "logger",
]