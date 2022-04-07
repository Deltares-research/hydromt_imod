"""hydroMT plugin for imod models."""

from os.path import dirname, join, abspath

__version__ = "0.0.5.dev"

try:
    import pcraster as pcr

    HAS_PCRASTER = True
except ImportError:
    HAS_PCRASTER = False

DATADIR = join(dirname(abspath(__file__)), "data")

from .imod import *
from .utils import *
