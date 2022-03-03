"""hydroMT plugin for imod models."""

from os.path import dirname, join, abspath

__version__ = "0.0.1.dev"

DATADIR = join(dirname(abspath(__file__)), "data")

from .imod import *
