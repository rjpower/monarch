# pyre-strict
"""
Builtins for Monarch is a set of remote function defintions for PyTorch functions and other utilities.
"""

from .log import log_remote, set_logging_level_remote

__all__ = ["log_remote", "set_logging_level_remote"]
