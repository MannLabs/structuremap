#!python
"""This module provides generic utilities.
These utilities primarily focus on:
    - logging
"""

# builtin
import logging
import os
import sys

PROGRESS_CALLBACK = True

def set_logger(
    *,
    stream: bool = True,
    log_level: int = logging.INFO,
):
    """Set the log stream and file.
    All previously set handlers will be disabled with this command.
    Parameters
    ----------
    stream : bool
        If False, no log data is sent to stream.
        If True, all logging can be tracked with stdout stream.
        Default is True.
    log_level : int
        The logging level. Usable values are defined in Python's "logging"
        module.
        Default is logging.INFO.
    """
    import time
    global PROGRESS_CALLBACK
    root = logging.getLogger()
    formatter = logging.Formatter(
        '%(asctime)s> %(message)s', "%Y-%m-%d %H:%M:%S"
    )
    root.setLevel(log_level)
    while root.hasHandlers():
        root.removeHandler(root.handlers[0])
    if stream:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(log_level)
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)
