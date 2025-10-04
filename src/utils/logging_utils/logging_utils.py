"""Logging Utility Module."""

import os
import sys
import time
from typing import Literal, Optional
import logging
import tqdm


class TqdmLoggingHandler(logging.Handler):
    """Custom handler to log messages without breaking tqdm progress bars."""

    def __init__(self, level=logging.NOTSET):
        """Initialize the TqdmLoggingHandler.

        Args:
            level (int, optional): The logging level for this handler.
                Defaults to logging.NOTSET.
        """
        super().__init__(level)

    def emit(self, record):
        """Emit a log record using tqdm-safe output.

        Args:
            record (logging.LogRecord): The log record to be emitted.
        """
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (BrokenPipeError, OSError, RuntimeError, ValueError, TypeError):
            self.handleError(record)


def setup_logger(
    logger_name: Optional[str] = None,
    log_file: Optional[str] = None,
    log_dir: str = "logs",
    log_to_console: bool = False,
    use_tqdm: bool = False,
    file_mode: Literal["a", "w"] = "a",
) -> logging.Logger:
    """Set up a logger with optional per-file handler, aggregate logs, and tqdm-safe console output.

    Args:
        logger_name (Optional[str]): The name or path for the per-file log (optional).
        log_file (Optional[str]): File name for the log file.
        log_dir (str, optional): Directory where log files will be created. Defaults to "logs".
        log_to_console (bool, optional): Whether to log messages to the console. Defaults to False.
        use_tqdm (bool, optional): If True, console output will use tqdm-safe handler.
        file_mode (str, optional): File open mode: "a" (append) or "w" (overwrite).
    """

    if file_mode not in {"a", "w"}:
        raise ValueError("file_mode must be 'a' (append) or 'w' (write).")

    os.makedirs(log_dir, exist_ok=True)

    if log_file:
        if os.path.isfile(log_file) or os.path.isdir(log_file):
            log_file = (
                os.path.splitext(os.path.basename(log_file))[0]
                if log_file
                else "all_logs"
            )

    if logger_name:
        logger = logging.getLogger(logger_name)
    elif log_file:
        logger = logging.getLogger(log_file)
    else:
        logger = logging.getLogger()

    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] - [%(filename)s:%(lineno)d] - [%(levelname)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Per-file handler (optional)
    log_files = [(os.path.join(log_dir, "all_logs"), "a")]
    if log_file:
        log_files.append((os.path.join(log_dir, log_file), file_mode))

    for _file, _mode in log_files:
        per_file_path = os.path.join(f"{_file}.log")
        file_handler = logging.FileHandler(per_file_path, mode=_mode)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Console logging
    if log_to_console:
        if use_tqdm:
            console_handler = TqdmLoggingHandler()
        else:
            console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def main():
    """Entry point for the program."""

    print(f"Welcome from `{os.path.basename(__file__).split('.')[0]}` Module.")

    # Example: tqdm logger (console-safe for progress bars)
    logger = setup_logger(
        log_file=__file__,
        log_dir="logs",
        log_to_console=True,
        use_tqdm=True,
        file_mode="w",
    )

    for i in tqdm.tqdm(range(50)):
        if i == 25:
            logger.info("Half-way there!\n Hi\n Hi\n Hi\n Hi\n Hi")
        time.sleep(0.05)


if __name__ == "__main__":
    main()