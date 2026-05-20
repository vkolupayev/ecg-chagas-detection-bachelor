import os
import logging

from src.constants import LOG_LEVEL


def setup_logger(log_file_name: str | None = None, log_path: str = "logs"):
    level = getattr(logging, LOG_LEVEL, logging.INFO)

    if not log_file_name:
        log_file_name = "app.log"
    log_file_name = (
        log_file_name if (log_file_name.endswith(".log")) else f"{log_file_name}.log"
    )

    if not os.path.exists(log_path):
        os.mkdir(log_path)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(f"{log_path}/{log_file_name}"),
            logging.StreamHandler(),
        ],
    )
