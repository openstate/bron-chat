import logging
import os

from app.config import settings

logger = logging.getLogger(__name__)

EPOCH = "1970-01-01T00:00:00"


def read_last_run() -> str:
    """Last successful run timestamp from the state file, or epoch if missing."""
    try:
        if os.path.exists(settings.STATE_FILE):
            with open(settings.STATE_FILE) as f:
                value = f.read().strip()
                if value:
                    return value
    except Exception as e:
        logger.warning(f"Could not read state file {settings.STATE_FILE}: {e}")
    logger.info(f"No state file found at {settings.STATE_FILE}, starting from epoch.")
    return EPOCH


def write_last_run(timestamp: str) -> None:
    """Advance the state file after a successful upsert.

    Never regresses to an older timestamp: an interrupted backfill (--since)
    must not move the watermark back."""
    try:
        current = ""
        if os.path.exists(settings.STATE_FILE):
            with open(settings.STATE_FILE) as f:
                current = f.read().strip()
        if current and timestamp <= current:
            logger.debug(f"State already at {current}, not regressing to {timestamp}")
            return
        os.makedirs(os.path.dirname(settings.STATE_FILE), exist_ok=True)
        with open(settings.STATE_FILE, "w") as f:
            f.write(timestamp)
        logger.debug(f"State advanced to {timestamp}")
    except Exception as e:
        logger.error(f"Could not write state file {settings.STATE_FILE}: {e}")
