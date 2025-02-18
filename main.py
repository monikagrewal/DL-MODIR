import logging
import os
from datetime import datetime

from cli import cli_args
from config import config
from setup import setup_test, setup_train

# decide log level based on config
if config.DEBUG:
    logging_level = logging.DEBUG
else:
    logging_level = logging.INFO

# if someone tried to log something before basicConfig is called, Python creates a default handler that
# goes to the console and will ignore further basicConfig calls. Remove the handler if there is one.
os.makedirs(config.OUT_DIR, exist_ok=True)
t0 = datetime.now()
t0_str = datetime.strftime(t0, "%d%m%Y_%H%M%S")
root = logging.getLogger()
if root.handlers:
    for handler in root.handlers:
        root.removeHandler(handler)
logging.basicConfig(
    level=logging_level,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(f"{config.OUT_DIR}/info_{t0_str}.log"),
        logging.StreamHandler(),
    ],
)


if __name__ == "__main__":
    try:
        start_time = datetime.now()
        logging.info(f"Start time: {start_time}")
        if cli_args.env_file:
            # Train model and test on validation dataset
            logging.info("Training model")
            setup_train()
        elif cli_args.test_env_file:
            # Run model on test dataset
            logging.info("Testing model")
            setup_test()
        else:
            logging.warning("No config supplied.")
        end_time = datetime.now()
        logging.info(f"End time: {end_time}. Duration: {end_time - start_time}")
    except Exception as e:
        logging.warning(e)
        raise e
