import logging
import time
from tqdm import trange
from tqdm.contrib.logging import logging_redirect_tqdm

LOG = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    with logging_redirect_tqdm():
        for i in trange(9):
            time.sleep(0.1)

            if i == 4:
                LOG.info("console logging redirected to `tqdm.write()`")
    # logging restored
