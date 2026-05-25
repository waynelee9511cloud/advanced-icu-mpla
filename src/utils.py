import time
import logging
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger('ICUPredictionSystem')

def timer(func):
    """
    A custom decorator that logs the execution time of a function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        logger.info(f"Starting execution of '{func.__name__}'...")
        try:
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            elapsed = end_time - start_time
            logger.info(f"Finished execution of '{func.__name__}' in {elapsed:.4f} seconds.")
            return result
        except Exception as e:
            end_time = time.perf_counter()
            elapsed = end_time - start_time
            logger.error(f"Function '{func.__name__}' failed after {elapsed:.4f} seconds with error: {e}")
            raise
    return wrapper
