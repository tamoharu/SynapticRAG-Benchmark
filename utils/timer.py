import time
import functools


def average_timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        wrapper.times.append(elapsed)
        wrapper.count += 1
        return result
    wrapper.times = []
    wrapper.count = 0
    return wrapper
