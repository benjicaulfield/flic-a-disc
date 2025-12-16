from time import monotonic, sleep
from collections import deque
from functools import wraps

class RateLimiter:
    def __init__(self, max_per_min=60):
        self.max = max_per_min
        self.window = 60.0
        self.times = deque()

    def acquire(self):
        now = monotonic()
        # purge old
        while self.times and now - self.times[0] >= self.window:
            self.times.popleft()
        if len(self.times) >= self.max:
            wait = self.window - (now - self.times[0])
            sleep(wait)
            # after sleeping, purge again
            now = monotonic()
            while self.times and now - self.times[0] >= self.window:
                self.times.popleft()
        self.times.append(now)

def rate_limit_client(client, limiter=None):
    limiter = limiter or RateLimiter(60)
    original_get = client._get
    original_post = client._post

    @wraps(original_get)
    def rl_get(*args, **kwargs):
        limiter.acquire()
        return original_get(*args, **kwargs)

    @wraps(original_post)
    def rl_post(*args, **kwargs):
        limiter.acquire()
        return original_post(*args, **kwargs)

    client._get = rl_get
    client._post = rl_post
    return client

