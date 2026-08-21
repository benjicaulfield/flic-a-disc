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
        import time
        url = args[0] if args else 'unknown'

        print(f"[API GET] {url}")
        limiter.acquire()
        start = time.time()

        try:
            result = original_get(*args, **kwargs)
            elapsed = time.time() - start
            print(f"[API GET] ✓ {url} ({elapsed:.2f}s)")
            return result
        except Exception as e:
            elapsed = time.time() - start
            print(f"[API GET] ✗ {url} ({elapsed:.2f}s)")
            print(f"  Error: {e}")
            if hasattr(e, 'response'):
                print(f"  Response status: {getattr(e.response, 'status_code', 'unknown')}")
                print(f"  Response headers: {getattr(e.response, 'headers', {})}")
                try:
                    print(f"  Response body: {e.response.text[:200]}")
                except:
                    pass
            raise

    @wraps(original_post)
    def rl_post(*args, **kwargs):
        import time
        url = args[0] if args else 'unknown'

        print(f"[API POST] {url}")
        limiter.acquire()
        start = time.time()

        try:
            result = original_post(*args, **kwargs)
            elapsed = time.time() - start
            print(f"[API POST] ✓ {url} ({elapsed:.2f}s)")
            return result
        except Exception as e:
            elapsed = time.time() - start
            print(f"[API POST] ✗ {url} ({elapsed:.2f}s)")
            print(f"  Error: {e}")
            raise

    client._get = rl_get
    client._post = rl_post
    return client

