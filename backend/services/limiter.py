import time
from fastapi import HTTPException, status

class TokenBucket:
    def __init__(self, rate_per_min=60, capacity=None):
        self.rate = rate_per_min / 60.0
        self.capacity = capacity or rate_per_min
        self.tokens = self.capacity
        self.timestamp = time.time()

    def consume(self, amount=1):
        now = time.time()
        delta = now - self.timestamp
        self.timestamp = now
        self.tokens = min(self.capacity, self.tokens + delta * self.rate)
        if self.tokens < amount:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded")
        self.tokens -= amount

buckets = {}

def check_rate_limit(ip: str, rate_per_min: int):
    bucket = buckets.get(ip)
    if not bucket:
        bucket = TokenBucket(rate_per_min=rate_per_min)
        buckets[ip] = bucket
    bucket.consume()
