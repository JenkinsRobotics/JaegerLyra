"""Request-scoped cancellation, including cancel-before-request HTTP races."""
import re
import threading
import time

GUARD = threading.Lock()
REQUESTS = {}

class VoiceCancelled(Exception):
    pass

def check(event):
    if event.is_set():
        raise VoiceCancelled('Speech cancelled')

def token(value):
    if not isinstance(value, str) or not re.fullmatch(r'[a-zA-Z0-9_-]{16,80}', value):
        raise ValueError('Invalid speech request ID')
    return value

def register(value):
    if value is None:
        return threading.Event()
    key = token(value)
    with GUARD:
        now = time.monotonic()
        for old, (event, deadline, active) in list(REQUESTS.items()):
            if not active and deadline < now:
                del REQUESTS[old]
        if key in REQUESTS and REQUESTS[key][2]:
            raise ValueError('Speech request is already active')
        event = REQUESTS.get(key, (threading.Event(), 0, False))[0]
        REQUESTS[key] = (event, now + 120, True)
        return event

def finish(value):
    if value is None:
        return
    with GUARD:
        if value in REQUESTS:
            event, _, _ = REQUESTS[value]
            REQUESTS[value] = (event, time.monotonic() + 120, False)

def cancel(value):
    key = token(value)
    with GUARD:
        now = time.monotonic()
        for old, (event, deadline, active) in list(REQUESTS.items()):
            if not active and deadline < now:
                del REQUESTS[old]
        if len(REQUESTS) >= 512 and key not in REQUESTS:
            raise ValueError('Too many speech requests')
        event, _, active = REQUESTS.get(key, (threading.Event(), 0, False))
        event.set()
        REQUESTS[key] = (event, now + 120, active)
    return {'ok': True}
