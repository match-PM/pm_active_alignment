"""Bounded waits for use in executor worker threads, without nested spinning."""

import threading
import time

from ..optimization.base import OptimizationCancelled


def wait_future(future, timeout, cancelled=lambda: False):
    event = threading.Event()
    future.add_done_callback(lambda _: event.set())
    deadline = time.monotonic() + timeout
    while not future.done():
        if cancelled():
            raise OptimizationCancelled('Cancelled while waiting for a ROS response.')
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Timed out waiting for a ROS response.')
        event.wait(min(0.05, remaining))
    return future.result()


def wait_service(client, timeout, cancelled=lambda: False):
    deadline = time.monotonic() + timeout
    while not client.wait_for_service(timeout_sec=0.05):
        if cancelled():
            raise OptimizationCancelled('Cancelled while waiting for a service.')
        if time.monotonic() >= deadline:
            raise TimeoutError(f'Service unavailable: {client.srv_name}')


def interruptible_sleep(duration, cancelled=lambda: False):
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        if cancelled():
            raise OptimizationCancelled('Cancelled while settling.')
        time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))
