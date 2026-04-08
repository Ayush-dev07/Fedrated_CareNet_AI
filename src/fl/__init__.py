"""Federated learning core — client, server, strategies, privacy, personalization."""

from src.fl.client.client import HealthClient
from src.fl.server.server import start_server

__all__ = [
    "HealthClient",
    "start_server",
]