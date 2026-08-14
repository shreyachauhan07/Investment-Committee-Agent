"""Shared pytest fixtures."""

from __future__ import annotations

import json

import pytest

from models import Fund, Portfolio, UserProfile

BASE = __file__.rsplit("\\", 1)[0]  # tests/ dir


def load_sample() -> Portfolio:
    with open(f"{BASE}\\..\\data\\sample_portfolio.json", encoding="utf-8") as fh:
        data = json.load(fh)
    return Portfolio(
        profile=UserProfile(**data["profile"]),
        funds=[Fund(**fund) for fund in data["funds"]],
    )


@pytest.fixture
def sample_portfolio() -> Portfolio:
    return load_sample()
