"""Test fixtures shared across the autumn_py test suite."""
from __future__ import annotations

import pytest

from autumn_py.api import _pending_on_clauses


@pytest.fixture(autouse=True)
def _clear_on_clause_pending_list():
    """The @on decorator pushes onto a module-level pending list that @program
    drains. Standalone @on usage in tests (without a @program class) leaves
    stale entries; clear before and after every test for isolation."""
    _pending_on_clauses.clear()
    yield
    _pending_on_clauses.clear()
