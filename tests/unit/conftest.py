"""Unit test conftest — registers stub modules for optional SDK dependencies."""

from __future__ import annotations

import sys
import types
import unittest.mock


def _register_sdk_stub(sdk_name: str, **attrs: object) -> None:
    """Put a stub module in sys.modules if the real SDK is not installed."""
    if sdk_name in sys.modules:
        return
    try:
        __import__(sdk_name)
    except ImportError:
        stub = types.ModuleType(sdk_name)
        for name, value in attrs.items():
            setattr(stub, name, value)
        sys.modules[sdk_name] = stub


# ---------------------------------------------------------------------------
# SDK stubs
# ---------------------------------------------------------------------------

_register_sdk_stub("anthropic", Anthropic=unittest.mock.MagicMock)
_register_sdk_stub("groq", Groq=unittest.mock.MagicMock)
