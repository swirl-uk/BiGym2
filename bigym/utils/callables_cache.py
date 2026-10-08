"""Cache for callables."""

from __future__ import annotations

from typing import Any, Callable, Protocol


class _NamedCallable(Protocol):
    @property
    def __name__(self) -> str: ...

    def __call__(self) -> Any: ...


class CallablesCache:
    """Cache for callables."""

    def __init__(self):
        """Init."""
        self._callables: dict[str, Callable] = {}
        self._values: dict[str, Any] = {}

    def get(self, func: _NamedCallable) -> Any:
        """Get value of callable."""
        key = func.__name__
        if key not in self._callables:
            self._callables[key] = func
        value = self._values.get(key, None)
        if value is None:
            value = self._callables[key]()
            self._values[key] = value
        return value

    def clean(self):
        """Clean cache values."""
        self._values.clear()
