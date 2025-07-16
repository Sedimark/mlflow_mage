"""Helper class for working with MLFlow tags."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Optional
from typing import Iterator


class Tags(Mapping[str, str]):
    def __init__(self, tags: Optional[dict[str, str]] = None) -> None:
        if tags is not None:
            Tags._ensure_type(str, **tags)
        self._tags = tags or {}

    @staticmethod
    def _ensure_type(type: object, *args, **kwargs):
        for key, value in kwargs.items():
            if not isinstance(key, type):
                raise TypeError(f"Key {key} is not of type: {type}.")

            if not isinstance(value, type):
                raise TypeError(f"Value {value} is not of type: {type}.")

        for value in args:
            if not isinstance(value, type):
                raise TypeError(f"Value {value} is not of type: {type}.")

    @property
    def tags(self) -> dict[str, str]:
        return self._tags
    
    def update(self, tags: dict[str, str]):
        if tags is None:
            raise ValueError("Input must not be none.") 

        for key, value in tags.items():
            self.__setitem__(key, value)
    
    def __getitem__(self, key: str) -> Optional[str]:
        return self._tags.get(key)

    def __setitem__(self, name: str, value: str) -> Optional[str]:
        prev_val = self._tags.get(name)

        Tags._ensure_type(str, *[name, value])
        
        self._tags[name] = value

        return prev_val

    def __len__(self):
        return len(self._tags)

    def __iter__(self) -> Iterator[str]:
        return iter(self._tags)
