from __future__ import annotations

import typing as tp
from datetime import datetime, timedelta
from pathlib import Path


class Stopwatch:

    def __init__(self, callback: tp.Callable[[timedelta], None] | None = None) -> None:
        self.__callback = callback
        self.__start = datetime.now()
        self.__elapsed = timedelta()

    def __str__(self) -> str:
        return f'{self.__elapsed.total_seconds():.3f}'

    def __repr__(self):
        return self.__str__()

    def __enter__(self) -> Stopwatch:
        return self

    def __exit__(self, *_) -> None:
        try:
            self.close()
        except AttributeError:
            pass

    def close(self) -> None:
        self.__elapsed = datetime.now() - self.__start
        if self.__callback is not None:
            self.__callback(self.__elapsed)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            f.write(f'{self}\n')
