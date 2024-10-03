from __future__ import annotations

from datetime import datetime


class Stopwatch:

    def __init__(self, callback) -> None:
        self.callback = callback
        self.start = datetime.now()

    def __enter__(self) -> Stopwatch:
        return self

    def __exit__(self, *_) -> None:
        try:
            self.close()
        except AttributeError:
            pass

    def close(self) -> None:
        self.callback(datetime.now() - self.start)
