from datetime import datetime


class CallbackTimer:

    def __init__(self, callback):
        self.callback = callback
        self.start = datetime.now()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        try:
            self.close()
        except AttributeError:
            pass

    def close(self):
        self.callback(datetime.now() - self.start)
