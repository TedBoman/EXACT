import sys
class DebugLogger:
    _debug = False  # Class variable shared across all instances

    @classmethod
    def set_debug(cls, value: bool):
        cls._debug = value

    @classmethod
    def debug_print(cls, *args, **kwargs):
        if cls._debug:
            print(*args, **kwargs)
            sys.stdout.flush()
    
    @classmethod
    def print_exception(cls, *args, **kwargs):
        print(*args, **kwargs)
        sys.stdout.flush()