try:
    import importlib.metadata
    __version__ = importlib.metadata.version("understanding-llms")
except Exception:
    __version__ = "0.1.0"
