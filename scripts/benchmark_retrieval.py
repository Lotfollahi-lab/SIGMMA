"""Compatibility entrypoint; implementation lives in benchmarks/."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("sigmma_benchmarks.runners.benchmark_retrieval", run_name="__main__")
else:
    from importlib import import_module
    _implementation = import_module("sigmma_benchmarks.runners.benchmark_retrieval")

    def __getattr__(name):
        return getattr(_implementation, name)
