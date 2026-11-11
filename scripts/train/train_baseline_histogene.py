"""Compatibility entrypoint; implementation lives in benchmarks/."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("sigmma_benchmarks.runners.train_baseline_histogene", run_name="__main__")
else:
    from importlib import import_module
    _implementation = import_module("sigmma_benchmarks.runners.train_baseline_histogene")

    def __getattr__(name):
        return getattr(_implementation, name)
