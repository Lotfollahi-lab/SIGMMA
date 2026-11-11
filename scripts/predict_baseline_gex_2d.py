"""Compatibility entrypoint; implementation lives in benchmarks/."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("sigmma_benchmarks.runners.predict_baseline_gex_2d", run_name="__main__")
else:
    from importlib import import_module
    _implementation = import_module("sigmma_benchmarks.runners.predict_baseline_gex_2d")

    def __getattr__(name):
        return getattr(_implementation, name)
