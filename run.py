"""CLI runner for launching the Streamlit control dashboard."""

import sys


def _apply_torch_watcher_patch() -> None:
    """
    Patch torch.classes.__getattr__ to prevent Streamlit's local_sources_watcher
    from raising RuntimeError: Tried to instantiate class '__path__._path'.
    """
    try:
        import torch
        orig_getattr = torch._classes._Classes.__getattr__

        def safe_getattr(self, attr):
            if attr in ("__path__", "__file__"):
                raise AttributeError(attr)
            return orig_getattr(self, attr)

        torch._classes._Classes.__getattr__ = safe_getattr
    except Exception:
        pass


if __name__ == "__main__":
    _apply_torch_watcher_patch()
    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", "src/ui/app.py"]
    sys.exit(stcli.main())
