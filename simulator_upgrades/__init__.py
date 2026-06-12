"""Dockable simulator upgrades for the QP-SCM experiments.

The original event engine lives in ``simulation/``, ``limit_order_book/``,
``config/``, and ``rl_files/``.  This package is intentionally separate:
experiments opt in by using :class:`DockedMarket` or :func:`make_docked_env`.
No module in the original simulator imports this package.
"""

from simulator_upgrades.dock import DockedMarket, make_docked_env
from simulator_upgrades.hooks import HookContext, SimulatorHook, UpgradeBundle, NoOpHook

__all__ = [
    "DockedMarket",
    "HookContext",
    "NoOpHook",
    "SimulatorHook",
    "UpgradeBundle",
    "make_docked_env",
]
