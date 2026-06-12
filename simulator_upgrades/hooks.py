"""Hook interfaces for dockable simulator upgrades.

Hooks are deliberately conservative.  They receive the live
``simulation.market_gym.Market`` instance but live outside the original
engine.  A hook may inspect the market, modify generated order lists,
or attach diagnostics to ``market.upgrade_state``.  Returning ``None``
from an order hook means "leave the order list unchanged".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol


@dataclass(frozen=True)
class HookContext:
    """Minimal event context passed to upgrade hooks."""

    time: float | int
    priority: int
    agent_id: str


class SimulatorHook(Protocol):
    """Protocol for optional realism/stress-test modules."""

    name: str

    def before_reset(self, market: Any) -> None:
        ...

    def after_reset(self, market: Any, observation: Any, info: dict[str, Any]) -> None:
        ...

    def before_event(self, market: Any, context: HookContext) -> None:
        ...

    def after_generate_orders(self, market: Any, context: HookContext, orders: Any) -> Any | None:
        ...

    def before_process_orders(self, market: Any, context: HookContext, orders: Any) -> Any | None:
        ...

    def after_process_messages(self, market: Any, context: HookContext, orders: Any, messages: Any) -> None:
        ...

    def after_transition(self, market: Any, observation: Any, reward: float, terminated: bool, info: dict[str, Any]) -> None:
        ...


class NoOpHook:
    """A hook that documents the baseline behavior and changes nothing."""

    name = "noop"

    def before_reset(self, market: Any) -> None:
        return None

    def after_reset(self, market: Any, observation: Any, info: dict[str, Any]) -> None:
        return None

    def before_event(self, market: Any, context: HookContext) -> None:
        return None

    def after_generate_orders(self, market: Any, context: HookContext, orders: Any) -> Any | None:
        return None

    def before_process_orders(self, market: Any, context: HookContext, orders: Any) -> Any | None:
        return None

    def after_process_messages(self, market: Any, context: HookContext, orders: Any, messages: Any) -> None:
        return None

    def after_transition(self, market: Any, observation: Any, reward: float, terminated: bool, info: dict[str, Any]) -> None:
        return None


@dataclass
class UpgradeBundle:
    """Ordered collection of simulator hooks.

    Hooks are applied in list order.  This gives future realism modules a
    deterministic composition rule, e.g. hidden liquidity before quote
    revision before refill diagnostics.
    """

    hooks: list[SimulatorHook] = field(default_factory=list)
    name: str = "custom"

    @classmethod
    def noop(cls) -> "UpgradeBundle":
        return cls(hooks=[NoOpHook()], name="noop")

    @classmethod
    def from_hooks(cls, hooks: Iterable[SimulatorHook], *, name: str = "custom") -> "UpgradeBundle":
        return cls(hooks=list(hooks), name=name)

    def hook_names(self) -> list[str]:
        return [getattr(hook, "name", hook.__class__.__name__) for hook in self.hooks]
