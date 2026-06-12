"""Docking layer for optional simulator upgrades.

``DockedMarket`` subclasses the original ``simulation.market_gym.Market``
without editing that file.  The override copies only the event loop and
reset scaffolding so hooks can be inserted at stable points.  With a
``NoOpHook`` bundle, the intended behavior is identical to the original
market environment.
"""

from __future__ import annotations

from queue import PriorityQueue
from typing import Any

import numpy as np

from limit_order_book.limit_order_book import LimitOrderBook
from simulation.market_gym import Market
from simulator_upgrades.hooks import HookContext, UpgradeBundle


class DockedMarket(Market):
    """Original market engine with opt-in external hooks.

    Parameters
    ----------
    config:
        The same config dictionary accepted by ``simulation.market_gym.Market``.
    upgrade_bundle:
        Ordered hook bundle.  If omitted, a no-op bundle is installed.

    Notes
    -----
    This class intentionally does not alter the author's original engine.
    Existing scripts that import ``Market`` continue to use the baseline.
    New realism experiments can opt in by importing ``DockedMarket``.
    """

    def __init__(self, config: dict[str, Any], upgrade_bundle: UpgradeBundle | None = None):
        self.upgrade_bundle = upgrade_bundle or UpgradeBundle.noop()
        self.upgrade_state: dict[str, Any] = {"bundle": self.upgrade_bundle.name, "hooks": self.upgrade_bundle.hook_names()}
        super().__init__(config)

    def _call(self, method_name: str, *args: Any) -> None:
        for hook in self.upgrade_bundle.hooks:
            method = getattr(hook, method_name, None)
            if method is not None:
                method(self, *args)

    def _apply_order_hook(self, method_name: str, context: HookContext, orders: Any) -> Any:
        out = orders
        for hook in self.upgrade_bundle.hooks:
            method = getattr(hook, method_name, None)
            if method is None:
                continue
            candidate = method(self, context, out)
            if candidate is not None:
                out = candidate
        return out

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        self._call("before_reset")
        self.lob = LimitOrderBook(list_of_agents=list(self.agents.keys()), level=30, only_volumes=False)
        for agent_id in self.agents:
            self.agents[agent_id].reset()
        self.pq = PriorityQueue()
        for agent_id in self.agents:
            out = self.agents[agent_id].initial_event()
            self.pq.put(out)
        observation, reward, terminated, info = self.transition()
        self._call("after_reset", observation, info)
        return observation, info

    def transition(self, action: Any = None):
        terminated = False
        transition_reward = 0
        if action is not None and self.transform_action:
            action = np.exp(action) / np.sum(np.exp(action))

        while not self.pq.empty():
            t, priority, agent_id = self.pq.get()
            if t > self.agents[self.execution_agent_id].terminal_time:
                raise ValueError("time is greater than execution agents terminal time")

            context = HookContext(time=t, priority=priority, agent_id=agent_id)
            self._call("before_event", context)

            if agent_id == "rl_agent":
                orders = self.agents[agent_id].generate_order(lob=self.lob, time=t, action=action)
            else:
                orders = self.agents[agent_id].generate_order(lob=self.lob, time=t)
            orders = self._apply_order_hook("after_generate_orders", context, orders)

            if orders is not None or orders == []:
                orders = self._apply_order_hook("before_process_orders", context, orders)
                msgs = self.lob.process_order_list(orders)
                self._call("after_process_messages", context, orders, msgs)
                reward, terminated = self.agents[self.execution_agent_id].update_position_from_message_list(msgs)
                transition_reward += reward
                if terminated:
                    break

            out = self.agents[agent_id].new_event(t, agent_id)
            if out is not None:
                self.pq.put(out)
            if agent_id == "observation_agent":
                break

        if terminated:
            assert self.agents[self.execution_agent_id].volume == 0

        mid_price = (self.lob.data.best_bid_prices[-1] + self.lob.data.best_ask_prices[-1]) / 2
        initial_mid_price = (self.agents["initial_agent"].initial_ask + self.agents["initial_agent"].initial_bid) / 2
        info = {
            "reward": self.agents[self.execution_agent_id].cummulative_reward,
            "passive_fill_rate": self.agents[self.execution_agent_id].limit_sells
            / self.agents[self.execution_agent_id].initial_volume,
            "time": t,
            "drift": mid_price - initial_mid_price,
            "n_events": self.agents["noise_agent"].n_events,
            "terminated": terminated,
            "volume": self.agents[self.execution_agent_id].volume,
            "upgrade_bundle": self.upgrade_bundle.name,
            "upgrade_hooks": self.upgrade_bundle.hook_names(),
        }
        if self.execution_agent_id == "rl_agent":
            observation = self.agents[self.execution_agent_id].get_observation(t, self.lob)
        else:
            observation = np.array([None], dtype=np.float32)

        self._call("after_transition", observation, transition_reward, terminated, info)
        return observation, transition_reward, terminated, info


def make_docked_env(config: dict[str, Any], upgrade_bundle: UpgradeBundle | None = None):
    """Gym-style thunk matching ``simulation.market_gym.make_env``."""

    def thunk() -> DockedMarket:
        return DockedMarket(config, upgrade_bundle=upgrade_bundle)

    return thunk
