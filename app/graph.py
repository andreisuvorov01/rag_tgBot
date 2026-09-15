"""Лёгкий графовый движок агентов (в духе LangGraph: узлы, условные рёбра,
трассировка, защита от циклов). QA-конвейер выражен графом: классификация →
резолв показателя → ветвление по намерению → исполнитель → композитор.

Компромисс осознанный: не тащим LangGraph как зависимость — семантика
(состояние, условные переходы, шаг-лимит) покрывает наши сценарии в ~80 строк.
"""
from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

log = logging.getLogger(__name__)

END = "__end__"

NodeFn = Callable[[dict], Awaitable[dict]]
RouterFn = Callable[[dict], str]


class GraphError(RuntimeError):
    pass


class AgentGraph:
    def __init__(self, name: str = "graph", max_steps: int = 16):
        self.name = name
        self.max_steps = max_steps
        self._nodes: dict[str, NodeFn] = {}
        self._edges: dict[str, str] = {}
        self._branches: dict[str, tuple[RouterFn, dict[str, str]]] = {}
        self._entry: str | None = None

    def node(self, name: str, fn: NodeFn) -> AgentGraph:
        self._nodes[name] = fn
        return self

    def edge(self, a: str, b: str) -> AgentGraph:
        self._edges[a] = b
        return self

    def branch(self, a: str, router: RouterFn, mapping: dict[str, str]) -> AgentGraph:
        """Условные рёбра: router(state) -> ключ; mapping[key] -> следующий узел."""
        self._branches[a] = (router, mapping)
        return self

    def set_entry(self, name: str) -> AgentGraph:
        self._entry = name
        return self

    async def run(self, state: dict | None = None) -> dict:
        if self._entry is None:
            raise GraphError("не задан входной узел")
        state = dict(state or {})
        current: str | None = self._entry
        steps = 0
        trace: list[str] = []
        while current and current != END:
            steps += 1
            if steps > self.max_steps:
                raise GraphError(f"граф '{self.name}': превышен лимит шагов ({self.max_steps})")
            fn = self._nodes.get(current)
            if fn is None:
                raise GraphError(f"граф '{self.name}': неизвестный узел '{current}'")
            state = await fn(state) or state
            trace.append(current)
            if current in self._branches:
                router, mapping = self._branches[current]
                key = router(state)
                current = mapping.get(key, mapping.get("__default__", END))
            elif current in self._edges:
                current = self._edges[current]
            else:
                current = END
        state["trace"] = trace
        return state
