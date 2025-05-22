"""Shared no-op agent for YAML prototyping."""
from __future__ import annotations

from typing import ClassVar
from pydantic import BaseModel

from autogen_core import Component, ComponentBase

# ── 1. pydantic config model ──────────────────────────────────────────────
class StubConfig(BaseModel):
    name: str | None = None
    system_message: str | None = None

# ── 2. reusable agent class ───────────────────────────────────────────────
class StubAgent(ComponentBase[StubConfig], Component[StubConfig]):
    """A do-nothing agent that only exists so YAML imports succeed."""
    component_type:   ClassVar[str]             = "agent"
    component_config_schema: ClassVar[type]     = StubConfig

    # ----- plumbing for ComponentBase -----
    def __init__(self, cfg: StubConfig) -> None:
        self.cfg = cfg

    def _to_config(self) -> StubConfig:         # for dump_component()
        return self.cfg

    @classmethod
    def _from_config(cls, cfg: StubConfig) -> "StubAgent":  # for load_component()
        return cls(cfg)
