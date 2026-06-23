from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Decision:
    endpoint: str   # "assign-ramp" or "reject-truck"
    supplier_id:   int
    supplier_name: str
    parcel_count:  int
    has_damage:    bool
    unit:          str              # "parcels" | "pallets"
    assigned_ramp: str | None = None  # required when endpoint == "assign-ramp"


class Strategy(ABC):
    """
    Implement decide() to build a new ramp-assignment strategy.
    The truck dict is the raw WebSocket message from the server.
    """

    @abstractmethod
    async def decide(self, truck: dict) -> Decision:
        ...
