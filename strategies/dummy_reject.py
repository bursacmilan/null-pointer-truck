import logging

from .base import Decision, Strategy

log = logging.getLogger(__name__)


class DummyRejectStrategy(Strategy):
    """
    Stub strategy — always rejects every truck with placeholder extraction data.
    Used to verify end-to-end WebSocket and POST plumbing before real logic.
    """

    async def decide(self, truck: dict) -> Decision:
        log.debug("DummyRejectStrategy: rejecting truck unconditionally")
        return Decision(
            endpoint      = "reject-truck",
            supplier_id   = 1000000,    # placeholder
            supplier_name = "Unknown",
            parcel_count  = 1,          # placeholder
            has_damage    = True,
            unit          = "pallets",
        )
