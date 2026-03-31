"""LarkIGTLClient: IGTLClient configured for FALCON/LARK tracking system."""
from __future__ import annotations
from typing import Optional
from surgical_nav.tracking.igtl_client import IGTLClient
from surgical_nav.tracking.transform_store import TransformStore


class LarkIGTLClient(IGTLClient):
    """IGTLClient pre-configured for FALCON/LARK optical tracking.

    Connects to LARK's pyigtl server (default port 18995) and remaps
    "PointerDevice" -> "PointerToTracker" so the rest of the pipeline
    works unchanged.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 18995,
        store: Optional[TransformStore] = None,
        poll_hz: float = 10.0,
        parent=None,
    ):
        super().__init__(
            host=host,
            port=port,
            store=store,
            poll_hz=poll_hz,
            device_name_map={"PointerDevice": "PointerToTracker"},
            parent=parent,
        )
