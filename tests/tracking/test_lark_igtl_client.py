"""Tests for LarkIGTLClient."""
from __future__ import annotations
import pytest
from unittest.mock import MagicMock, patch
import numpy as np


def test_lark_igtl_client_defaults():
    """LarkIGTLClient should default to port 18995."""
    from surgical_nav.tracking.lark_igtl_client import LarkIGTLClient
    client = LarkIGTLClient()
    assert client._port == 18995
    assert client._host == "localhost"


def test_lark_igtl_client_name_map():
    """LarkIGTLClient should remap PointerDevice -> PointerToTracker."""
    from surgical_nav.tracking.lark_igtl_client import LarkIGTLClient
    client = LarkIGTLClient()
    assert client._name_map == {"PointerDevice": "PointerToTracker"}


def test_lark_igtl_client_custom_host_port():
    """Custom host/port should be forwarded."""
    from surgical_nav.tracking.lark_igtl_client import LarkIGTLClient
    client = LarkIGTLClient(host="192.168.1.10", port=19000)
    assert client._host == "192.168.1.10"
    assert client._port == 19000


def test_igtl_worker_name_remapping():
    """_IGTLWorker should remap device names before emitting."""
    from surgical_nav.tracking.igtl_client import _IGTLWorker
    from surgical_nav.tracking.transform_store import TransformStore
    store = TransformStore()
    worker = _IGTLWorker(
        host="localhost", port=18995, store=store, poll_hz=10.0,
        name_map={"PointerDevice": "PointerToTracker"},
    )
    assert worker._name_map == {"PointerDevice": "PointerToTracker"}


def test_igtl_worker_no_name_map():
    """_IGTLWorker with no name_map should leave names unchanged."""
    from surgical_nav.tracking.igtl_client import _IGTLWorker
    from surgical_nav.tracking.transform_store import TransformStore
    store = TransformStore()
    worker = _IGTLWorker(host="localhost", port=18944, store=store, poll_hz=10.0)
    assert worker._name_map == {}
