"""VC-6 — the offline guard (PLAN.md §11, task V2).

``BNQA_OFFLINE=1`` installs a socket guard that raises on any outbound
connection.  The demo then runs identically, which is the claim: **unplug the
Wi-Fi in front of the examiner** and nothing changes, because after T3 nothing
in Tier A needs the network.

This is stronger than "we did not notice any requests".  A guard that raises
turns a silent dependency into a stack trace with the host name in it, which
is the only way to *know* rather than believe.  ``loopback_ok`` stays open
because Streamlit binds a local port to serve the UI — that is the app talking
to the examiner's own browser, not to us.

    $env:BNQA_OFFLINE = "1"
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os
import socket

ENV_VAR = "BNQA_OFFLINE"

_original_socket = socket.socket
_original_create_connection = socket.create_connection
_installed = False


class OutboundBlocked(RuntimeError):
    """Raised when offline mode is on and something tries to leave the machine."""


def _is_loopback(address) -> bool:
    try:
        host = address[0] if isinstance(address, (tuple, list)) else address
    except Exception:  # pragma: no cover
        return False
    return str(host) in ("127.0.0.1", "::1", "localhost", "0.0.0.0", "")


class _GuardedSocket(_original_socket):  # type: ignore[misc,valid-type]
    def connect(self, address):  # noqa: D102
        if not _is_loopback(address):
            raise OutboundBlocked(
                f"{ENV_VAR}=1 — outbound connection to {address!r} refused. "
                "Tier A needs no network after the corpus is built (VC-6).")
        return super().connect(address)

    def connect_ex(self, address):  # noqa: D102
        if not _is_loopback(address):
            raise OutboundBlocked(
                f"{ENV_VAR}=1 — outbound connection to {address!r} refused.")
        return super().connect_ex(address)


def _guarded_create_connection(address, *args, **kwargs):
    if not _is_loopback(address):
        raise OutboundBlocked(
            f"{ENV_VAR}=1 — outbound connection to {address!r} refused.")
    return _original_create_connection(address, *args, **kwargs)


def install(force: bool = False) -> bool:
    """Install the guard when ``BNQA_OFFLINE`` is set.  Returns whether it is on."""
    global _installed
    if not force and os.environ.get(ENV_VAR, "") not in ("1", "true", "True", "yes"):
        return False
    if not _installed:
        socket.socket = _GuardedSocket  # type: ignore[misc,assignment]
        socket.create_connection = _guarded_create_connection  # type: ignore[assignment]
        _installed = True
    return True


def uninstall() -> None:
    """Restore the real socket — used by the test, not by the app."""
    global _installed
    socket.socket = _original_socket  # type: ignore[misc,assignment]
    socket.create_connection = _original_create_connection  # type: ignore[assignment]
    _installed = False


def is_active() -> bool:
    return _installed


def status() -> dict:
    return {"env": os.environ.get(ENV_VAR, ""), "guard_installed": _installed}
