"""Length-prefixed TCP framing with bounded message sizes."""
from __future__ import annotations

import socket
import struct


MAX_FRAME_BYTES = 64 * 1024 * 1024


def recv_exact(sock: socket.socket, size: int) -> bytes:
    """Read exactly size bytes or fail on disconnect/truncation."""
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("Peer disconnected during framed message")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_frame(sock: socket.socket, payload: bytes) -> None:
    """Send one bounded length-prefixed payload."""
    if not payload or len(payload) > MAX_FRAME_BYTES:
        raise ValueError("Frame must be non-empty and within the configured size limit")
    sock.sendall(struct.pack("!I", len(payload)) + payload)


def recv_frame(sock: socket.socket) -> bytes:
    """Receive one bounded length-prefixed payload."""
    size = struct.unpack("!I", recv_exact(sock, 4))[0]
    if size < 1 or size > MAX_FRAME_BYTES:
        raise ValueError(f"Invalid frame size: {size}")
    return recv_exact(sock, size)
