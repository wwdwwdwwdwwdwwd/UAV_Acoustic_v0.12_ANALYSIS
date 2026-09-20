from __future__ import annotations
import socket
import time
from dataclasses import dataclass
import numpy as np
from ..types import AcousticFrame
from .channel_mapping import physical_order_metadata, to_vendor_physical_order


class WaveFragUDPError(RuntimeError):
    """Base class for actionable WaveFrag capture failures."""


class PortBindError(WaveFragUDPError):
    pass


class CaptureTimeoutError(WaveFragUDPError):
    pass


class WrongSampleCountError(WaveFragUDPError):
    pass


class ChannelMismatchError(WaveFragUDPError):
    pass


@dataclass
class WaveFragUDPConfig:
    host: str = "0.0.0.0"
    port: int = 3860
    channels: int = 128
    fs: int = 80000
    reorder_vendor_channels: bool = True
    recv_bytes: int = 1 << 20
    timeout_s: float = 1.0
    socket_buffer_bytes: int = 8 << 20
    expected_source_ip: str | None = None
    expected_source_port: int | None = None
    expected_datagram_bytes: int | None = None


class WaveFragUDPSource:
    """Minimal UDP receiver based on the vendor MATLAB demo.

    Assumptions confirmed by supplied MATLAB code:
      - little-endian int16 samples
      - 128 interleaved channels
      - 80 kHz nominal sample rate
      - configurable vendor-specific channel reorder; disabled for the observed
        192.168.0.100:5555 stream after Round-1 consistency analysis

    Not yet guaranteed by supplied material:
      - packet header/sequence/timestamp semantics
      - cross-device synchronization

    For reliable real-time use, replace the raw datagram accumulation below
    with the vendor's final packet protocol once it is supplied/verified.
    """
    def __init__(self, mic_xyz: np.ndarray, config: WaveFragUDPConfig | None = None):
        self.cfg = config or WaveFragUDPConfig()
        self.mic_xyz = np.asarray(mic_xyz, dtype=float)
        if self.mic_xyz.shape != (self.cfg.channels, 3):
            raise ValueError(f"mic_xyz must be ({self.cfg.channels}, 3)")
        if not self.cfg.reorder_vendor_channels:
            raise ValueError(
                "Raw-order AcousticFrame output is forbidden: WaveFrag acquisition must "
                "normalize to VendorPosMicRowOrder"
            )
        self.sock: socket.socket | None = None
        self._byte_buffer = bytearray()
        self._datagram_count = 0
        self._datagram_sizes: list[int] = []
        self._source_addresses: set[tuple[str, int]] = set()
        self.last_raw_int16: np.ndarray | None = None

    def open(self) -> None:
        if self.sock is not None:
            raise RuntimeError("UDP source is already open")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.cfg.socket_buffer_bytes)
            sock.bind((self.cfg.host, self.cfg.port))
            sock.settimeout(self.cfg.timeout_s)
        except OSError as exc:
            sock.close()
            raise PortBindError(
                f"Cannot bind UDP {self.cfg.host}:{self.cfg.port}; port may be occupied "
                f"or the local IP is not configured: {exc}"
            ) from exc
        self.sock = sock

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    def read_frame(self, n_samples: int) -> AcousticFrame:
        if self.sock is None:
            raise RuntimeError("Call open() first")
        bytes_needed = n_samples * self.cfg.channels * 2  # int16
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        while len(self._byte_buffer) < bytes_needed:
            try:
                packet, address = self.sock.recvfrom(self.cfg.recv_bytes)
            except socket.timeout as exc:
                received = len(self._byte_buffer)
                detail = "no UDP data" if received == 0 else f"wrong sample count: {received}/{bytes_needed} bytes"
                raise CaptureTimeoutError(
                    f"WaveFrag UDP timeout after {self.cfg.timeout_s:g}s ({detail}) on "
                    f"{self.cfg.host}:{self.cfg.port}"
                ) from exc
            if not packet:
                continue
            if self.cfg.expected_source_ip and address[0] != self.cfg.expected_source_ip:
                continue
            if self.cfg.expected_source_port and address[1] != self.cfg.expected_source_port:
                continue
            if self.cfg.expected_datagram_bytes and len(packet) != self.cfg.expected_datagram_bytes:
                raise WrongSampleCountError(
                    f"UDP datagram has {len(packet)} bytes; recorded hardware evidence expects "
                    f"exactly {self.cfg.expected_datagram_bytes} bytes"
                )
            if len(packet) % 2:
                raise WrongSampleCountError(
                    f"UDP datagram has odd byte count {len(packet)}; cannot decode little-endian int16"
                )
            sample_stride_bytes = self.cfg.channels * 2
            if len(packet) % sample_stride_bytes:
                raise ChannelMismatchError(
                    f"UDP datagram has {len(packet)} payload bytes, not a multiple of the "
                    f"{sample_stride_bytes}-byte 128-channel sample stride. A header, partial "
                    "sample, different channel count, or undocumented framing may be present."
                )
            self._datagram_count += 1
            self._datagram_sizes.append(len(packet))
            self._source_addresses.add(address)
            self._byte_buffer.extend(packet)
        raw = bytes(self._byte_buffer[:bytes_needed])
        del self._byte_buffer[:bytes_needed]
        x = np.frombuffer(raw, dtype="<i2")
        if x.size != n_samples * self.cfg.channels:
            raise WrongSampleCountError(
                f"Decoded {x.size} int16 values; expected {n_samples * self.cfg.channels}"
            )
        x = x.reshape(-1, self.cfg.channels)  # samples x raw channels
        self.last_raw_int16 = x.T.copy()  # preserve the actual wire order in NPZ raw_int16
        x, mapping_meta = to_vendor_physical_order(x, channel_axis=1)
        audio = (x.T.astype(np.float32) / 32768.0)
        frame = AcousticFrame(
            audio=audio,
            fs=self.cfg.fs,
            mic_xyz=self.mic_xyz.copy(),
            timestamp=time.time(),
            metadata={
                "wire_format": "payload_only_int16_le",
                **mapping_meta,
                "datagram_count": self._datagram_count,
                "datagram_sizes": sorted(set(self._datagram_sizes)),
                "source_addresses": [f"{ip}:{port}" for ip, port in sorted(self._source_addresses)],
                "packet_header_sequence_timestamp": "UNKNOWN",
            },
        )
        frame.validate()
        return frame

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
