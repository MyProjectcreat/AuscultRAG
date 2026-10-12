"""Selective access to a remote ZIP archive over HTTP Range requests.

Reads the central directory (ZIP64-aware) without downloading the archive, and
extracts individual members by fetching only their byte ranges. Used for corpora
whose archive is far larger than the subset we need (TBScreen: 395 GB).
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import requests

EOCD_SIG = b"PK\x05\x06"
ZIP64_LOCATOR_SIG = b"PK\x06\x07"
ZIP64_EOCD_SIG = b"PK\x06\x06"
CDIR_SIG = b"PK\x01\x02"
LOCAL_SIG = b"PK\x03\x04"


@dataclass(frozen=True)
class ZipMember:
    name: str
    compress_type: int
    compressed_size: int
    file_size: int
    crc32: int
    local_header_offset: int


class RemoteZip:
    def __init__(self, url: str, timeout: float = 120.0):
        self.url = url
        self.timeout = timeout
        self.session = requests.Session()
        head = self.session.head(url, timeout=timeout, allow_redirects=True)
        head.raise_for_status()
        self.size = int(head.headers["Content-Length"])
        # Some servers (e.g. Zenodo) honour Range without advertising Accept-Ranges; probe directly.
        probe = self.session.get(url, headers={"Range": "bytes=0-3"}, timeout=timeout)
        if probe.status_code != 206 or probe.content[:4] != LOCAL_SIG:
            raise RuntimeError(f"Byte-range access unavailable for {url} (status {probe.status_code})")

    def read_range(self, start: int, length: int) -> bytes:
        end = start + length - 1
        resp = self.session.get(self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=self.timeout)
        if resp.status_code != 206:
            raise RuntimeError(f"Expected 206 for range {start}-{end}, got {resp.status_code}")
        data = resp.content
        if len(data) != length:
            raise RuntimeError(f"Short read: wanted {length} bytes, got {len(data)}")
        return data

    def _central_directory_location(self) -> tuple[int, int, int]:
        tail_len = min(self.size, 65536 + 22)
        tail = self.read_range(self.size - tail_len, tail_len)
        idx = tail.rfind(EOCD_SIG)
        if idx < 0:
            raise RuntimeError("End-of-central-directory record not found")
        (_, _, _, _, n_entries, cd_size, cd_offset, _) = struct.unpack("<4sHHHHIIH", tail[idx : idx + 22])
        loc_idx = idx - 20
        if loc_idx >= 0 and tail[loc_idx : loc_idx + 4] == ZIP64_LOCATOR_SIG:
            (_, _, z64_eocd_offset, _) = struct.unpack("<4sIQI", tail[loc_idx : loc_idx + 20])
            z64 = self.read_range(z64_eocd_offset, 56)
            if z64[:4] != ZIP64_EOCD_SIG:
                raise RuntimeError("ZIP64 end-of-central-directory record not found")
            (_, _, _, _, _, _, _, n_entries, cd_size, cd_offset) = struct.unpack("<4sQHHIIQQQQ", z64)
        return n_entries, cd_size, cd_offset

    def members(self) -> list[ZipMember]:
        n_entries, cd_size, cd_offset = self._central_directory_location()
        cd = self.read_range(cd_offset, cd_size)
        out: list[ZipMember] = []
        pos = 0
        while pos < len(cd):
            if cd[pos : pos + 4] != CDIR_SIG:
                raise RuntimeError(f"Bad central directory signature at {pos}")
            fields = struct.unpack("<4sHHHHHHIIIHHHHHII", cd[pos : pos + 46])
            compress_type, crc, csize, usize = fields[4], fields[7], fields[8], fields[9]
            name_len, extra_len, comment_len = fields[10], fields[11], fields[12]
            local_offset = fields[16]
            name = cd[pos + 46 : pos + 46 + name_len].decode("utf-8", errors="replace")
            extra = cd[pos + 46 + name_len : pos + 46 + name_len + extra_len]
            csize, usize, local_offset = _apply_zip64_extra(extra, csize, usize, local_offset)
            out.append(ZipMember(name, compress_type, csize, usize, crc, local_offset))
            pos += 46 + name_len + extra_len + comment_len
        if len(out) != n_entries:
            raise RuntimeError(f"Central directory lists {len(out)} entries, EOCD says {n_entries}")
        return out

    def extract(self, member: ZipMember, dest: Path) -> None:
        header = self.read_range(member.local_header_offset, 30)
        if header[:4] != LOCAL_SIG:
            raise RuntimeError(f"Bad local header for {member.name}")
        name_len, extra_len = struct.unpack("<HH", header[26:30])
        data_start = member.local_header_offset + 30 + name_len + extra_len
        raw = self.read_range(data_start, member.compressed_size) if member.compressed_size else b""
        if member.compress_type == 0:
            data = raw
        elif member.compress_type == 8:
            data = zlib.decompress(raw, -15)
        else:
            raise RuntimeError(f"Unsupported compression {member.compress_type} for {member.name}")
        if zlib.crc32(data) & 0xFFFFFFFF != member.crc32 or len(data) != member.file_size:
            raise RuntimeError(f"CRC/size mismatch for {member.name}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(dest)


def _apply_zip64_extra(extra: bytes, csize: int, usize: int, offset: int) -> tuple[int, int, int]:
    pos = 0
    while pos + 4 <= len(extra):
        header_id, size = struct.unpack("<HH", extra[pos : pos + 4])
        body = extra[pos + 4 : pos + 4 + size]
        if header_id == 0x0001:
            vals = list(struct.unpack("<" + "Q" * (len(body) // 8), body[: (len(body) // 8) * 8]))
            if usize == 0xFFFFFFFF and vals:
                usize = vals.pop(0)
            if csize == 0xFFFFFFFF and vals:
                csize = vals.pop(0)
            if offset == 0xFFFFFFFF and vals:
                offset = vals.pop(0)
        pos += 4 + size
    return csize, usize, offset
