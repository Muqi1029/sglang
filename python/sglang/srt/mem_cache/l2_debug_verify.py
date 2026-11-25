"""Opt-in, synchronous byte verification for local HiCache debugging only.

Keep independent CPU snapshots of D2H source bytes. Poison H2D destinations
before copying and compare every restored byte before releasing layer events.
Unsupported pool geometries fail explicitly rather than silently passing.
"""

import json
import os
from pathlib import Path

import torch


class L2ByteVerifier:
    def __init__(self, directory):
        self.path = Path(directory) / f"bytes-{os.getpid()}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.saved = {}

    def _rows(self, t, layer=None):
        p = t.host_pool
        h, d = t.host_indices, t.device_indices
        if getattr(p, "dcp_size", 1) != 1:
            raise RuntimeError("L2 debug verifier requires DCP=1")
        name = type(p).__name__
        if name == "MLATokenToKVPoolHost":
            for lid, bid in p._buffer_index_by_layer.items():
                if layer is None or lid == layer:
                    yield f"kv.layer{lid}", p.device_kv_buffers[bid], h, d
        elif name == "DSAIndexerPoolHost":
            h, d = p._get_indexer_page_indices(h, d)
            for lid, bid in p._device_to_host_layer.items():
                if layer is None or lid == layer:
                    yield (
                        f"indexer.layer{lid}",
                        p.packed_device_index_buffers[bid],
                        h,
                        d,
                    )
        elif name == "MHATokenToKVPoolHost":
            for lid, (k, v) in enumerate(zip(p.device_k_buffers, p.device_v_buffers)):
                if layer is None or lid == layer:
                    yield f"k.layer{lid}", k, h, d
                    yield f"v.layer{lid}", v, h, d
        elif name == "MambaPoolHost":
            temporal = (
                p.temporal_device_layers
                if p.mtp_draft_buffer_infos
                else p.buffer_info.temporal
            )
            conv = (
                p.conv_device_layers if p.mtp_draft_buffer_infos else p.buffer_info.conv
            )
            for lid in range(p.num_mamba_layers):
                if layer is not None and lid != layer:
                    continue
                if p.temporal_state_elem_size > 0:
                    yield f"temporal.layer{lid}", temporal[lid], h, d
                for ci, layers in enumerate(conv):
                    yield f"conv{ci}.layer{lid}", layers[lid], h, d
            if layer is None or layer == 0:
                for si, state in enumerate(p.slot_state_device_tensors):
                    yield f"slot_state{si}", state, h, d
        else:
            raise RuntimeError(f"Unsupported L2 debug pool: {name}")

    @staticmethod
    def _bytes(buf):
        # Retain actual row strides; never poison a detached contiguous copy.
        return buf.view(torch.uint8).flatten(start_dim=1)

    def snapshot(self, transfer):
        for label, buf, h, d in self._rows(transfer):
            ids = h.cpu().tolist()
            data = self._bytes(buf).index_select(0, d.to(buf.device).long()).cpu()
            key = (id(transfer.host_pool), label)
            saved = self.saved.setdefault(key, {})
            for i, host_id in enumerate(ids):
                saved[host_id] = data[i].clone()

    def poison(self, transfer, layer):
        for label, buf, h, d in self._rows(transfer, layer):
            raw = self._bytes(buf)
            if raw.untyped_storage().data_ptr() != buf.untyped_storage().data_ptr():
                raise RuntimeError(f"Non-view byte layout for {label}")
            raw.index_fill_(0, d.to(buf.device).long(), 165)

    def compare(self, transfer, layer):
        for label, buf, h, d in self._rows(transfer, layer):
            ids = h.cpu().tolist()
            if not ids:
                continue
            saved = self.saved.get((id(transfer.host_pool), label), {})
            missing = [i for i in ids if i not in saved]
            if missing:
                raise RuntimeError(f"Missing L1 baseline for {label}: {missing[:8]}")
            expected = torch.stack([saved[i] for i in ids])
            actual = self._bytes(buf).index_select(0, d.to(buf.device).long()).cpu()
            neq = actual != expected
            mismatch = int(neq.sum())
            record = dict(
                pool=type(transfer.host_pool).__name__,
                component=label,
                rows=len(ids),
                bytes=actual.numel(),
                mismatch_bytes=mismatch,
                poisoned=True,
            )
            if mismatch:
                first = neq.nonzero()[0].tolist()
                record.update(
                    first_host_row=ids[first[0]],
                    first_byte_offset=first[1],
                    expected=int(expected[tuple(first)]),
                    actual=int(actual[tuple(first)]),
                )
            with self.path.open("a") as f:
                f.write(json.dumps(record) + "\n")
            if mismatch:
                raise RuntimeError(f"L2 byte mismatch: {record}")
