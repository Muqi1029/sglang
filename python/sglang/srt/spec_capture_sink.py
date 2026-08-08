# Copyright 2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Server-side spec-training capture sink (SpecForge DataFlow transport).

Under ``--enable-spec-capture``, a request's ``spec_capture`` dict tells this
sink to write the prefill's captured tensors straight into a Mooncake store
(one hard-pinned object per tensor at ``{store_id}/{sample_id}/g{gen}/{name}``,
raw bytes — shape/dtype travel on the returned spec). Feature tensors never
touch the response path; ``meta_info["spec_capture"]`` returns only keys +
shapes/dtypes. Strategy naming is the client's (the ``features`` mapping); the
server knows only generic artifacts. Self-contained: the scheduler hooks are
one-liners, every capture decision lives here.

Request schema::

    {"store_id", "sample_id", "gen", "replace", # key namespace / retry policy
     "features": {"aux": <name>, "last_hidden": <name>},   # artifact -> feature
     "passthrough": [{"name", "data", "shape", "dtype"}]}  # client tensors verbatim

Response (``meta_info["spec_capture"]``): ``{"sample_id", "store_id", "gen",
"aux_layer_ids", "features": {name: {"shape", "dtype"}}}``.

Mooncake connection uses the standard ``MOONCAKE_*`` env vars (see
``MooncakeFeatureStore``).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# torch dtype -> the FeatureSpec dtype string SpecForge's zero-copy get() maps
# back to a torch dtype. Keep in sync with MooncakeFeatureStore._TORCH_DTYPES.
_DTYPE_STR = {
    torch.float32: "float32",
    torch.float64: "float64",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.int64: "int64",
    torch.int32: "int32",
    torch.int16: "int16",
    torch.int8: "int8",
    torch.uint8: "uint8",
    torch.bool: "bool",
}
_STR_DTYPE = {v: k for k, v in _DTYPE_STR.items()}

_ARTIFACT_AUX = "aux"
_ARTIFACT_LAST_HIDDEN = "last_hidden"


class SpecCaptureSink:
    """Writes captured per-request tensors into Mooncake in SpecForge layout."""

    def __init__(self, aux_layer_ids: Optional[List[int]] = None) -> None:
        self.aux_layer_ids = list(aux_layer_ids) if aux_layer_ids else None
        self._store = None
        self._put_config = None
        self._lock = threading.Lock()
        # One store writer is sufficient: Mooncake already stripes a batched
        # transfer internally. The executor decouples that host transfer from
        # the scheduler so the next target prefill can run concurrently.
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="spec-capture-batch-put",
        )

    # -- connection ---------------------------------------------------------
    def _connect(self):
        if self._store is not None:
            return self._store
        with self._lock:
            if self._store is not None:
                return self._store
            from mooncake.store import MooncakeDistributedStore, ReplicateConfig

            store = MooncakeDistributedStore()
            rc = store.setup(
                local_hostname=os.environ.get("MOONCAKE_LOCAL_HOSTNAME", "localhost"),
                metadata_server=os.environ.get(
                    "MOONCAKE_METADATA_SERVER", "http://localhost:8080/metadata"
                ),
                global_segment_size=int(
                    os.environ.get("MOONCAKE_GLOBAL_SEGMENT_SIZE", 1 << 30)
                ),
                local_buffer_size=int(
                    os.environ.get("MOONCAKE_LOCAL_BUFFER_SIZE", 1 << 30)
                ),
                protocol=os.environ.get("MOONCAKE_PROTOCOL", "tcp"),
                rdma_devices=os.environ.get("MOONCAKE_RDMA_DEVICES", ""),
                master_server_addr=os.environ.get(
                    "MOONCAKE_MASTER_SERVER_ADDR", "localhost:50051"
                ),
            )
            if rc is not None and int(rc) != 0:
                raise RuntimeError(f"spec-capture mooncake setup failed (status {rc})")
            # Hard-pin every object: SpecForge (not Mooncake's LRU) is the
            # lifetime authority — a committed feature must never be evicted
            # before the trainer consumes it.
            cfg = ReplicateConfig()
            cfg.replica_num = 1
            cfg.with_hard_pin = True
            self._put_config = cfg
            self._store = store
            logger.info("spec-capture mooncake sink connected")
            return store

    # -- key/put primitives (pinned to MooncakeFeatureStore's layout) --------
    @staticmethod
    def _tkey(store_id: str, sample_id: str, gen: int, name: str) -> str:
        return f"{store_id}/{sample_id}/g{gen}/{name}"

    def _remove_quiet(self, key: str) -> None:
        try:
            self._connect().remove(key)
        except Exception:
            pass

    def _remove_many_quiet(self, keys: List[str]) -> None:
        if not keys:
            return
        store = self._connect()
        batch_remove = getattr(store, "batch_remove", None)
        if batch_remove is not None:
            try:
                batch_remove(keys)
                return
            except Exception:
                pass
        for key in keys:
            self._remove_quiet(key)

    # -- the batch entry point ------------------------------------------------
    def submit_samples(
        self,
        samples: List[
            Tuple[Dict[str, Any], Optional[torch.Tensor], Optional[torch.Tensor]]
        ],
    ) -> Future[List[Dict[str, Any]]]:
        """Queue one scheduler batch without blocking the scheduler thread."""
        return self._executor.submit(self.put_samples, samples)

    def put_samples(
        self,
        samples: List[
            Tuple[Dict[str, Any], Optional[torch.Tensor], Optional[torch.Tensor]]
        ],
    ) -> List[Dict[str, Any]]:
        """Publish a scheduler batch with one native Mooncake batch RPC.

        The response is emitted only after every object succeeds, so returned
        refs can never point at incomplete samples.
        """
        if not samples:
            return []

        store = self._connect()
        timing_enabled = os.environ.get("SGLANG_SPEC_CAPTURE_TIMING", "0") == "1"
        started = time.perf_counter()
        keys: List[str] = []
        tensors: List[torch.Tensor] = []
        sizes: List[int] = []
        replace_keys: List[str] = []
        results: List[Dict[str, Any]] = []

        def _stage(
            result_feats: Dict[str, Dict[str, Any]],
            *,
            store_id: str,
            sample_id: str,
            gen: int,
            replace: bool,
            name: str,
            tensor: torch.Tensor,
        ) -> None:
            tensor = tensor.detach().to("cpu").contiguous()
            key = self._tkey(store_id, sample_id, gen, name)
            keys.append(key)
            tensors.append(tensor)
            sizes.append(tensor.element_size() * tensor.numel())
            if replace:
                replace_keys.append(key)
            result_feats[name] = {
                "shape": list(tensor.shape),
                "dtype": _DTYPE_STR.get(
                    tensor.dtype, str(tensor.dtype).replace("torch.", "")
                ),
            }

        for spec, aux, last_hidden in samples:
            store_id = str(spec["store_id"])
            sample_id = str(spec["sample_id"])
            gen = int(spec.get("gen", 1))
            replace = bool(spec.get("replace", False))
            features: Dict[str, str] = dict(spec.get("features") or {})
            result_feats: Dict[str, Dict[str, Any]] = {}

            aux_name = features.get(_ARTIFACT_AUX)
            if aux_name is not None:
                if aux is None:
                    raise RuntimeError(
                        "spec_capture requested 'aux' but no aux hidden states were "
                        "captured — launch the server with --enable-spec-capture "
                        "(and optionally --spec-capture-aux-layer-ids)"
                    )
                _stage(
                    result_feats,
                    store_id=store_id,
                    sample_id=sample_id,
                    gen=gen,
                    replace=replace,
                    name=aux_name,
                    tensor=aux.unsqueeze(0),
                )
            lh_name = features.get(_ARTIFACT_LAST_HIDDEN)
            if lh_name is not None:
                if last_hidden is None:
                    raise RuntimeError(
                        "spec_capture requested 'last_hidden' but the logits "
                        "processor did not return it (is aux capture enabled?)"
                    )
                _stage(
                    result_feats,
                    store_id=store_id,
                    sample_id=sample_id,
                    gen=gen,
                    replace=replace,
                    name=lh_name,
                    tensor=last_hidden.unsqueeze(0),
                )
            for item in spec.get("passthrough") or []:
                dtype = _STR_DTYPE.get(str(item.get("dtype", "int64")))
                if dtype is None:
                    raise RuntimeError(
                        f"spec_capture passthrough {item.get('name')!r}: "
                        f"unsupported dtype {item.get('dtype')!r}"
                    )
                t = torch.tensor(item["data"], dtype=dtype).reshape(
                    [int(d) for d in item["shape"]]
                )
                _stage(
                    result_feats,
                    store_id=store_id,
                    sample_id=sample_id,
                    gen=gen,
                    replace=replace,
                    name=str(item["name"]),
                    tensor=t,
                )
            results.append(
                {
                    "sample_id": sample_id,
                    "store_id": store_id,
                    "gen": gen,
                    "aux_layer_ids": self.aux_layer_ids,
                    "features": result_feats,
                }
            )

        materialize_ms = (time.perf_counter() - started) * 1000.0
        self._remove_many_quiet(replace_keys)
        registered: List[torch.Tensor] = []
        register_started = time.perf_counter()
        try:
            for tensor, nbytes in zip(tensors, sizes):
                try:
                    store.register_buffer(tensor.data_ptr(), nbytes)
                    registered.append(tensor)
                except Exception:
                    pass  # TCP and some Mooncake builds auto-register
            register_ms = (time.perf_counter() - register_started) * 1000.0
            put_started = time.perf_counter()
            batch_put = getattr(store, "batch_put_from", None)
            if batch_put is None:
                statuses = [
                    store.put_from(key, tensor.data_ptr(), nbytes, self._put_config)
                    for key, tensor, nbytes in zip(keys, tensors, sizes)
                ]
            else:
                statuses = batch_put(
                    keys,
                    [tensor.data_ptr() for tensor in tensors],
                    sizes,
                    self._put_config,
                )
            put_ms = (time.perf_counter() - put_started) * 1000.0
        except Exception:
            self._remove_many_quiet(keys)
            raise
        finally:
            for tensor in registered:
                try:
                    store.unregister_buffer(tensor.data_ptr())
                except Exception:
                    pass

        if statuses is None:
            statuses = [0] * len(keys)
        if len(statuses) != len(keys):
            self._remove_many_quiet(keys)
            raise RuntimeError(
                "spec-capture batch_put_from returned "
                f"{len(statuses)} statuses for {len(keys)} keys"
            )
        failed = [
            (key, status)
            for key, status in zip(keys, statuses)
            if status is not None and int(status) < 0
        ]
        if failed:
            self._remove_many_quiet(keys)
            raise RuntimeError(
                "spec-capture batch_put_from failed for "
                f"{len(failed)}/{len(keys)} keys; first={failed[0]}"
            )

        if timing_enabled:
            logger.info(
                "[spec-capture-timing] batch_sink samples=%d objects=%d "
                "bytes=%d materialize_ms=%.3f register_ms=%.3f put_ms=%.3f "
                "total_ms=%.3f",
                len(samples),
                len(keys),
                sum(sizes),
                materialize_ms,
                register_ms,
                put_ms,
                (time.perf_counter() - started) * 1000.0,
            )
        return results

    def put_sample(
        self,
        spec: Dict[str, Any],
        *,
        aux: Optional[torch.Tensor],
        last_hidden: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        """Keep the single-sample entry point for external callers and tests."""
        return self.put_samples([(spec, aux, last_hidden)])[0]


_SINK: Optional[SpecCaptureSink] = None


def maybe_init_sink(server_args) -> None:
    """Called from Scheduler init on the writer rank when spec capture is on.

    Connection to Mooncake is lazy (first put), so a capture-enabled server
    without a reachable Mooncake master still boots and serves normal traffic.
    """
    global _SINK
    if getattr(server_args, "enable_spec_capture", False) and _SINK is None:
        _SINK = SpecCaptureSink(
            aux_layer_ids=getattr(server_args, "spec_capture_aux_layer_ids", None)
        )


def get_sink() -> Optional[SpecCaptureSink]:
    return _SINK
