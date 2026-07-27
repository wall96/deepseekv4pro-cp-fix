# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Temporary, throwaway diagnostic instrumentation for the DeepSeek-V4-Pro
prefill context-parallel (round-robin-split) precision-loss investigation.

Gated by SGLANG_CP_DEBUG_DUMP=1 (a plain os.environ check, not a registered
server-arg env var -- this is scratch debugging code, not a shipped feature).
Zero-cost when the env var is unset.

Not meant to be a permanent addition; delete once the root cause is found and
fixed. See maas/dsv4pro-cp-debug/README.md in the investigation repo for the
full list of instrumentation points and how to read the output.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

_CP_DEBUG_DUMP = os.environ.get("SGLANG_CP_DEBUG_DUMP", "0") not in ("0", "", "false", "False")


def cp_debug_enabled() -> bool:
    return _CP_DEBUG_DUMP


def cp_debug_dump(
    tag: str,
    tensor: Optional[torch.Tensor],
    *,
    layer_id: Optional[int] = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """Log cheap stats for one tensor at one CP/PP hand-off point.

    `extra` lets call sites attach point-specific context (e.g. mb_id,
    metadata object id()) without every call site needing its own logging
    boilerplate.
    """
    if not _CP_DEBUG_DUMP:
        return

    from sglang.srt.runtime_context import get_parallel

    ps = get_parallel()
    base = {
        "tag": tag,
        "layer": layer_id,
        "pp_rank": ps.pp_rank,
        "cp_rank": getattr(ps, "attn_cp_rank", None),
        "tp_rank": getattr(ps, "attn_tp_rank", None),
    }
    if extra:
        base.update(extra)
    kv = " ".join(f"{k}={v}" for k, v in base.items())

    if tensor is None:
        logger.warning("[CP_DEBUG] %s NONE_TENSOR", kv)
        return
    if tensor.numel() == 0:
        logger.warning("[CP_DEBUG] %s shape=%s EMPTY", kv, tuple(tensor.shape))
        return

    x = tensor.detach()
    if not torch.is_floating_point(x):
        # e.g. int32 next_token_ids -- stats are still meaningful, just skip
        # nan/inf checks which don't apply to integer dtypes.
        logger.warning(
            "[CP_DEBUG] %s shape=%s dtype=%s min=%s max=%s sum=%s",
            kv,
            tuple(x.shape),
            x.dtype,
            x.min().item(),
            x.max().item(),
            x.sum().item(),
        )
        return

    xf = x.float()
    logger.warning(
        "[CP_DEBUG] %s shape=%s dtype=%s mean=%.6g absmax=%.6g norm=%.6g "
        "has_nan=%s has_inf=%s",
        kv,
        tuple(x.shape),
        x.dtype,
        xf.mean().item(),
        xf.abs().max().item(),
        xf.norm().item(),
        bool(torch.isnan(xf).any().item()),
        bool(torch.isinf(xf).any().item()),
    )


def cp_debug_dump_dict(
    tag: str,
    tensors: Optional[dict],
    *,
    layer_id: Optional[int] = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """Same as cp_debug_dump, but for a {name: tensor} dict (e.g. the PP
    proxy-tensor payload, which may carry more than one named tensor)."""
    if not _CP_DEBUG_DUMP:
        return
    if not tensors:
        cp_debug_dump(tag, None, layer_id=layer_id, extra=extra)
        return
    for name, t in tensors.items():
        merged_extra = dict(extra or {})
        merged_extra["name"] = name
        cp_debug_dump(tag, t, layer_id=layer_id, extra=merged_extra)
