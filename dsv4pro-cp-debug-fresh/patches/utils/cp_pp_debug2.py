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

"""Fresh, from-scratch diagnostic instrumentation for the DeepSeek-V4-Pro
prefill context-parallel (round-robin-split) precision-loss investigation.

This is a clean rebuild starting from the pristine v0.5.15 tag source --
deliberately does NOT carry over any of the accumulated history from earlier
investigation rounds (no padding fix, no other behavior changes). This file,
and every call site that uses it, only ever calls ``logger.warning`` -- it
never mutates a tensor or influences control flow. Gated by
SGLANG_CP_DEBUG_DUMP=1 (a plain os.environ check). Zero-cost when unset.
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
    """Log cheap stats (and, for small tensors, full values) for one tensor
    at one hand-off point. Read-only: never mutates ``tensor``."""
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
        logger.warning("[CP_DEBUG2] %s NONE_TENSOR", kv)
        return
    if tensor.numel() == 0:
        logger.warning("[CP_DEBUG2] %s shape=%s EMPTY", kv, tuple(tensor.shape))
        return

    x = tensor.detach()
    if not torch.is_floating_point(x):
        logger.warning(
            "[CP_DEBUG2] %s shape=%s dtype=%s min=%s max=%s sum=%s",
            kv,
            tuple(x.shape),
            x.dtype,
            x.min().item(),
            x.max().item(),
            x.sum().item(),
        )
        if x.dim() == 1 and x.numel() <= 64:
            logger.warning("[CP_DEBUG2] %s values=%s", kv, x.tolist())
        elif x.dim() == 2 and x.shape[0] <= 16 and x.numel() <= 4096:
            logger.warning("[CP_DEBUG2] %s values=%s", kv, x.tolist())
        return

    xf = x.float()
    logger.warning(
        "[CP_DEBUG2] %s shape=%s dtype=%s mean=%.6g absmax=%.6g norm=%.6g "
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
