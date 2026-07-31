# DeepSeek-V4-Pro FP8 prefill weight-load crash -- pure diagnostic image (no behavior change)
#
# The FP8 checkpoint (DeepSeek-V4-Pro-deepgemm-fp8-hf) fails to load on B300
# single-node prefill (pp_size=1) with:
#   ValueError: Downcasting not allowed: target.dtype=torch.float8_e4m3fn,
#   loaded_weight.dtype=torch.bfloat16
# raised from layers/parameter.py::copy_with_check, reached via
# layers/linear.py::weight_loader_v2. The bare ValueError never carries the
# name of the offending weight (the try/except in models/deepseek_v4.py that
# would normally attach `name=` via e.add_note only wraps the synchronous
# "submit to executor" loop, not the actual weight_loader call that runs
# inside the ThreadPoolExecutor and raises via future.result()).
#
# This image patches ONLY layers/linear.py, taken directly from the official
# v0.5.15 tag source (not from any other patched copy), to:
#   1. Store `self.prefix` unconditionally in LinearBase.__init__ (previously
#      only MergedColumnParallelLinear/MergedColumnParallelRepeatedLinear set
#      it themselves).
#   2. Wrap the 4 weight_loader_v2 implementations that can reach
#      copy_with_check (ColumnParallelLinear, MergedColumnParallelLinear,
#      QKVParallelLinear, RowParallelLinear) in try/except that calls
#      e.add_note(...) with the layer prefix, param type, dtypes and shapes,
#      then re-raises unchanged.
#
# No control flow, no tensor value, no quantization decision is altered.
# When the crash does NOT happen, this image behaves identically to the
# official one. When it DOES happen, the crash's traceback now includes an
# extra note line starting with "[FP8_DEBUG] ...prefix=..." identifying the
# exact failing weight -- grep the crash log for "[FP8_DEBUG]".
#
# Build:
#   docker build -t harbor.local.clusters/bp/lmsysorg/sglang:v0.5.15-deepseekV4Pro-0723-fp8debug .
FROM harbor.local.clusters/bp/lmsysorg/sglang:v0.5.15-deepseekV4Pro-0723

COPY patches/layers/linear.py /sgl-workspace/sglang/python/sglang/srt/layers/linear.py

RUN python3 -m py_compile \
    /sgl-workspace/sglang/python/sglang/srt/layers/linear.py
