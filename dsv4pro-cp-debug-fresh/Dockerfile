# DeepSeek-V4-Pro CP(round-robin-split)精度问题 -- 全新、干净的纯诊断镜像(不含任何历史改动)
#
# 这份镜像是从官方 v0.5.15 tag 源码干干净净重新建的,不叠加之前任何一轮排查加过的
# 代码/打点(不带 padding position 撞车的修复,不带 F-N 那一串历史打点)。只加打点
# (tag 都是 `[CP_DEBUG2]`,跟之前的 `[CP_DEBUG]` 明确区分开),
# 都是 logger.warning 读取张量数值打日志,不修改任何张量、不改变任何控制流。
#
# 默认不开启(SGLANG_CP_DEBUG_DUMP 环境变量控制),不开的话行为跟官方镜像完全一样。
#
# Build:
#   docker build -t harbor.local.clusters/bp/lmsysorg/sglang:v0.5.15-deepseekV4Pro-0723-cpdebug2 .
FROM harbor.local.clusters/bp/lmsysorg/sglang:v0.5.15-deepseekV4Pro-0723

COPY patches/utils/cp_pp_debug2.py /sgl-workspace/sglang/python/sglang/srt/utils/cp_pp_debug2.py
COPY patches/layers/attention/deepseek_v4_backend.py /sgl-workspace/sglang/python/sglang/srt/layers/attention/deepseek_v4_backend.py
COPY patches/models/deepseek_v4.py /sgl-workspace/sglang/python/sglang/srt/models/deepseek_v4.py

RUN python3 -m py_compile \
    /sgl-workspace/sglang/python/sglang/srt/utils/cp_pp_debug2.py \
    /sgl-workspace/sglang/python/sglang/srt/layers/attention/deepseek_v4_backend.py \
    /sgl-workspace/sglang/python/sglang/srt/models/deepseek_v4.py
