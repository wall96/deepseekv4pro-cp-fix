# DeepSeek-V4-Pro CP(round-robin-split)精度问题 -- 调试插桩镜像
#
# 这不是一个"修复"镜像,是一个临时诊断工具:在 CP/PP 数据流的 7 个关键手交点上打印
# tensor 的统计量(mean/absmax/norm/has_nan/has_inf),打了 layer_id / pp_rank /
# attn_cp_rank / mb_id 标签,方便把 CP-on 和 CP-off(或不同 PP 拓扑)两次跑的日志
# 对齐 diff,找到数值第一次开始分叉的那一层/那一步。7 个点具体是什么见 README.md。
#
# 默认不开启(SGLANG_CP_DEBUG_DUMP 环境变量控制),不影响正常运行时的性能/行为。
#
# Build:
#   docker build -t harbor.local.clusters/bp/lmsysorg/sglang:v0.5.15-deepseekV4Pro-0723-cpdebug .
#
# 使用方法见 README.md。
FROM harbor.local.clusters/bp/lmsysorg/sglang:v0.5.15-deepseekV4Pro-0723

COPY patches/utils/cp_pp_debug.py /sgl-workspace/sglang/python/sglang/srt/utils/cp_pp_debug.py
COPY patches/models/deepseek_v4.py /sgl-workspace/sglang/python/sglang/srt/models/deepseek_v4.py
COPY patches/managers/scheduler_pp_mixin.py /sgl-workspace/sglang/python/sglang/srt/managers/scheduler_pp_mixin.py

RUN python3 -m py_compile \
    /sgl-workspace/sglang/python/sglang/srt/utils/cp_pp_debug.py \
    /sgl-workspace/sglang/python/sglang/srt/models/deepseek_v4.py \
    /sgl-workspace/sglang/python/sglang/srt/managers/scheduler_pp_mixin.py
