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

# 根因已确认(见 README/log-analysis.md 第十四节),这两个文件同时带上了 dsv4pro-cp-fix-v2/
# 的修复(padding 槽 position 不再撞车到 0),用于验证修复是否解决乱码。如果只想要纯诊断、
# 不带修复的镜像,去掉下面这两行 COPY 即可。
COPY patches/layers/attention/deepseek_v4_backend.py /sgl-workspace/sglang/python/sglang/srt/layers/attention/deepseek_v4_backend.py
COPY patches/layers/attention/deepseek_v4_backend_hip_radix.py /sgl-workspace/sglang/python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py

# 第十八节:C4/C128 压缩 KV 写入路径打点。compressor.py(v1)已确认是死代码——GPU 后端实际
# import 的 CompressorBackendMixin 来自 compressor_v2.py,所以这里改成打在真正生效的
# compressor_v2.py 上,核实"8 个 cp_rank 是不是真的冗余算出完全一样的 kv_score_input/
# kv_compressed/out_loc"这个假设。
COPY patches/layers/attention/dsv4/compressor_v2.py /sgl-workspace/sglang/python/sglang/srt/layers/attention/dsv4/compressor_v2.py

# 第十九节:GitHub 调研发现 CP 开启时会被无条件强制禁用
# SGLANG_OPT_FLASHMLA_SPARSE_PREFILL,导致 prefill 100% 被切到 flash_mla_with_kvcache
# 这条历史更久、验证没那么充分的 kernel 路径。加了一个新的测试开关
# SGLANG_TEST_CP_KEEP_SPARSE_PREFILL=1,可以在不改动其它行为的前提下让 CP 也走
# _forward_prefill_sparse(flash_mla_sparse_fwd),验证乱码是否消失。
COPY patches/environ.py /sgl-workspace/sglang/python/sglang/srt/environ.py
COPY patches/arg_groups/deepseek_v4_hook.py /sgl-workspace/sglang/python/sglang/srt/arg_groups/deepseek_v4_hook.py

RUN python3 -m py_compile \
    /sgl-workspace/sglang/python/sglang/srt/utils/cp_pp_debug.py \
    /sgl-workspace/sglang/python/sglang/srt/models/deepseek_v4.py \
    /sgl-workspace/sglang/python/sglang/srt/managers/scheduler_pp_mixin.py \
    /sgl-workspace/sglang/python/sglang/srt/layers/attention/deepseek_v4_backend.py \
    /sgl-workspace/sglang/python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py \
    /sgl-workspace/sglang/python/sglang/srt/layers/attention/dsv4/compressor_v2.py \
    /sgl-workspace/sglang/python/sglang/srt/environ.py \
    /sgl-workspace/sglang/python/sglang/srt/arg_groups/deepseek_v4_hook.py
