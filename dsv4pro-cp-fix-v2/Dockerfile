# DeepSeek-V4-Pro CP(round-robin-split)精度问题 -- 修复镜像
#
# 修复 dsv4pro/log-analysis.md 第十四节确认的根因:CP round-robin 把 batch token 数 pad 到
# cp_size 整数倍时,padding 槽的 seq_lens_casual 用常数 1 填充(-> position 恒为 0),跟真实的
# position-0 token 撞车。改成从最后一个真实 token 之后连续递增,消除撞车。
#
# 这是纯修复镜像,不含 dsv4pro-cp-debug/ 那套 [CP_DEBUG] 诊断插桩。如果要一边验证一边看
# debug 输出,用 dsv4pro-cp-debug/(已经把这份修复也同步进去了)。
#
# Build:
#   docker build -t harbor.local.clusters/bp/lmsysorg/sglang:v0.5.15-deepseekV4Pro-0723-cpfix-v2 .
FROM harbor.local.clusters/bp/lmsysorg/sglang:v0.5.15-deepseekV4Pro-0723

COPY patches/layers/attention/deepseek_v4_backend.py /sgl-workspace/sglang/python/sglang/srt/layers/attention/deepseek_v4_backend.py
COPY patches/layers/attention/deepseek_v4_backend_hip_radix.py /sgl-workspace/sglang/python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py

RUN python3 -m py_compile \
    /sgl-workspace/sglang/python/sglang/srt/layers/attention/deepseek_v4_backend.py \
    /sgl-workspace/sglang/python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py
