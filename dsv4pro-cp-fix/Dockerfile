# Fixes 3 DSA prefill-CP (round-robin-split) correctness bugs that produce
# garbled/incoherent output when running DeepSeek-V4-Pro with
# --enable-dsa-prefill-context-parallel --dsa-prefill-cp-mode round-robin-split.
#
# All three bugs were originally diagnosed and fixed once against the old
# NSA-named communicator (sgl-project/sglang#20360, communicator_nsa_cp.py),
# but the fix was never ported when that module was renamed to
# communicator_dsa_cp.py during the NSA -> DSA rename (#25821). See README.md
# in this folder for the full analysis.
#
# Build:
#   docker build -t harbor.local.clusters/bp/lmsysorg/sglang:v0.5.15-deepseekV4Pro-0723-cpfix .
FROM harbor.local.clusters/bp/lmsysorg/sglang:v0.5.15-deepseekV4Pro-0723

COPY patches/communicator.py /sgl-workspace/sglang/python/sglang/srt/layers/communicator.py
COPY patches/communicator_dsa_cp.py /sgl-workspace/sglang/python/sglang/srt/layers/communicator_dsa_cp.py

# sglang is an editable install (pip install -e) from /sgl-workspace/sglang in
# the base image, so overwriting the .py files above is sufficient -- no
# reinstall/rebuild step is needed for a pure-Python change like this one.
RUN python3 -m py_compile \
    /sgl-workspace/sglang/python/sglang/srt/layers/communicator.py \
    /sgl-workspace/sglang/python/sglang/srt/layers/communicator_dsa_cp.py
