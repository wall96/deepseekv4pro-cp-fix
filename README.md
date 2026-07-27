# deepseekv4pro-cp-fix

DeepSeek-V4-Pro 在 tp8pp4(H100,PD 分离式部署)下开启 prefill context-parallel
(`--enable-dsa-prefill-context-parallel --dsa-prefill-cp-mode round-robin-split`)
出现输出乱码问题的排查记录与配套代码。基础镜像:
`harbor.local.clusters/bp/lmsysorg/sglang:v0.5.15-deepseekV4Pro-0723`。

排查过程的完整记录(时间线、每一轮实验、每一次假设的证伪/证实)不在本仓库,单独维护在
排查现场的 `dsv4pro/log-analysis.md` 里,本仓库只包含**跟着排查过程产出的、要 build 成镜像的代码**。

## 目录

- **`dsv4pro-cp-debug/`** —— 诊断工具。**不修复问题**,只在 CP/PP 数据流的
  7 个关键手交点打印 tensor 统计量(mean/absmax/norm/has_nan/has_inf),通过
  `SGLANG_CP_DEBUG_DUMP=1` 环境变量开关,默认零开销。用于定位乱码到底是从哪一层/哪一步开始出现的。
  详见该目录下的 `README.md`。这个镜像现在同时也带上了 `dsv4pro-cp-fix-v2/` 的修复,方便一边跑
  一边用 `[CP_DEBUG]` 输出验证修复效果。

- **`dsv4pro-cp-fix-v2/`** —— **当前有效的修复**。根因:DSA prefill CP round-robin 把 batch
  token 数 pad 到 `cp_size` 整数倍时,padding 槽的 `seq_lens_casual` 用常数 `1` 填充,导致
  `positions_casual`(= `seq_lens_casual - 1`)恒为 0,跟真实的 position-0 token 撞车,round-robin
  切分后部分 cp_rank 会把这个撞车的 padding 槽当成对 position-0 token 的又一次合法计算,污染最终
  输出。修复:把 padding 槽的 position 改成从最后一个真实 token 连续递增,不再跟任何真实 position
  撞车。详见该目录下的 `README.md`。

- **`dsv4pro-cp-fix/`** —— **已废弃,不要 build/部署**,保留仅作为排查历史记录。这是排查早期基于
  一次已被证伪的假设(对照 sgl-project/sglang#20360 那次针对旧 NSA 代码的诊断)做的修复尝试,后来
  确认 `DeepseekV4ForCausalLM` 根本不走这次改动涉及的 `LayerCommunicator` 代码路径,这次修复对
  实际的 forward 路径完全不生效。目录顶部的 `README.md` 有完整的废弃说明。

## 现状

根因已确认并已实施修复(`dsv4pro-cp-fix-v2/`),过程记录在 `dsv4pro/log-analysis.md` 第十一至
十四节:先排除了 PP 跨 stage 的 rank 对应关系、"CP-V2" 新框架冲突、`forward_metadata` 初始化顺序、
MoE runner backend marlin→triton(triton 本身跟 PP>1 不兼容,是另一个已知的、无关的社区 bug,见
sgl-project/sglang#27109/#27497)等假设;再用 `dsv4pro-cp-debug/` 插桩跑出真实日志,确认 PP0→PP1
的 send/recv 传输本身是干净的,但 `G_post_cp_reindex_positions` 显示 CP round-robin 的 padding
槽 position 撞车到 0;最后读代码定位到 `expand_prefill_casually`/`_expand_prefill_casually_vectorized`
(`deepseek_v4_backend.py`/`deepseek_v4_backend_hip_radix.py`)里 `F.pad(..., value=1)` 是根因,
并已修复。**修复尚待用真实请求做端到端验证(乱码是否消失)**,验证方法见
`dsv4pro/log-analysis.md` 第十四节末尾。
