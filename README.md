# deepseekv4pro-cp-fix

DeepSeek-V4-Pro 在 tp8pp4(H100,PD 分离式部署)下开启 prefill context-parallel
(`--enable-dsa-prefill-context-parallel --dsa-prefill-cp-mode round-robin-split`)
出现输出乱码问题的排查记录与配套代码。基础镜像:
`harbor.local.clusters/bp/lmsysorg/sglang:v0.5.15-deepseekV4Pro-0723`。

排查过程的完整记录(时间线、每一轮实验、每一次假设的证伪/证实)不在本仓库,单独维护在
排查现场的 `dsv4pro/log-analysis.md` 里,本仓库只包含**跟着排查过程产出的、要 build 成镜像的代码**。

## 目录

- **`dsv4pro-cp-debug/`** —— 当前有效、正在使用的诊断工具。**不修复任何问题**,只在 CP/PP 数据流的
  7 个关键手交点打印 tensor 统计量(mean/absmax/norm/has_nan/has_inf),通过
  `SGLANG_CP_DEBUG_DUMP=1` 环境变量开关,默认零开销。用于定位乱码到底是从哪一层/哪一步开始出现的。
  详见该目录下的 `README.md`。

- **`dsv4pro-cp-fix/`** —— **已废弃,不要 build/部署**,保留仅作为排查历史记录。这是排查早期基于
  一次已被证伪的假设(对照 sgl-project/sglang#20360 那次针对旧 NSA 代码的诊断)做的修复尝试,后来
  确认 `DeepseekV4ForCausalLM` 根本不走这次改动涉及的 `LayerCommunicator` 代码路径,这次修复对
  实际的 forward 路径完全不生效。目录顶部的 `README.md` 有完整的废弃说明。

## 现状

截至最近一次更新,排查已经排除了以下几个方向:PP 跨 stage 的 rank 对应关系(架构上看是自洽的)、
"CP-V2" 新框架冲突(未被激活,与本问题无关)、`forward_metadata` 初始化顺序(顺序是对的)、
MoE runner backend 从 marlin 换成 triton(triton 本身跟 PP>1 不兼容,是另一个已知的、无关的社区
bug,见 sgl-project/sglang#27109/#27497)。当前处于:**已插桩,等待一次真实的 CP-on 运行日志来
定位具体分叉点**,尚未有确认的根因或修复。
