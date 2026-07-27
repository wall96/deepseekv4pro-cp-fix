# ⚠️ 已废弃,不要 build/部署这个镜像

**结论已被推翻:`DeepseekV4ForCausalLM`/`DeepseekV4DecoderLayer`(`models/deepseek_v4.py`)根本不使用
本文档下面分析的 `LayerCommunicator`/`DSACPLayerCommunicator`(`communicator.py`/`communicator_dsa_cp.py`)
这套抽象——那是给 `deepseek_v2.py` 通用 `DeepseekV2DecoderLayer`(DeepSeek V2/V3/V3.2 等模型)用的。
`patches/` 下的两个改动对 V4-Pro 的实际 forward 路径完全不生效,build 出来的镜像跟原镜像行为一样,
不会修好乱码,也不会引入新问题——纯粹是白改。**

最新分析和下一步实验计划见 `../dsv4pro/log-analysis.md` 第十一节。这个文件夹保留仅作反面记录(记录一次
"读错代码路径导致修复无效"的教训),不要基于这里的 Dockerfile 构建镜像。

---

# 以下是原始分析(已知不适用于实际代码路径,仅供参考)

# DeepSeek-V4-Pro prefill-CP (round-robin-split) 乱码修复

## 背景

在 `harbor.local.clusters/bp/lmsysorg/sglang:v0.5.15-deepseekV4Pro-0723` 这个镜像上,给 prefill
加上 `--enable-dsa-prefill-context-parallel --dsa-prefill-cp-mode round-robin-split` 之后,
输出会出现乱码/替换字符,尤其是需要精确计算的 prompt(如算术题)。排查过程见
`../dsv4pro/log-analysis.md`。

## 根因:三个 bug,当年在旧代码里被诊断并修复过一次,但改名重构之后又原样回来了

`sgl-project/sglang` 的 issue/PR **#20360**(2026-03-11,针对当时叫 `communicator_nsa_cp.py`
的文件)诊断出三个导致 round-robin-split 输出乱码的 bug 并给出了修复,在 8×MI300X 上验证有效。
但 2026-06-18,`communicator_nsa_cp.py` 在 NSA→DSA 重命名(#25821)中被删除、内容搬到了
`communicator_dsa_cp.py`,**这次修复没有被移植过去**,维护者关闭该 PR 时明确说"如果新的 DSA
路径上还能复现,欢迎提一个新 PR"。我们对照 v0.5.15 的 `communicator_dsa_cp.py` /
`communicator.py` 源码逐条核实,确认三个 bug 原样都在。

### Bug 1 —— `_gather_hidden_states_and_residual` 里,attn-TP all-reduce 被漏掉了

`layers/communicator_dsa_cp.py` 里 `DSACPCommunicateWithAllReduceAndLayerNormFn` 这个类,
docstring 写着"1. All reduce in tp_attn_group on hidden_states / 2. Apply layer norm",但
函数体从头到尾只调了 `layernorm`,从没调用过任何 all-reduce。attention 的输出投影(`o_proj`)
是 `RowParallelLinear`,每个 attn-TP rank 算出来的只是部分和,不做 all-reduce 直接进 layernorm,
数值从这一步就开始偏了。

### Bug 2 —— `should_use_reduce_scatter` 对 round-robin-split 和 in-seq-split 一视同仁

`layers/communicator.py` 里:
```python
if dsa_use_prefill_cp(forward_batch) or mla_use_prefill_cp(forward_batch):
    return True
```
只要开了 CP 就无条件告诉 MoE 层"可以跳过你自己的完整 TP all-reduce"。这个跳过逻辑是给
`in-seq-split + DeepEP` 场景设计的(DeepEP 的 all-to-all 自己处理了通信)。round-robin-split
用的是普通 TP-sharded MoE(`ep_size=1`,没有 DeepEP),必须做这次 all-reduce,不能跳,否则 MoE
输出就是 TP 维度上的部分和,不是完整值。

### Bug 3 —— `dsa_cp_reduce_scatter_hidden_states` 多做了一次不该做的真实通信

`layers/communicator_dsa_cp.py`:
```python
hidden_states = hidden_states.tensor_split(cp_size)[cp_rank]   # 本地切片,这一步对
attn_cp_reduce_scatter_tensor(hidden_states, input_hidden_states)  # 多余的一次真实 reduce-scatter
```
`attn_cp_reduce_scatter_tensor` 是 `get_attention_cp_group().reduce_scatter_tensor(...)`,一次
真正会把各 rank 数据加总再切分的集合通信。round-robin-split 场景下,只要 Bug 2 修复、MoE 自己
做了完整 all-reduce,这里的数据在每个 CP rank 上就应该已经完整且一致——本地切一刀就够了,不该
再做一次通信,否则相当于把同一份值在 cp_size 个 rank 间又加了一遍(数值被放大)。

### 为什么表现成"乱码"

CP 在每一层都会触发这三步,三个误差沿着残差流(residual stream)逐层累积、逐层放大。"你好"这类
模型有很强先验倾向的短回复,内部数值算歪了也有一定概率蒙对;"3+3 等于几"这种需要忠实计算的
问题,数值一旦跑偏就直接崩,输出替换字符——这是"内部数值系统性放大/偏移"类 bug 的典型表现。

## 改了什么(对照 v0.5.15 源码)

- `patches/communicator.py`:`should_use_reduce_scatter()` 改成只在 in-seq-split(或
  MLA CP)时才跳过 MoE 的 all-reduce;round-robin-split 走正常的完整 all-reduce。
- `patches/communicator_dsa_cp.py`:
  - `_gather_hidden_states_and_residual()` 补上缺失的 `attention_tensor_model_parallel_all_reduce`。
  - `dsa_cp_reduce_scatter_hidden_states()` 在 round-robin-split 模式下不再额外调用
    `attn_cp_reduce_scatter_tensor`,只保留本地 `tensor_split`。

两个文件都是在本地 clone 的 `sglang-v0.5.15-src`(即 `../sglang-v0.5.15-src`)基础上直接改的,
`patches/` 下是改完之后的**完整文件**(不是 diff),Dockerfile 直接把它们整个覆盖到镜像里对应
路径。

## 怎么 build

```bash
cd /Users/wall/Desktop/jiliu/maas/dsv4pro-cp-fix
docker build -t harbor.local.clusters/bp/lmsysorg/sglang:v0.5.15-deepseekV4Pro-0723-cpfix .
```

Dockerfile 假设基础镜像里 sglang 是从 `/sgl-workspace/sglang`(`pip install -e`)装的可编辑
安装——这跟我们这几轮排查里所有报错栈的路径(`/sgl-workspace/sglang/python/sglang/srt/...`)
一致。**如果这个假设不对**(比如实际是装到了 site-packages 而不是可编辑安装),需要先
`docker run --rm -it <base image> python3 -c "import sglang.srt.layers.communicator as m; print(m.__file__)"`
确认一下真实路径,再改 Dockerfile 里两行 COPY 的目标路径。

build 完之后只是覆盖了两个纯 Python 文件,不涉及任何编译产物,不需要额外的 `pip install`/
重新编译步骤。

## 怎么验证修复生效

按 #20360 原 PR 和 #28463 里都用到的验证方法:**同一个长 prompt,CP 开/关各跑一次(greedy,
`temperature=0`),比较输出 token id 是否完全一致(或 logprob 在 fp8 误差范围内一致)**。比如:

```bash
# 不开 CP
curl http://<prefill>:<port>/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "DeepSeek-V4-Pro",
  "messages": [{"role": "user", "content": "<你的长 prompt>"}],
  "max_tokens": 64, "temperature": 0
}'

# 开 CP(--enable-dsa-prefill-context-parallel --dsa-prefill-cp-mode round-robin-split)
curl http://<prefill-cp>:<port>/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "DeepSeek-V4-Pro",
  "messages": [{"role": "user", "content": "<同一个 prompt>"}],
  "max_tokens": 64, "temperature": 0
}'
```

两次输出应该完全一致(或非常接近)。此前复现过的乱码 case(比如"3+3 等于几,直接告诉我答案"
`max_tokens=1`)建议重点回归一下。

## 局限性说明

这个修复是基于**静态代码分析**(对照 #20360 的历史诊断 + 通读 v0.5.15 的
`communicator.py`/`communicator_dsa_cp.py`/`dsa/utils.py`)得出的,**没有在真实多机环境上跑过
CP-on vs CP-off 的回归对比**。建议按上面"怎么验证"那一节实测之后再上生产。另外我们在排查过程中
还确认过 `arg_groups/deepseek_v4_hook.py` 里有一条已知限制——"Context parallel only supports
single machine (tp_size <= 8). Cross-machine CP has precision issues."——这三个 bug 修复的是
"round-robin-split 本身的正确性问题",不代表跨机器(tp8pp4 这种 PP 跨多机)场景下 CP 就完全没有
风险,请仍按官方这条限制,只在单机(不叠加 PP)范围内使用 CP。
