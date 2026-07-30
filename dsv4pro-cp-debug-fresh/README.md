# DeepSeek-V4-Pro CP 精度问题 —— 全新纯诊断镜像(第二十四节)

## 为什么要新建这个目录

之前 `dsv4pro-cp-debug/` 是在排查过程中一轮一轮叠加出来的,里面混了:padding position
撞车的真实修复(第十四节)、七七八八一大串历史打点(A-N)、一个测试用的
`SGLANG_TEST_CP_KEEP_SPARSE_PREFILL` 开关。用户担心这些历史改动本身(尤其是当时基于
"可能是 PP 导致"这个后来被推翻的假设做的部分)会不会反而混淆或加剧了问题,要求
**完全基于官方 v0.5.15 tag 源码重新开始,只加新的打点,不带任何其它代码改动**
(包括不带 padding 修复)。

这个目录就是这次"从零开始"的产物:
- 基础镜像:`harbor.local.clusters/bp/lmsysorg/sglang:v0.5.15-deepseekV4Pro-0723`
  (跟之前一样,没有换)。
- `patches/layers/attention/deepseek_v4_backend.py`:直接取自官方 v0.5.15 tag 源码
  (不是从 `dsv4pro-cp-debug/` 里那份改过的文件再改),只加了两个新的打点。
- `patches/utils/cp_pp_debug2.py`:全新写的打印工具,故意用新文件名、新 tag 前缀
  `[CP_DEBUG2]`(跟旧的 `[CP_DEBUG]` 区分开,方便确认这次日志确实是新代码产生的,
  不会跟旧插桩的输出混在一起)。
- **没有 padding 修复,没有其它任何行为改动**——不开 `SGLANG_CP_DEBUG_DUMP` 的话,
  这个镜像跟官方原版镜像的行为**完全一致**。

## 打了什么点

第二十二/二十三节一路查下来,当前最直接的疑点是 `deepseek_v4_backend.py` 注意力
`forward` 里 `flash_mla_with_kvcache` 分支调用前的 `match_num_queries` 函数——如果
C4/C128 稀疏 indexer 选出来的 `extra_indices`(`c4_sparse_page_indices`/
`c128_page_indices`)行数跟这个 cp_rank 的本地 query 数(`q.shape[0]`)对不上,会被
截断成"全局数组的前 N 行"而不是这个 cp_rank 该有的 round-robin 那几行。

| 标签 | 位置 | 含义 |
|---|---|---|
| `P2_pre_match_shapes` | `match_num_queries` 定义之前 | `q`/`extra_indices`/`extra_topk_lengths`/`swa_page_indices`/`swa_topk_lengths` 各自的行数(`shape[0]`),截断前的原始形状,用来看这几个东西本来是不是就对不上 |
| `P3_post_match_extra_indices` | `match_num_queries` 处理完之后 | 处理后 `extra_indices` 的完整取值(小张量会打 `values=[...]`) |
| `P3_post_match_extra_topk_lengths` | 同上 | 处理后 `extra_topk_lengths` 的完整取值 |

## 怎么用

```bash
docker build -t harbor.local.clusters/bp/lmsysorg/sglang:v0.5.15-deepseekV4Pro-0723-cpdebug2 .
```
起服务时加 `SGLANG_CP_DEBUG_DUMP=1`,其它参数不变(`--enable-dsa-prefill-context-parallel
--dsa-prefill-cp-mode round-robin-split` 等)。跑同一条"讲个冷笑话"(8 token,不需要
padding,最干净的复现场景),`grep "\[CP_DEBUG2\]"` 就能捞出这次的打点(注意是
`CP_DEBUG2`,不是旧的 `CP_DEBUG`)。

重点看:
- `P2_pre_match_shapes` 里 `extra_indices_shape0` 是不是跟 `q_shape0` 不一样(如果
  一样,说明 `match_num_queries` 对这次请求是个 no-op,截断假设不成立)。
- 如果不一样,`P3_post_match_extra_indices` 里截断/填充之后的实际取值,是不是这个
  cp_rank 真正该有的那一份 topk 页索引,还是明显是别的 cp_rank/别的全局位置的数据。

## 第二轮追加:P2/P3 已经用真实数据排除(`match_num_queries` 从未触发截断,`extra_indices`
的 -1/有效值分布完全符合"C4 压缩需要攒够 4 个 token"这个正常设计),转去查压缩 KV cache
写入 cache 之后的实际内容、以及 SWA(滑动窗口)部分自己的数值

新加的点,还是同一个 attention `forward` 函数里,`match_num_queries` 之后、真正调用
`flash_mla_with_kvcache`/`flash_mla_with_kvcache_sm120` 之前和之后:

| 标签 | 位置 | 含义 |
|---|---|---|
| `Q4_swa_page_indices` | kernel 调用之前 | SWA 滑动窗口的页索引实际取值(小张量会打完整 `values=[...]`) |
| `Q4_swa_topk_lengths` | 同上 | SWA 每个 query 实际有效的窗口长度 |
| `Q5_swa_k_cache` | 同上 | 真正从 KV pool 里读出来的 SWA K cache 内容(整块 buffer 的统计量,不是某一行) |
| `Q5_extra_k_cache` | 同上(`extra_k_cache is not None` 时) | 真正读出来的 C4/C128 压缩 KV cache 内容 |
| `R6_attn_output` / `R6_attn_output_sm120` | kernel 调用之后 | 这次 attention 计算的实际输出 `o`(mean/absmax/norm/has_nan/has_inf)——这是判断"这一步计算本身是不是已经算出异常结果"最直接的信号 |

重点看:
- `R6_attn_output` 有没有 `has_nan=True`/数值明显爆炸(对照之前在别的地方看到的量级)。
- `Q5_swa_k_cache`/`Q5_extra_k_cache` 是不是全零/全 NaN/明显不合理的量级(说明写缓存
  这一步本身出了问题)。
- `Q4_swa_page_indices`/`Q4_swa_topk_lengths` 的实际取值是否合理(页索引是不是在合理
  范围内,不是全 -1 或全 0)。
