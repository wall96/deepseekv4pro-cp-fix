# DeepSeek-V4-Pro CP(round-robin-split)精度问题 —— 修复(v2,已确认根因后的正式修复)

## 这是什么

`dsv4pro-cp-fix/`(第一版)已废弃——那次是基于对照 NSA 旧 issue 的猜测改的
`LayerCommunicator`/`DSACPLayerCommunicator`,后来确认 `DeepseekV4ForCausalLM` 根本不走那条
代码路径,完全不生效。

这一版(`dsv4pro-cp-fix-v2/`)是在 `dsv4pro-cp-debug/` 插桩镜像跑出**真实 debug 日志**、
定位到具体分叉点之后,再去读代码确认的**已验证根因**的修复,过程记录见
`dsv4pro/log-analysis.md` 第十四节。

## 根因

DSA prefill context-parallel(`--dsa-prefill-cp-mode round-robin-split`)会把一个 batch 的
token 数向上 pad 到 `cp_size` 的整数倍(比如 12 个真实 token pad 到 16,`cp_size=8`)。

`expand_prefill_casually` / `_expand_prefill_casually_vectorized`
(`layers/attention/deepseek_v4_backend.py`)和 `expand_prefill_casually`
(`layers/attention/deepseek_v4_backend_hip_radix.py`)给 padding 槽的 `seq_lens_casual` 填的是
常数 `1`:
```python
seq_lens_casual = torch.nn.functional.pad(seq_lens_casual, (0, pad_size), value=1)
```
而 `positions_casual = seq_lens_casual - 1`(`make_core_attn_metadata`)。所以**每一个 padding
槽算出来的 position 都是 `1 - 1 = 0`**——跟真实的 position-0 token 完全撞车。

CP round-robin 按 `apply_cp_reindex` 的 `idx = slice(cp_rank, None, cp_size)` 切分之后,
`cp_rank=4..7` 本该各自拿到一个真实 token + 一个 padding 槽(如 `{4,12}`),但因为这个 bug
实际拿到的是 `{4,0},{5,0},{6,0},{7,0}`——这些 rank 的"padding 槽"变成了对真实 position-0
token 的又一次合法计算,而不是一个可以安全丢弃的哑元。真实抓包验证:这几个 rank 的
hidden_states(mean/absmax/norm)几乎跟 cp_rank 0 自己的结果一模一样,印证了这个撞车。

已确认这个撞车正是"跨机 CP 输出乱码"最直接、证据最充分的根因:padding 行本来就会在
`models/deepseek_v4.py` 的 `real_num_tokens = forward_batch.num_token_non_padded_cpu`
截断步骤里按**原始 slot 数量**被丢弃,只要它的 position 不跟任何真实 token 撞车,它算出来的
(该丢弃的)结果就不可能被误当成真实数据保留。

## 修复

把 padding 槽 `seq_lens_casual` 的填充值,从常数 `1` 改成**从最后一个真实 token 的值开始
连续递增**,让 padding 槽的 position 落在 `[real_num_tokens, padded_num_tokens)` 这个真实
token 永远不会用到的区间:
```python
pad_seq_lens = seq_lens_casual[-1:] + torch.arange(1, pad_size + 1, **self.cuda_int32_kwargs)
seq_lens_casual = torch.cat([seq_lens_casual, pad_seq_lens])
```
三处都改(两个 attention backend 文件里各自的实现):
- `layers/attention/deepseek_v4_backend.py::expand_prefill_casually`
- `layers/attention/deepseek_v4_backend.py::_expand_prefill_casually_vectorized`
- `layers/attention/deepseek_v4_backend_hip_radix.py::expand_prefill_casually`

`req_pool_indices_repeated` 的 padding 方式不变(本来就合理,重复最后一个真实 request 的
pool index)。`swa_topk_lengths = clamp(seq_lens_casual, max=SWA_WINDOW)` 等派生逻辑也不用改,
已确认能正确处理变大的 position 值。已确认 `req_to_token` 每个 request 的行容量远大于
padding 会用到的偏移量,不会越界/崩溃。

## 怎么用

```bash
docker build -t harbor.local.clusters/bp/lmsysorg/sglang:v0.5.15-deepseekV4Pro-0723-cpfix-v2 .
```
起 prefill 时正常带上原来的 CP 相关参数(`--enable-dsa-prefill-context-parallel
--dsa-prefill-cp-mode round-robin-split` 等),**不需要**额外的环境变量——这是修复,不是
可选开关。

如果想一边验证一边看 `[CP_DEBUG]` 输出,用 `../dsv4pro-cp-debug/`(已经把这份修复同步
进去了,`SGLANG_CP_DEBUG_DUMP=1` 打开插桩)。验证方法见 `dsv4pro/log-analysis.md` 第十四节
"验证计划":重点看同一条 curl 请求下,`G_post_cp_reindex_positions` 里 cp_rank 4-7 是否变成
`{4,12},{5,13},{6,14},{7,15}`(不再是 `{4,0},{5,0},{6,0},{7,0}`),以及业务输出是否不再乱码。
