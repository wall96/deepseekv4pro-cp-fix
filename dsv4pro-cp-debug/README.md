# DeepSeek-V4-Pro CP(round-robin-split)精度问题 —— 调试插桩

> **根因已确认,修复见 `../dsv4pro-cp-fix-v2/`。** 本目录的插桩帮我们从真实日志里定位到了
> 具体分叉点(`G_post_cp_reindex_positions` 显示 CP round-robin 的 padding 槽 position 撞车到
> 0),过程记录在 `dsv4pro/log-analysis.md` 第十四节。这个镜像现在同时也带上了那份修复
> (见 `Dockerfile` 里最后两行 `COPY`),方便一边跑一边用 `[CP_DEBUG]` 输出验证修复效果;
> 如果只要纯诊断、不要修复,把 Dockerfile 里那两行 COPY 去掉即可。
>
> **用户实测:修复后 `G` 点的 position 确认不再撞车,但业务输出仍然乱码。** 说明 padding
> position 撞车是一个真实 bug,但不是(或不是唯一)导致这次乱码的根因。已经加了第 8 个
> 打点粒度(不是新 tag,是给现有的 `cp_debug_dump` 加了按行输出):当被打印的张量是
> `shape[0] <= 16` 的二维张量时(正好覆盖 CP 切分后每个 cp_rank 只剩 1-2 行的场景),额外
> 打一行 `row_norms=[...]`/`row_absmax=[...]`,把"真实 token 那一行"和"padding token 那一行"
> 的数值分开看——之前 `A-E`/`PP_SEND`/`PP_RECV` 这些点打的是整个 2 行张量合在一起算的
> mean/absmax/norm,没法区分是 padding 行本身数值大(无害,反正会被截断丢弃)还是真实
> token 那一行也被污染了。见 `dsv4pro/log-analysis.md` 第十五节。

## 这是什么

**不是修复,是诊断工具。** 排查到目前这一步,静态读代码已经把好几个具体假设(PP 跨 stage 的
`attn_cp_rank` 对应关系、"CP-V2" 新框架有没有跟 DSA 的 CP 冲突、`forward_metadata` 初始化顺序)
一个个查过、一个个排除,没找到能拍板的确凿 bug。继续凭读代码猜下去性价比已经很低,所以换成
**跑起来、打印中间数值、拿真实数据定位**这条路。

对照 `../dsv4pro/log-analysis.md` 第十一节"CP 的完整逻辑"里理清楚的数据流,一共打了 **7 个点**,
覆盖三个不同层面:layer 内部的 CP 数据流(A-E,`models/deepseek_v4.py`)、跨 PP stage 真正传输
数据的边界(`PP_SEND_*`/`PP_RECV`,`managers/scheduler_pp_mixin.py`)、以及 CP 用到的 reindex
元数据本身(F/G,专门针对"PP 是不是导致精度问题"这个方向加的)。

### A-E:layer 内部的 CP 数据流(`models/deepseek_v4.py`)

| 标签 | 位置 | 含义 |
|---|---|---|
| `E_post_kv_allgather_rerange` | `_forward_prepare` 里 `cp_all_gather_rerange_output` 之后 | attention 用的 K/V,跨 CP rank all-gather + 转置去交织之后 |
| `A_post_attn_pre_moe_gather` | `DecoderLayer.forward`,attention 输出、MoE-gather 之前 | 这张卡负责的那一份 attention 输出(交织的一小撮 token) |
| `B_post_moe_gather_pre_mlp` | MoE 前 `dsa_cp_gather_hidden_states` 之后、`self.mlp(...)` 之前 | gather 成全量之后、真正进 MoE 之前 |
| `C_post_mlp_pre_reduce_scatter` | `self.mlp(...)` 之后、`dsa_cp_reduce_scatter_hidden_states` 之前 | MoE 的原始输出(此时是 TP 维度的部分和,还没规约) |
| `D_post_reduce_scatter` | `dsa_cp_reduce_scatter_hidden_states` 之后 | 这一层最终交给下一层的 hidden_states |

### PP_SEND_*/PP_RECV:跨 PP stage 真正传输数据的边界(`managers/scheduler_pp_mixin.py`)

**这是专门针对"是不是 PP 导致精度损失"这个假设加的**——直接检验同一份 hidden_states,从上一个
PP stage 发出去,到下一个 PP stage 收到,数值有没有变。三个 event loop 变体(`event_loop_pp`/
`event_loop_pp_disagg_prefill`/`event_loop_pp_disagg_decode`)各打了一份发送点(你这次用的是
disagg prefill,对应 `PP_SEND_disagg_prefill`),接收端统一在 `_pp_recv_proxy_tensors()` 一处
(`PP_RECV`)。

| 标签 | 位置 | 含义 |
|---|---|---|
| `PP_SEND_event_loop_pp` / `PP_SEND_disagg_prefill` / `PP_SEND_disagg_decode` | 各自 event loop 里 `_pp_send_dict_to_next_stage(result.pp_hidden_states_proxy_tensors.tensors, ...)` 之前 | 这个 PP stage 算完、正要发给下一个 stage 的 hidden_states(带 `mb_id`,同一份数据在流水线里对应哪个 microbatch) |
| `PP_RECV` | `_pp_recv_proxy_tensors()` 里 `PPProxyTensors(...)` 构造之后 | 下一个 PP stage 刚收到、还没进它自己 forward 的 hidden_states |

**看数据的方法**:把 PP0(N号 stage)的 `PP_SEND_disagg_prefill` 那一行,跟 PP1(N+1号 stage)紧接着
的 `PP_RECV` 那一行对比——**这两行理论上应该是同一份数据,`shape`/`mean`/`absmax`/`norm` 应该完全
相同(除非中间真的被什么东西改动过)**。如果这两行对不上,就实锤是 PP 传输这一跳本身把数据弄错了
(不是 CP 逻辑的问题,是更基础的 PP send/recv 或者跟 CP 混在一起用的某个 all_gather_group 参数
的问题);如果这两行完全一致,但下一个 stage 后续的 `A_post_attn_pre_moe_gather` 等等开始跑偏,
说明问题出在"收到数据之后,这个 stage 自己怎么用 CP 逻辑处理它"这一步,不是传输本身的问题。

### F/G:CP reindex 元数据本身(`models/deepseek_v4.py`,`DeepseekV4ForCausalLM.forward` 顶层)

**这两个点专门针对"multi-microbatch 流水线里,`apply_cp_reindex()` 用的那份 metadata 对象,
会不会被后一个 microbatch 提前踩踏"这个假设**——PP 的 `event_loop_pp_disagg_prefill` 会在多个
`mb_id` 之间轮转,如果 `attn_backend.forward_metadata.core_attn_metadata` 是一个被反复原地
mutate 的共享对象,而不是每个 microbatch 各自独立的一份,不同 microbatch 的 reindex 结果就有
可能互相污染——这种 bug 只有 PP 的流水线重叠机制才会触发,单机 CP(没有多 microbatch 轮转)
不会碰到。

| 标签 | 位置 | 含义 |
|---|---|---|
| `F_pre_cp_reindex_positions` | `core_meta.apply_cp_reindex()` 之前 | 这个 stage 本地看到的、还没按 cp_rank 切过的完整 positions |
| `G_post_cp_reindex_positions` | `core_meta.apply_cp_reindex()` 之后 | 按 cp_rank 交织切完之后,这张卡实际负责的那部分 positions |

这两条额外带了一个 `meta_id=<python id()>` 字段——**同一个 mb_id 连续两次调用之间,`meta_id`
应该保持一致(说明是同一个 metadata 对象在正常复用);但如果不同 mb_id 交替调用时看到 F 打印
的"pre-reindex 完整 positions"里混进了不该属于当前 mb_id 的 token,或者 `meta_id` 在两次调用
间没变但 G 打印出来的 positions 集合跟这次 mb_id 对不上,就实锤是共享 metadata 被踩踏了。**

每条打印格式(用 `logger.warning`,`grep "\[CP_DEBUG\]"` 就能捞出来):
```
[CP_DEBUG] tag=A_post_attn_pre_moe_gather layer=12 pp_rank=2 cp_rank=5 tp_rank=5 shape=(37, 7168) dtype=torch.bfloat16 mean=0.00123 absmax=4.56 norm=812.3 has_nan=False has_inf=False
[CP_DEBUG] tag=PP_SEND_disagg_prefill layer=None pp_rank=0 cp_rank=3 tp_rank=3 mb_id=1 name=hidden_states shape=(37, 7168) ...
[CP_DEBUG] tag=G_post_cp_reindex_positions layer=-1 pp_rank=1 cp_rank=3 meta_id=140234... shape=(37,) dtype=torch.int32 min=3 max=291 sum=5402
```

### H/I(已作废,`compressor.py` 是死代码):~~C4/C128 压缩 KV 写入路径的 `out_loc`~~

**这两个标签已经不会再出现了,留着这段是记录踩过的坑**:最初以为
`layers/attention/dsv4/compressor.py` 的 `forward_core_compressor`/`forward_indexer_compressor`
(`out_loc[: new_compressed_kv.shape[0]]` 截断)是压缩 KV 写入的实际路径,加了 H/I 两个点,
但后来发现 **GPU 后端实际 import 的 `CompressorBackendMixin` 来自 `compressor_v2.py`,
不是 `compressor.py`**——`compressor.py` 那两个函数对这次的部署完全不会被调用,H/I 这两个
标签在真实日志里注定是零命中。已经把这次的 `COPY` 从 Dockerfile 里换成
`compressor_v2.py`,详见下面 J/K 两个新点。见 `dsv4pro/log-analysis.md` 第十八节 18.6。

### J/K:真正生效的 C4/C128 压缩 KV 写入路径(`layers/attention/dsv4/compressor_v2.py`)

**这两个点打在真正会执行到的代码上**,核实"CP 开启时,8 个 cp_rank 是不是真的在全局尺度上
冗余、各自独立地算出完全一样的压缩结果"这个假设——`compressor_v2.py` 全文没有任何
`cp_rank`/`cp_size` 相关代码,所有 CP 处理都在更上游(`compute_kv_score` 里对 `kv_score`
的 all-gather),如果这个假设成立,8 个 cp_rank 打出来的 `kv_score_input`/`kv_compressed`/
`out_loc` 应该逐位完全相同;如果有任何一个 cp_rank 跟别的不一样,就实锤这里是问题所在。

| 标签 | 位置 | 含义 |
|---|---|---|
| `K_v2_kv_score_input` | `forward_unified` 里 `compute_kv_score` 之后 | 这个 cp_rank 看到的 kv_score(CP 开启时理论上已经 all-gather 回全局) |
| `K_v2_out_loc` | `forward_unified` 里 `_get_out_loc` 之后(非 HIP 分支) | 这个 cp_rank 拿到的 `out_loc`(全局长度,理论上 8 个 cp_rank 应该完全一样) |
| `J_v2_kv_compressed` | `_forward_compress_all_in_one` 里 `compress_forward` 之后 | 压缩计算的实际输出(带 `ratio`/`is_indexer`/`out_loc_shape`) |

int 类型的小张量(`out_loc` 这种)现在除了 `min`/`max`/`sum` 之外,还会额外打一行完整的
`values=[...]`(数量 <=64 时),方便直接对比不同 cp_rank 的槽位列表是不是完全一样。

### L:MoE 前实际喂给 `self.mlp` 的 `input_ids`(`models/deepseek_v4.py`,第十九节)

第十九节查到 DeepSeek V4 的 Hash MoE(早期 `layer_id < n_hash_layers` 的层)路由完全靠
`tid2eid[input_ids]` 这个查表,不看 `hidden_states`/router logits——如果 CP 场景下喂给
`self.mlp` 的 `input_ids` 跟 `hidden_states` 的行对不齐(`moe_a2a_backend="none"` 时
`cp_round_robin_input_ids` 返回的是全局重排数组,不是这个 cp_rank 的本地切片),会导致
hash 路由层被路由到错误的专家——这能完美解释确定性、流畅但答非所问的乱码。但代码复算
下来,`dsa_cp_gather_hidden_states` 跟 `cp_round_robin_input_ids` 的 `is_none()` 分支
理论上是同一种"按 rank 分块"的重排顺序,`_use_tp_attn_a2a_scatter` 在 CP 开启时又恒为
`False`(`_use_tp_attn_a2a_scatter = (not _use_cp and ...)`)不会再插一次手——单靠代数推
不出确切结论,需要看真实数值。

| 标签 | 位置 | 含义 |
|---|---|---|
| `L_pre_mlp_input_ids` | `DecoderLayer.forward` 里,`self.mlp(...)` 调用之前,跟 `B_post_moe_gather_pre_mlp` 同一个点 | 这一层实际喂给 MoE 的 `input_ids`(带 `is_hash`/`use_tp_attn_a2a_scatter`/`hidden_states_shape`),用来跟 `hidden_states` 的行对齐情况做交叉验证 |

int 类型的小张量同样会打完整的 `values=[...]`,方便直接看这层是不是 hash 层
(`is_hash=True`)、`input_ids` 的具体取值序列跟预期的 token 序列是否一致。

### M/N:`flash_mla_with_kvcache` 分支里 `match_num_queries` 的 shape 对齐情况(第二十一节)

单机 B300(不跨机、没有 PP)也复现了乱码之后,排查重心从"PP 相关代码"转回纯粹的
DeepSeek-V4 + CP 注意力代码本身。`deepseek_v4_backend.py` 的注意力 `forward` 里,
`flash_mla_with_kvcache` 分支(CP 场景下唯一会走到的路径,见 `SGLANG_TEST_CP_KEEP_SPARSE_PREFILL`
那节)在真正调用 kernel 之前,有一个 `match_num_queries` 帮手函数:

```python
def match_num_queries(x, value):
    if x is None or x.shape[0] == q.shape[0]:
        return x
    if x.shape[0] > q.shape[0]:
        return x[: q.shape[0]]          # <-- 截断成"前 q.shape[0] 行"
    return _pad_tensor_to_size(x, q.shape[0], value=value)
```

如果 `extra_indices`(C4 稀疏 indexer 选出来的 topk 页索引,`c4_sparse_page_indices`)
的行数跟这个 cp_rank 的本地 query 数(`q.shape[0]`)对不上,就会被截断成"全局数组的前 N 行"
而不是这个 cp_rank 应该拿到的 round-robin 那几行——这跟第十四节 padding 撞车、第十八节
（已排除的）compressor 截断,是同一个"用错误的下标截断代替正确的 round-robin 切片"模式。
读代码追了一遍 `c4_sparse_page_indices`/`c4_sparse_topk_lengths` 的构造时机(在
`init_flashmla_related` 里,而这个函数在 CP 场景下会在 `apply_cp_reindex()` 之后被
重新调用一次,理论上应该用重新reindex过的本地长度重建),看起来设计上是自洽的——但
连续几次"看代码觉得是 bug、细看又不是"的教训(compressor.py、Hash MoE `input_ids`)
让人不敢再只凭读代码下结论,所以直接加了打点用真实数据验证。

| 标签 | 位置 | 含义 |
|---|---|---|
| `M_pre_match_num_queries_shapes` | `match_num_queries` 调用之前 | `q`/`swa_page_indices`/`swa_topk_lengths`/`extra_indices`(C4 或 C128)/`extra_topk_lengths` 各自的行数(`shape[0]`),用来看这几个东西在 truncate/pad 之前是不是本来就对得上 |
| `N_c4_extra_indices_post_match` | `match_num_queries` 处理完之后,仅 `compress_ratio==4` 时 | 处理后的 `extra_indices` 完整取值(小张量会打 `values=[...]`),看这个 cp_rank 拿到的 topk 页索引是不是它自己 round-robin 该有的那一份,还是被截断成了别的 cp_rank/别的全局位置的数据 |

### 实验开关:`SGLANG_TEST_CP_KEEP_SPARSE_PREFILL`(第十九节)

GitHub 调研(`sgl-project/sglang#27384` 等)+ 读代码发现:`arg_groups/deepseek_v4_hook.py`
的 `validate_deepseek_v4_cp()` 只要 CP 开启就会**无条件**把 `SGLANG_OPT_FLASHMLA_SPARSE_PREFILL`
(默认 `True`)强制 `.set(False)`——用户在启动命令里设这个环境变量也没用,会被这段校验逻辑
覆盖。`deepseek_v4_backend.py` 里 prefill attention 按 query 数分派:
`q.shape[0] > _LARGE_INDEXER_QUERY_THRESHOLD(11673) or envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.get()`
决定走 `_forward_prefill_sparse`(`flash_mla_sparse_fwd`,大 query 数场景验证最充分)还是
`flash_mla_with_kvcache`(历史更久的另一套实现)。CP 会把每个 cp_rank 的本地 query 数除以
`cp_size`,几乎不可能达到 11673,再叠加上面这个强制禁用——**CP 一开,prefill 100% 会被
切到 `flash_mla_with_kvcache`,永远走不到 `_forward_prefill_sparse`**。

`SGLANG_TEST_CP_KEEP_SPARSE_PREFILL=1` 这个新开关(`environ.py`)让 CP 场景也保留
`SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=True`,强制走 `_forward_prefill_sparse`,用来验证
乱码是不是出在 `flash_mla_with_kvcache` 这条被 CP 强制切过去的路径上。**这是纯诊断/测试
开关,默认关闭(`False`),不开这个环境变量的话行为跟原来完全一样。**

## 怎么用

1. Build:
   ```bash
   docker build -t harbor.local.clusters/bp/lmsysorg/sglang:v0.5.15-deepseekV4Pro-0723-cpdebug .
   ```
2. 用这个镜像起 prefill,启动命令在原来的基础上加一个环境变量:
   ```
   SGLANG_CP_DEBUG_DUMP=1
   ```
   （不加这个变量,插桩完全不生效,可以放心当正常镜像用）
3. **同一个长 prompt**(足够长才会真正触发 CP,`can_dsa_cp_split` 要求 `seq_len >= cp_size`),
   分别用 **CP 开**和**参照组**各跑一次,把日志分别存下来,比如 `cp_on.log` / `cp_off.log`(或者
   如果没法关 CP 做对照,就跟第一次触发乱码那次的日志比,重点看 PP0 vs PP1/2/3 之间数值有没有
   在某一层突然跳变、或者出现 `has_nan=True`)。
4. 用下面这条命令把两份日志按 `layer/pp_rank/cp_rank/tag` 对齐,找第一次出现明显差异
   (数量级不一样、has_nan 变 True、absmax 突然变得很大)的那一行:
   ```bash
   grep "\[CP_DEBUG\]" cp_on.log  | sort > cp_on.sorted.txt
   grep "\[CP_DEBUG\]" cp_off.log | sort > cp_off.sorted.txt
   diff cp_on.sorted.txt cp_off.sorted.txt | less
   ```
   （如果两次跑的 batch/调度顺序不完全确定性,可能需要按 `layer=` 分组、只挑某一个固定
   请求/token 位置来比,而不是整份日志硬 diff。)

## 看数据时重点关注什么

- **`PP_SEND_disagg_prefill`(PP0)跟紧接着的 `PP_RECV`(PP1)对不上** → 实锤是 PP 传输这一跳本身
  出了问题,不是 CP 逻辑的问题,直接去查 `_pp_send_dict_to_next_stage`/`send_tensor_dict` 那条
  路径(`all_gather_group`/`msg_type` 这些参数在 attn_tp_group 存在时会不会多做一次不该做的
  all-gather)。**这是最优先该看的一条**,因为它是"PP 导致精度损失"这个假设里最直接、最不需要
  额外推理就能下结论的证据。
- **`PP_SEND`/`PP_RECV` 完全对得上,但 `F_pre_cp_reindex_positions`/`G_post_cp_reindex_positions`
  在不同 `mb_id` 之间看起来串了**(比如同一个 `meta_id` 在两次不同 `mb_id` 的调用里,positions
  的取值范围明显对不上当前 mb_id 该有的 token 数)→ 实锤是流水线里多个 microbatch 共享/踩踏了
  同一份 CP metadata 对象,根因在 `attn_backend.forward_metadata` 有没有做到每个 microbatch
  独立(或者有没有正确的同步机制防止下一个 mb_id 的 reindex 抢在当前 mb_id 的 GPU kernel 读取
  之前发生)。
- **`E_post_kv_allgather_rerange` 和 `A_post_attn_pre_moe_gather` 先出问题**(而 PP_SEND/RECV、
  F/G 都正常)→ 说明是 attention/K-V-gather 那条链路(第十一节 2.3/2.4 节)本身出的问题,重点去查
  `cp_all_gather_rerange_output` 的转置去交织逻辑。
- **`A`/`E` 都正常,`B`(MoE-gather 之后)开始出问题** → 说明是 `dsa_cp_gather_hidden_states`
  这一步本身(纯 all_gather,不做转置还原)有问题——按第十一节 2.5 节的说法,这一步理论上不需要
  转置也能自洽,但如果数据实测在这里就已经不对,说明这个"不需要转置"的假设本身是错的,需要
  回头重新验证这一步。
- **`B`/`C` 都正常,`D`(reduce-scatter 之后)开始出问题** → 说明 `dsa_cp_reduce_scatter_hidden_states`
  里 `attn_cp_reduce_scatter_tensor` 这次真正的通信本身有问题(要么规约的次数不对,要么
  `tensor_split` 切片跟 all_gather 摆放顺序没对齐)。
- **同一层里,PP0 正常但 PP1/PP2/PP3 开始出问题**(或者反过来)→ 支持"PP 导致精度损失"这个假设,
  但要结合上面 `PP_SEND`/`PP_RECV`/`F`/`G` 的具体证据才能说清楚是 PP 的哪个具体环节导致的,不能
  只停留在"确实是 PP 的锅"这个结论上。

## 后续

等有了实测数据、确定了具体是哪个点先出问题,再回来商量具体怎么修——上次 `dsv4pro-cp-fix/`
那次教训就是没有实测验证就直接改代码,结果改错了地方。这次先用这份插桩拿到真实证据,再动手修。
