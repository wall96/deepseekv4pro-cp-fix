# DeepSeek-V4-Pro FP8 prefill 权重加载崩溃 —— 纯诊断镜像(不改变任何行为)

## 背景

`DeepSeek-V4-Pro-deepgemm-fp8-hf` 这份 FP8 checkpoint,在 B300 单机 prefill(不传
`--pp-size`,默认 `pp_size=1`)启动时,`load_weights` 阶段崩溃:

```
ValueError: Downcasting not allowed: target.dtype=torch.float8_e4m3fn,
loaded_weight.dtype=torch.bfloat16
```

报错位置是 `layers/parameter.py::copy_with_check`,经 `layers/linear.py::weight_loader_v2`
调用到达,是某个普通 `Linear`(不是 MoE/专家权重)的问题。同一份 checkpoint 在 H100
`tp8pp4` prefill、B300 `tp8dp8ep8` decode 上都能正常加载。

试过给 prefill 加 `SGLANG_FP8_IGNORED_LAYERS=lm_head`(怀疑是 `lm_head` 没被
checkpoint 的 `config.json` 标进 `ignored_layers`),没有解决,说明 `lm_head` 不是
(唯一的)根因。

## 为什么报错本身是"匿名"的

`copy_with_check`(`parameter.py:73`)只打印两边的 dtype,不带权重名字。
`models/deepseek_v4.py::load_weights` 里唯一给异常加 `e.add_note(name=...)` 的
try/except(约 2804-2806 行)只包住"遍历权重、提交给线程池"这段**同步**代码,而
真正抛错的 `weight_loader` 调用是在 `ThreadPoolExecutor` 里异步跑的,通过
`future.result()`(约 2809 行)重新抛出时,不在那个 try 范围内——所以现有代码
结构性地拿不到权重名字,继续读代码猜没有意义,需要插桩拿真实数据。

## 打了什么点

只改了一个文件:`patches/layers/linear.py`,直接取自官方 v0.5.15 tag 源码(不是
从别的已改过的副本改的),两处改动:

1. `LinearBase.__init__` 里补一行 `self.prefix = prefix`——这行之前只有
   `MergedColumnParallelLinear`/`MergedColumnParallelRepeatedLinear` 自己单独设置,
   其余(`ColumnParallelLinear`/`QKVParallelLinear`/`RowParallelLinear` 等)实例上
   根本没有这个属性,不补的话下面的 `except` 块拿不到。
2. 给 4 个可能触达 `copy_with_check` 的 `weight_loader_v2` 实现
   (`ColumnParallelLinear`/`MergedColumnParallelLinear`/`QKVParallelLinear`/
   `RowParallelLinear`)分别包了一层 `try/except Exception as e: e.add_note(...); raise`,
   note 里带 `prefix`(层名)、`param_type`、`param_dtype`、`loaded_weight_dtype`、
   两边的 shape,不吞异常、不改变任何控制流/数值。

`ReplicatedLinear`(`wq_a`/`wkv`/`wqkv_a`/`compressor.wkv_gate` 这类)走的是
`weight_loader`(v1),不是 `weight_loader_v2`,不会命中 `copy_with_check`,所以
没有改它。

**没有开关、默认永远生效**——这次不需要像 CP 那次一样用环境变量控制开关,因为
`add_note` 本身零开销、不影响正常运行时的任何行为,只在真的抛异常时才会体现出来。
不崩溃的话,这个镜像跟官方原版行为完全一致。

## 怎么用

```bash
docker build -t harbor.local.clusters/bp/lmsysorg/sglang:v0.5.15-deepseekV4Pro-0723-fp8debug .
```

把原来失败的 B300 prefill 启动命令里的 `--image` 换成这个新 tag,其它参数不变
(建议先去掉 `SGLANG_FP8_IGNORED_LAYERS=lm_head`,验证纯净场景),重新跑一次。
崩溃时的完整 traceback 最后会多出一行(Python 3.11+ 的 `add_note` 会在
"ValueError: Downcasting not allowed..." 之后单独打印出来),形如:

```
[FP8_DEBUG] ColumnParallelLinear.weight_loader_v2 failed: prefix='model.layers.3.self_attn.wq_b' param_type=... param_dtype=torch.float8_e4m3fn loaded_weight_dtype=torch.bfloat16 param_shape=(...) loaded_weight_shape=(...)
```

`grep "\[FP8_DEBUG\]"` 就能直接拿到具体是哪一层(`prefix`)、哪种 Linear
(`ColumnParallelLinear`/`MergedColumnParallelLinear`/`QKVParallelLinear`/
`RowParallelLinear`)出的问题,不用再靠排除法猜。

## 拿到权重名之后怎么办

- 如果是 `lm_head` 或 `model.norm`:回到 `arg_groups/deepseek_v4_hook.py`/
  `models/deepseek_v4.py` 里 `pp_group.is_last_rank` 相关逻辑,确认为什么
  `SGLANG_FP8_IGNORED_LAYERS=lm_head` 没生效(可能是别的层同名冲突、或者
  `is_layer_skipped` 的前缀匹配没对上实际传进去的 `prefix`)。
- 如果是 `model.layers.N.self_attn.*`(`wq_b`/`wo_b` 等):大概率是
  attention 相关的量化/分片逻辑跟 `attn_cp_size`/`attn_tp_size` 的组合有关,
  需要回到 `deepseek_v4.py` 里这几个 `ColumnParallelLinear`/`RowParallelLinear`
  的构造参数(`tp_rank=attn_tp_rank, tp_size=attn_tp_size`)去看。
- 如果是 MoE 相关但走了 `weight_loader_v2`(理论上不应该,MoE 走 FusedMoE 专用
  loader,但如果真是这样说明模型构造有问题):需要重新核实是不是走错了 loader。
- 不管是哪一层,下一步应该去检查 checkpoint 实际磁盘上这个 tensor 的 dtype
  (确认它确实是 bf16、没有配套的 `.weight_scale_inv`),再决定是该把这一层加进
  `ignored_layers`(checkpoint 侧修复 / `SGLANG_FP8_IGNORED_LAYERS` 补上正确的
  层名),还是该修 SGLang 这边对这一层的量化判断逻辑。
