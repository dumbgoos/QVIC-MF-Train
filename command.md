# HTM Stage B GPU 短跑验证

使用训练脚本，在修复分支上完成 10 个优化器更新步骤。从 Stage A
`checkpoint-5000` 初始化，输出到新的目录。以下命令沿用 launcher 默认训练数据路径。

```bash
conda activate video
cd /mnt/data_nvme1/wenbin/luoling/cvpr/code/ours/QVIC-MF-Train

git fetch origin
git checkout fix/htm-gradient-diagnostics
git pull --ff-only

# 确认梯度修复存在
grep -n "retain_memory_grad=" qvic/model/qvic_meta_mixin.py

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1

OUT_PARENT=/mnt/sdb1/wenbin/luoling/QVIC-MF-checkpoint/htm_stage_b_2xh100
SMOKE_OUT="$OUT_PARENT/gradient_smoke_$(date +%Y%m%d_%H%M%S)"

mkdir -p logs/htm_checks
set -o pipefail

NUM_GPUS=2 \
OUTPUT_DIR="$SMOKE_OUT" \
HTM_INIT_CHECKPOINT=/mnt/sdb1/wenbin/luoling/QVIC-MF-checkpoint/htm_stage_a_2xh100/checkpoint-5000 \
MAX_RECORDS=64 \
GRAD_ACCUM=1 \
LOGGING_STEPS=1 \
SAVE_STEPS=5 \
REPORT_TO=none \
EXTRA_ARGS="--max_steps 10" \
bash scripts/train/train_htm_stage_b_2xh100.sh \
  2>&1 | tee logs/htm_checks/stage_b_smoke.log

TRAIN_EXIT=${PIPESTATUS[0]}
echo "training_exit=$TRAIN_EXIT"
echo "output_dir=$SMOKE_OUT"
ls -lh "$SMOKE_OUT"
ls -lh "$SMOKE_OUT/checkpoint-10"
```

## 路径配置

这里需要训练数据和 Stage A checkpoint，不使用 LVBench 的 `SLICE`。
确认上面的 Stage A checkpoint 存在。若默认训练路径不对，在训练命令前设置：

```bash
export DATASET_ROOT="实际训练媒体目录"
export ANNOTATION_PATH="实际训练标注JSON路径"
```

## 预期输出

- 完成 10 个优化器更新步骤，每步打印 loss。
- 输出目录生成 `checkpoint-5` 和 `checkpoint-10`。
- 正常结束时 `training_exit=0`。

这次保持 `L=256` 等默认配置，先验证 GPU 训练、显存和 checkpoint 保存。
短跑不要求 loss 稳定下降，也不能用于判断 LVBench 准确率是否恢复。

## 需要回传的内容

- `logs/htm_checks/stage_b_smoke.log` 全文。
- 最后的 `training_exit`、`output_dir` 和两个 `ls` 输出。
- 如果失败，完整 traceback。

## 比较 checkpoint-5 和 checkpoint-10 的参数更新

无需 GPU，仅加载 adapter 和 non-LoRA 权重到 CPU。使用此次短跑的实际目录：

```bash
RUN_DIR=/mnt/sdb1/wenbin/luoling/QVIC-MF-checkpoint/htm_stage_b_2xh100/gradient_smoke_20261013_134226
mkdir -p logs/htm_checks
set -o pipefail
python scripts/tools/check_htm_checkpoint_updates.py \
  --before "$RUN_DIR/checkpoint-5" \
  --after "$RUN_DIR/checkpoint-10" \
  --out logs/htm_checks/checkpoint_updates.json \
  2>&1 | tee logs/htm_checks/checkpoint_updates.log
COMPARE_EXIT=${PIPESTATUS[0]}
echo "compare_exit=$COMPARE_EXIT"
```

预期三个组 `compressor_lora`、`context_seed`、`predictor` 都显示 `UPDATED`，
随后输出 `PASS`，退出码为 0。`UNCHANGED` 表示该组没有超过阈值的存储参数变化；
`NONFINITE` 表示含 NaN/Inf；缺少必需组也会失败。结构不匹配或加载失败退出码为 2，
未更新或非有限值退出码为 1。默认阈值为 0。

请回传 `checkpoint_updates.log`、`checkpoint_updates.json` 和 `compare_exit`。

======

## checkpoint-10 加载与真实 LVBench 推理验证

在训练机器原 `video` 环境运行，使用一张空闲 GPU。脚本优先选择两个不同视频，
每个视频均匀采样 64 帧，开启与此前推理一致的 relevance feedback。
这是加载和推理路径验证，不是全量准确率评估。

```bash
conda activate video
cd /mnt/data_nvme1/wenbin/luoling/cvpr/code/ours/QVIC-MF-Train
git fetch origin
git checkout fix/htm-gradient-diagnostics
git pull --ff-only

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

RUN_DIR=/mnt/sdb1/wenbin/luoling/QVIC-MF-checkpoint/htm_stage_b_2xh100/gradient_smoke_20261013_134226
SLICE=/mnt/sdb1/wenbin/luoling/LVDU/eval/LVBench/data/lvbench_qa_smoke16.jsonl
VIDEO_DIR=/mnt/sdb1/wenbin/luoling/LVDU/eval/LVBench/data/video/all_videos
INFER_OUT="$RUN_DIR/inference_smoke_$(date +%Y%m%d_%H%M%S)"

mkdir -p logs/htm_checks
set -o pipefail
python scripts/tools/check_htm_checkpoint_inference.py \
  --checkpoint "$RUN_DIR/checkpoint-10" \
  --questions "$SLICE" \
  --video-dir "$VIDEO_DIR" \
  --outdir "$INFER_OUT" \
  --num-questions 2 \
  --frames 64 \
  --feedback on \
  2>&1 | tee logs/htm_checks/checkpoint_inference.log
INFER_EXIT=${PIPESTATUS[0]}
echo "inference_exit=$INFER_EXIT"
echo "inference_out=$INFER_OUT"
cat "$INFER_OUT/report.json"
```

### 预期输出与回传内容

- 打印选中的题目 key，加载 6 个 HTM predictor tensors。
- 出现 `[HTM] eval runtime: stage_a=False` 和每题 `[HTM τ]` 统计。
- 两题均输出非空、可解析的选项答案。
- `report.json` 中所有 `checks` 为 true，并输出
  `PASS: checkpoint reload and HTM inference`，退出码为 0。

请回传 `logs/htm_checks/checkpoint_inference.log`、`$INFER_OUT/report.json`、
`$INFER_OUT/predictions.json` 和 `inference_exit`。若失败，完整日志包括 traceback。
答错一题或两题不会导致加载验证失败；10 步模型和两题样本不用于判断准确率恢复。
报告中的得分仅供观察。

如需验证关闭 feedback 的路径，使用新的输出目录并将 `--feedback on` 改为
`--feedback off`。不要复用已有输出目录，脚本会拒绝覆盖先前诊断文件。
本机只验证了输入准备和脚本语法；真实 checkpoint 加载及 GPU 推理需由训练机运行。

======

## 使用现有 smoke16 标注重跑推理验证

之前的 `lvbench_qa_tau128.jsonl` 不存在，改用已确认的 smoke16 文件。
脚本仍只选择两题。激活 `video` 环境并进入仓库根目录后执行：

```bash
git pull --ff-only
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

RUN_DIR=/mnt/sdb1/wenbin/luoling/QVIC-MF-checkpoint/htm_stage_b_2xh100/gradient_smoke_20261013_134226
SLICE=/mnt/sdb1/wenbin/luoling/LVDU/eval/LVBench/data/lvbench_qa_smoke16.jsonl
VIDEO_DIR=/mnt/sdb1/wenbin/luoling/LVDU/eval/LVBench/data/video/all_videos
INFER_OUT="$RUN_DIR/inference_smoke_$(date +%Y%m%d_%H%M%S)"

mkdir -p logs/htm_checks
set -o pipefail
python scripts/tools/check_htm_checkpoint_inference.py \
  --checkpoint "$RUN_DIR/checkpoint-10" \
  --questions "$SLICE" \
  --video-dir "$VIDEO_DIR" \
  --outdir "$INFER_OUT" \
  --num-questions 2 \
  --frames 64 \
  --feedback on \
  2>&1 | tee logs/htm_checks/checkpoint_inference.log
INFER_EXIT=${PIPESTATUS[0]}
echo "inference_exit=$INFER_EXIT"
echo "inference_out=$INFER_OUT"
cat "$INFER_OUT/report.json"
```

请回传 `checkpoint_inference.log`、`$INFER_OUT/report.json`、
`$INFER_OUT/predictions.json` 和 `inference_exit`。

======

## 从 full LVBench 随机抽题检查推理顺序敏感性

脚本按固定种子从已有视频的题目中随机抽取两题，要求来自不同视频。
依次在独立模型进程中运行单题 A、单题 B、A→B、B→A；每次用相同
checkpoint、帧采样、memory 预算和 feedback 设置。完整推理需一张 GPU。

```bash
conda activate video
cd /mnt/data_nvme1/wenbin/luoling/cvpr/code/ours/QVIC-MF-Train
git fetch origin
git checkout fix/htm-gradient-diagnostics
git pull --ff-only

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
RUN_DIR=/mnt/sdb1/wenbin/luoling/QVIC-MF-checkpoint/htm_stage_b_2xh100/gradient_smoke_20261013_134226
FULL_QA=/mnt/sdb1/wenbin/luoling/LVDU/eval/LVBench/data/lvbench_qa_full.jsonl
VIDEO_DIR=/mnt/sdb1/wenbin/luoling/LVDU/eval/LVBench/data/video/all_videos
ORDER_OUT="$RUN_DIR/order_diag_$(date +%Y%m%d_%H%M%S)"

mkdir -p logs/htm_checks
set -o pipefail
python scripts/tools/check_htm_question_order.py \
  --checkpoint "$RUN_DIR/checkpoint-10" \
  --questions "$FULL_QA" \
  --video-dir "$VIDEO_DIR" \
  --outdir "$ORDER_OUT" \
  --seed 42 --frames 64 --feedback on \
  2>&1 | tee logs/htm_checks/question_order.log
ORDER_EXIT=${PIPESTATUS[0]}
echo "order_exit=$ORDER_EXIT"
echo "order_out=$ORDER_OUT"
```

成功后 `order_report.json` 汇总同一题在单题、正序和反序下的答案、
合并次数、最终记忆 token 数以及 τ 阈值。差异是诊断结果，不会使脚本失败。
请回传 `question_order.log`、`$ORDER_OUT/selection.json`、
`$ORDER_OUT/order_report.json` 和 `order_exit`。每次输出目录必须是新的。
这项检查使用 10 步 checkpoint 与每视频 64 帧，仅用于观察顺序敏感性。
检查不要求每个张量都变化。参数变化不能单独证明 QA 梯度来源或准确率恢复；
低精度舍入也可能使小更新在保存权重中不可见。

======

## 最新 checkpoint 比较命令

脚本已推送至 `fix/htm-gradient-diagnostics`，提交 `b3eecc1`。无需 GPU。
在该分支的训练机仓库根目录执行：

```bash
git pull --ff-only

RUN_DIR=/mnt/sdb1/wenbin/luoling/QVIC-MF-checkpoint/htm_stage_b_2xh100/gradient_smoke_20261013_134226

mkdir -p logs/htm_checks
set -o pipefail

python scripts/tools/check_htm_checkpoint_updates.py \
  --before "$RUN_DIR/checkpoint-5" \
  --after "$RUN_DIR/checkpoint-10" \
  --out logs/htm_checks/checkpoint_updates.json \
  2>&1 | tee logs/htm_checks/checkpoint_updates.log

COMPARE_EXIT=${PIPESTATUS[0]}
echo "compare_exit=$COMPARE_EXIT"
```

预期输出类似：

```text
compressor_lora: UPDATED changed=.../... tensors ...
context_seed: UPDATED changed=1/1 tensors ...
predictor: UPDATED changed=.../6 tensors ...
PASS: all three groups have stored parameter changes
```

若显示 `UNCHANGED`、`MISSING` 或 `NONFINITE`，需要继续排查。
请回传 `checkpoint_updates.log`、`checkpoint_updates.json` 和 `compare_exit`。

======

## 阈值跨视频状态修复后，重跑相同随机题目的顺序检查

评估新视频/题目开始时清空阈值历史；训练仍保留滚动历史。
使用相同 full 标注、checkpoint 和 seed=42，保持数据文件及媒体目录不变，
应再次抽到 `JlrzSvCsIjE_4004` 和 `5dZ_lvDgevk_914`。

```bash
conda activate video
cd /mnt/data_nvme1/wenbin/luoling/cvpr/code/ours/QVIC-MF-Train
git fetch origin
git checkout fix/htm-gradient-diagnostics
git pull --ff-only

python scripts/tools/test_htm_eval_threshold_reset.py

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
RUN_DIR=/mnt/sdb1/wenbin/luoling/QVIC-MF-checkpoint/htm_stage_b_2xh100/gradient_smoke_20261013_134226
FULL_QA=/mnt/sdb1/wenbin/luoling/LVDU/eval/LVBench/data/lvbench_qa_full.jsonl
VIDEO_DIR=/mnt/sdb1/wenbin/luoling/LVDU/eval/LVBench/data/video/all_videos
ORDER_OUT="$RUN_DIR/order_reset_diag_$(date +%Y%m%d_%H%M%S)"

mkdir -p logs/htm_checks
set -o pipefail
python scripts/tools/check_htm_question_order.py \
  --checkpoint "$RUN_DIR/checkpoint-10" \
  --questions "$FULL_QA" \
  --video-dir "$VIDEO_DIR" \
  --outdir "$ORDER_OUT" \
  --seed 42 --frames 64 --feedback on \
  2>&1 | tee logs/htm_checks/question_order_reset.log
ORDER_EXIT=${PIPESTATUS[0]}
echo "order_exit=$ORDER_EXIT"
echo "order_out=$ORDER_OUT"
```

预期 CPU 测试打印两行 PASS；真实推理报告的
`order_sensitive_logged_behavior` 为 false，各题相对 solo 的合并次数、
记忆长度及 τ 差值均为 0。若仍有差异，继续排查其他状态或数值非确定性。
两题答对与否不作为本项通过条件。
请回传 `question_order_reset.log`、`$ORDER_OUT/selection.json`、
`$ORDER_OUT/order_report.json` 和 `order_exit`。

======

## 事件 token 表示与写入预算消融

顺序重置复测已通过。接下来在相同随机 16 题、相同 checkpoint、采样和
feedback 下比较三组：pooled_256、tokens_256、tokens_4096。
pooled 每个事件保留一个向量；tokens 对事件内各帧的相同 token 位置平均，
保留 C=16 的结构。三组 relevance 记忆容量均固定为 256 个帧条目，
4096 仅是 HTM 写入 token 上限。这是无需重新训练的推理消融。

```bash
conda activate video
cd /mnt/data_nvme1/wenbin/luoling/cvpr/code/ours/QVIC-MF-Train
git fetch origin
git checkout fix/htm-gradient-diagnostics
git pull --ff-only
python scripts/tools/test_htm_event_tokens.py

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
RUN_DIR=/mnt/sdb1/wenbin/luoling/QVIC-MF-checkpoint/htm_stage_b_2xh100/gradient_smoke_20261013_134226
FULL_QA=/mnt/sdb1/wenbin/luoling/LVDU/eval/LVBench/data/lvbench_qa_full.jsonl
VIDEO_DIR=/mnt/sdb1/wenbin/luoling/LVDU/eval/LVBench/data/video/all_videos
ABLATION_OUT="$RUN_DIR/event_ablation_$(date +%Y%m%d_%H%M%S)"

mkdir -p logs/htm_checks
set -o pipefail
python scripts/tools/check_htm_event_ablation.py \
  --checkpoint "$RUN_DIR/checkpoint-10" \
  --questions "$FULL_QA" \
  --video-dir "$VIDEO_DIR" \
  --outdir "$ABLATION_OUT" \
  --num-questions 16 --seed 42 --frames 64 --feedback on \
  2>&1 | tee logs/htm_checks/event_ablation.log
ABLATION_EXIT=${PIPESTATUS[0]}
echo "ablation_exit=$ABLATION_EXIT"
echo "ablation_out=$ABLATION_OUT"
```

预期三个推理组均通过加载检查，最后输出 COMPLETE，
`ablation_report.json` 包含每组得分、平均记忆长度和逐题比较。
预计合并次数一致，因为 predictor 和 gating 仍使用相同 pooled 特征；
表示与预算影响的是 decoder 记忆内容及预算淘汰。
请回传 `event_ablation.log`、`$ABLATION_OUT/selection.json`、
`$ABLATION_OUT/ablation_report.json` 和 `ablation_exit`。
当前为 10 步模型、16 题、每视频均匀采样 64 帧，只能观察表示敏感性，
不能视作完整 LVBench 或训练后最终效果。

======

## 帧级不合并基线与事件时间排序消融

三组都使用 tokens 表示、4096 写入上限、256 relevance 帧容量：
`tokens_legacy_4096` 保持 recent→events 排列；
`tokens_chronological_4096` 在相同记忆选择完成后按时间排序；
`frames_4096` 不合并，每帧保留 16 tokens，按时间排列，移除 recent 重复条目。
64 帧的帧级基线应有 1024 tokens。事件组的实际长度较短，比较还涉及重复条目，
因此帧级结果不能单独归因于跨帧平均。时间排序组保留重复条目以隔离排序因素。
帧级基线输出原生 `[B, frames, C, D]` 分组，事件组继续使用扁平
`[B, tokens, 1, D]`；decoder 的 grid/frame 换行插入也受这一形状影响。
因此帧级对照包含分组恢复，实际 decoder 序列长度还包含换行 tokens。

```bash
conda activate video
cd /mnt/data_nvme1/wenbin/luoling/cvpr/code/ours/QVIC-MF-Train
git fetch origin
git checkout fix/htm-gradient-diagnostics
git pull --ff-only
python scripts/tools/test_htm_memory_layout.py

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
RUN_DIR=/mnt/sdb1/wenbin/luoling/QVIC-MF-checkpoint/htm_stage_b_2xh100/gradient_smoke_20261013_134226
FULL_QA=/mnt/sdb1/wenbin/luoling/LVDU/eval/LVBench/data/lvbench_qa_full.jsonl
VIDEO_DIR=/mnt/sdb1/wenbin/luoling/LVDU/eval/LVBench/data/video/all_videos
STRUCTURE_OUT="$RUN_DIR/structure_ablation_$(date +%Y%m%d_%H%M%S)"

mkdir -p logs/htm_checks
set -o pipefail
python scripts/tools/check_htm_event_ablation.py \
  --checkpoint "$RUN_DIR/checkpoint-10" \
  --questions "$FULL_QA" \
  --video-dir "$VIDEO_DIR" \
  --outdir "$STRUCTURE_OUT" \
  --ablation structure \
  --num-questions 16 --seed 42 --frames 64 --feedback on \
  2>&1 | tee logs/htm_checks/structure_ablation.log
STRUCTURE_EXIT=${PIPESTATUS[0]}
echo "structure_exit=$STRUCTURE_EXIT"
echo "structure_out=$STRUCTURE_OUT"
```

保持 full 标注和媒体可用性不变，应随机抽到上次相同的 16 题。
预期三个推理组完成；两个事件组的合并次数和记忆长度相同，帧级组
`n_merges=0` 且实际记忆为 1024 tokens。
请回传 `structure_ablation.log`、`$STRUCTURE_OUT/selection.json`、
`$STRUCTURE_OUT/ablation_report.json` 和 `structure_exit`。
当前 checkpoint 仅训练 10 步，这组仍是诊断，不用于判断最终准确率。

======

## 修复结构消融汇总错误：直接汇总已有结果

三组 GPU 推理已完成。最后的 `ERROR: 'pooled_256'` 是汇总代码引用了错误组名。
以下命令仅读取已保存的 report/predictions/日志，不加载模型，不需要 GPU。
它会重建顶层 `ablation_report.json`，不会重跑推理。

```bash
git pull --ff-only
python scripts/tools/test_htm_ablation_summary.py

# 在原终端直接复用 STRUCTURE_OUT；新终端需设置为上次打印的 structure_out。
: "${STRUCTURE_OUT:?请先设置 STRUCTURE_OUT 为上次结构消融的实际输出目录}"
mkdir -p logs/htm_checks
set -o pipefail
python scripts/tools/check_htm_event_ablation.py \
  --outdir "$STRUCTURE_OUT" \
  --ablation structure \
  --summarize-only \
  2>&1 | tee logs/htm_checks/structure_summary.log
SUMMARY_EXIT=${PIPESTATUS[0]}
echo "summary_exit=$SUMMARY_EXIT"
echo "structure_out=$STRUCTURE_OUT"
```

成功时输出 COMPLETE，退出码为 0。请回传 `$STRUCTURE_OUT/ablation_report.json`，
若失败则回传 `structure_summary.log` 和 `summary_exit`。

======

## 从头进行修复后的 Stage B 全量训练

本轮保留原始 HTM pooled/256 策略，建立梯度修复后的训练基线。
compressor LoRA 和 context seed 重新初始化；predictor 使用 Stage A
checkpoint-5000 初始化。不会恢复旧 Stage B 或十步短跑权重。
全部数据、1 epoch、2 GPU、batch=1、累积=8（全局 batch=16），ZeRO-3，
lr=1e-4、warmup=0.03、λ_pred=0.1。训练 K_r=0。

wrapper 清除 MAX_RECORDS/EXTRA_ARGS 的短跑限制。默认要求新输出目录；
每 500 步保存，保留最近 3 个 checkpoint，启动配置写入 launch_config.json。
若有之前确认的坏媒体列表，可在启动前设置 SKIP_LIST_PATH 为其实际路径。

```bash
conda activate video
cd /mnt/data_nvme1/wenbin/luoling/cvpr/code/ours/QVIC-MF-Train
git fetch origin
git checkout fix/htm-gradient-diagnostics
git pull --ff-only

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1
export OUTPUT_DIR="/mnt/sdb1/wenbin/luoling/QVIC-MF-checkpoint/htm_stage_b_fresh_$(date +%Y%m%d_%H%M%S)"
export HTM_INIT_CHECKPOINT=/mnt/sdb1/wenbin/luoling/QVIC-MF-checkpoint/htm_stage_a_2xh100/checkpoint-5000
export DATASET_ROOT=/mnt/sdb1/wenbin/luoling/LVDU/LLaVA-Video-83K
export ANNOTATION_PATH=/mnt/sdb1/wenbin/luoling/LLaVA-Video-83K/annotations/all_sampled.json

mkdir -p logs/htm_checks
TRAIN_LOG="logs/htm_checks/$(basename "$OUTPUT_DIR").log"
set -o pipefail
RESUME=0 bash scripts/train/train_htm_stage_b_fresh_2xh100.sh \
  2>&1 | tee "$TRAIN_LOG"
TRAIN_EXIT=${PIPESTATUS[0]}
echo "training_exit=$TRAIN_EXIT"
echo "output_dir=$OUTPUT_DIR"
echo "training_log=$TRAIN_LOG"
```

使用持久终端会话运行长任务。启动日志应显示全部数据（约 83K records），
global batch=16、从头训练，并成功加载 6 个 Stage A predictor tensors。
初始若显示 0/10 或仅 64 records，停止检查配置。总优化器步数随有效样本数确定；
83K single-QA 样本约 5188 步，实际以 Trainer 为准。

先回传启动配置、前 20 个优化器步骤的 loss/grad_norm/显存日志。
checkpoint-500 生成后，回传保存日志和 launch_config.json，再安排参数更新及评估。
暂不根据训练 loss 判断 LVBench 效果。

### 中断后恢复这一轮

重新激活环境，进入仓库，并将 OUTPUT_DIR 设置为上次打印的实际目录。
恢复时不要生成新的时间戳；保持同一训练配置和 GPU 数量。

```bash
set -o pipefail
RESUME=1 bash scripts/train/train_htm_stage_b_fresh_2xh100.sh \
  2>&1 | tee -a "logs/htm_checks/$(basename "$OUTPUT_DIR").log"
TRAIN_EXIT=${PIPESTATUS[0]}
echo "training_exit=$TRAIN_EXIT"
```

恢复日志应显示 resuming from checkpoint-*，而不是 starting from scratch。

======

## 已确认的新方案：复用 Stage A，16-token 事件先短训

Stage A predictor 的输入/目标仍是每帧 16 context tokens 的平均向量，
因此无需为存储方式变化重训 Stage A。新 Stage B 继续更新 predictor。
新方案为 tokens/4096、chronological、grouped decoder memory，W=4、q=0.9、
λ_pred=0.1；训练与评估均在每个新视频重置阈值。Compressor LoRA/context seed
从头初始化；decoder、视觉编码器/projector 冻结。训练 K_r=0。

这里替代前面的 pooled 全量启动方案。新 checkpoint 保存全部 HTM 配置，
评估默认从 config.json 恢复。不要用旧的 pooled wrapper 开始这轮长训练。

### 第一步：完整模型分组梯度检查

```bash
conda activate video
cd /mnt/data_nvme1/wenbin/luoling/cvpr/code/ours/QVIC-MF-Train
git fetch origin
git checkout fix/htm-gradient-diagnostics
git pull --ff-only

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p logs/htm_checks
set -o pipefail
PYTHONPATH=. python scripts/train/selftest_train_cpu.py \
  --htm --htm-grouped --memory-length 64 \
  2>&1 | tee logs/htm_checks/grouped_gradients.log
GRAD_EXIT=${PIPESTATUS[0]}
echo "grouped_gradient_exit=$GRAD_EXIT"
```

预期 memory 的第三维为 16，seed、LoRA、predictor 梯度均非零，
checkpoint 配置检查通过，最终 SELFTEST OK。通过后才做以下 GPU 短训。

### 第二步：固定 128 records、100 步 GPU pilot

pilot 用双 GPU、累积 4（全局 batch=8），在固定样本上重复训练以检查
QA 可学习性；它与全量模式的累积 8 区分。每 25 步保存，共 4 个 checkpoint。

```bash
export CUDA_VISIBLE_DEVICES=0,1
export HTM_INIT_CHECKPOINT=/mnt/sdb1/wenbin/luoling/QVIC-MF-checkpoint/htm_stage_a_2xh100/checkpoint-5000
export OUTPUT_DIR="/mnt/sdb1/wenbin/luoling/QVIC-MF-checkpoint/htm_tokens_pilot_$(date +%Y%m%d_%H%M%S)"
PILOT_LOG="logs/htm_checks/$(basename "$OUTPUT_DIR").log"
set -o pipefail
MODE=pilot RESUME=0 bash scripts/train/train_htm_stage_b_tokens_2xh100.sh \
  2>&1 | tee "$PILOT_LOG"
PILOT_EXIT=${PIPESTATUS[0]}
echo "pilot_exit=$PILOT_EXIT"
echo "pilot_dir=$OUTPUT_DIR"
```

日志应显示 tokens、4096、chronological、grouped=True，并加载 6 个
Stage A predictor tensors。训练字典新增 htm_qa_loss_local、
htm_prediction_loss_local、htm_final_M_local、htm_merge_rate_local。
这些是日志所在 rank 的 microbatch 均值，区别于 Trainer 汇总的总 loss。
请回传 grouped_gradients.log、完整 PILOT_LOG、pilot_dir/exit 和 launch_config.json。
先确认 QA loss 趋势和 checkpoint-25→100 参数变化；短训 loss 不降时检查原因，
不直接投入全量。短训过拟合不证明泛化或 LVBench 收益。

### 第三步：pilot 确认后再启动全量

全量仍从 Stage A predictor 和初始 compressor/seed 开始，使用新的输出目录。
不要恢复 pilot 的优化器/调度器状态。全量为 1 epoch、全局 batch=16。

```bash
export OUTPUT_DIR="/mnt/sdb1/wenbin/luoling/QVIC-MF-checkpoint/htm_tokens_full_$(date +%Y%m%d_%H%M%S)"
FULL_LOG="logs/htm_checks/$(basename "$OUTPUT_DIR").log"
set -o pipefail
MODE=full RESUME=0 bash scripts/train/train_htm_stage_b_tokens_2xh100.sh \
  2>&1 | tee "$FULL_LOG"
TRAIN_EXIT=${PIPESTATUS[0]}
echo "training_exit=$TRAIN_EXIT"
echo "output_dir=$OUTPUT_DIR"
```

恢复全量需将 OUTPUT_DIR 设为这次的实际目录，并保留 MODE=full：

```bash
set -o pipefail
MODE=full RESUME=1 bash scripts/train/train_htm_stage_b_tokens_2xh100.sh \
  2>&1 | tee -a "logs/htm_checks/$(basename "$OUTPUT_DIR").log"
TRAIN_EXIT=${PIPESTATUS[0]}
echo "training_exit=$TRAIN_EXIT"
```

======

## 16-token pilot：checkpoint-25 / checkpoint-50 长训前验收

在训练服务器执行。GPU 推理请在 pilot 结束、GPU 空闲后运行。
两次推理使用同一份 seed=42 随机抽取的 16 道题，优先不同视频。
这是 LVBench 独立推理诊断；未核查它与训练集的视频重叠，不能据此声称无泄漏泛化。
评估不覆盖 checkpoint 配置，以检查保存的 tokens/4096/chronological/grouped 设置。
整段在子 shell 内运行；检查失败只结束子 shell，保留交互终端。

```bash
(
conda activate video
cd /mnt/data_nvme1/wenbin/luoling/cvpr/code/ours/QVIC-MF-Train
git pull --ff-only
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PILOT_DIR=/mnt/sdb1/wenbin/luoling/QVIC-MF-checkpoint/htm_tokens_pilot_20261013_160959
export FULL_QA=/mnt/sdb1/wenbin/luoling/LVDU/eval/LVBench/data/lvbench_qa_full.jsonl
export VIDEO_DIR=/mnt/sdb1/wenbin/luoling/LVDU/eval/LVBench/data/video/all_videos
export CHECK_OUT="$PILOT_DIR/acceptance_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$CHECK_OUT"

# 此检查仅用 CPU；失败时不要继续 GPU 推理。
python scripts/tools/check_htm_checkpoint_updates.py \
  --before "$PILOT_DIR/checkpoint-25" \
  --after "$PILOT_DIR/checkpoint-50" \
  --out "$CHECK_OUT/checkpoint_updates.json" || exit $?

python - <<'PY'
import json, os, random
from pathlib import Path
from scripts.tools.check_htm_checkpoint_inference import read_rows, select_rows

pilot = Path(os.environ["PILOT_DIR"])
expected = {
    "htm_token_budget": 4096,
    "htm_event_representation": "tokens",
    "htm_memory_layout": "chronological",
    "htm_group_tokens": True,
    "htm_reset_thresholds_per_video": True,
}
for step in (25, 50):
    config = json.loads((pilot / f"checkpoint-{step}" / "config.json").read_text())
    for key, value in expected.items():
        if config.get(key) != value:
            raise SystemExit(f"checkpoint-{step}: {key}={config.get(key)!r}, expected {value!r}")
rows = read_rows(Path(os.environ["FULL_QA"]))
random.Random(42).shuffle(rows)
selected = select_rows(rows, Path(os.environ["VIDEO_DIR"]), 16)
subset = Path(os.environ["CHECK_OUT"]) / "questions.jsonl"
subset.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected))
print("PASS: checkpoint configs; fixed random subset:", [row["key"] for row in selected])
PY
CONFIG_EXIT=$?
if [ "$CONFIG_EXIT" -ne 0 ]; then exit "$CONFIG_EXIT"; fi

export CUDA_VISIBLE_DEVICES=0
for STEP in 25 50; do
  python scripts/tools/check_htm_checkpoint_inference.py \
    --checkpoint "$PILOT_DIR/checkpoint-$STEP" \
    --questions "$CHECK_OUT/questions.jsonl" \
    --video-dir "$VIDEO_DIR" \
    --outdir "$CHECK_OUT/checkpoint-$STEP" \
    --num-questions 16 --frames 64 --feedback on || exit $?
done

python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ["CHECK_OUT"])
predictions = {}
for step in (25, 50):
    folder = root / f"checkpoint-{step}"
    report = json.loads((folder / "report.json").read_text())
    if not report["passed"]:
        raise SystemExit(f"checkpoint-{step}: inference checks failed")
    print(f"checkpoint-{step}: {report['score']['correct']}/16; checks={report['checks']}")
    predictions[step] = {row["key"]: row for row in json.loads((folder / "predictions.json").read_text())}
if predictions[25].keys() != predictions[50].keys():
    raise SystemExit("ERROR: question sets differ")
print("PASS: both reload/inference runs; inspect inference.log HTM telemetry before full training")
print("Artifacts:", root)
PY
)
```

请回传 checkpoint_updates.json、两份 report.json 和 inference.log 中的 `[HTM τ]` 行。
参数更新和重载检查通过、没有非有限值或异常记忆坍缩后，可使用上一节的
MODE=full RESUME=0 命令，从 Stage A predictor 开始新目录全量训练。
16 题准确率用于诊断，不设必须上涨的门槛，也不作为最终准确率结论。

======

## pilot 验收推理失败后的重试（保留终端和错误日志）

前两项已 PASS 时无需重复检查。重新打开 session 后激活环境，下面自动复用
最近 acceptance 目录的 questions.jsonl，并打印所选路径。每次使用新的推理
输出目录，避免已有诊断文件触发拒绝覆盖。所有 exit 均在括号子 shell 内。

```bash
conda activate video
cd /mnt/data_nvme1/wenbin/luoling/cvpr/code/ours/QVIC-MF-Train
git pull --ff-only
(
set -o pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
export PILOT_DIR=/mnt/sdb1/wenbin/luoling/QVIC-MF-checkpoint/htm_tokens_pilot_20261013_160959
VIDEO_DIR=/mnt/sdb1/wenbin/luoling/LVDU/eval/LVBench/data/video/all_videos
QUESTIONS=$(python - <<'PY'
import os
from pathlib import Path
paths = list(Path(os.environ["PILOT_DIR"]).glob("acceptance_*/questions.jsonl"))
if not paths:
    raise SystemExit("ERROR: no prepared acceptance questions.jsonl found")
print(max(paths, key=lambda path: path.stat().st_mtime))
PY
) || exit $?
RETRY_OUT="$PILOT_DIR/inference_retry_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RETRY_OUT" || exit $?
echo "questions=$QUESTIONS"
echo "retry_out=$RETRY_OUT"
for STEP in 25 50; do
  python -u scripts/tools/check_htm_checkpoint_inference.py \
    --checkpoint "$PILOT_DIR/checkpoint-$STEP" \
    --questions "$QUESTIONS" \
    --video-dir "$VIDEO_DIR" \
    --outdir "$RETRY_OUT/checkpoint-$STEP" \
    --num-questions 16 --frames 64 --feedback on \
    2>&1 | tee "$RETRY_OUT/checkpoint-$STEP.driver.log"
  RUN_EXIT=$?
  echo "checkpoint-$STEP exit=$RUN_EXIT"
  if [ "$RUN_EXIT" -ne 0 ]; then
    echo "推理失败，错误日志：$RETRY_OUT/checkpoint-$STEP.driver.log"
    tail -80 "$RETRY_OUT/checkpoint-$STEP.driver.log"
    exit "$RUN_EXIT"
  fi
done
echo "PASS: both inference checks; reports under $RETRY_OUT"
)
```

若仍失败，回传 driver.log 最后 80 行（包含 ERROR/Traceback）及打印的退出码。
当前尚不能从 session 退出判断是路径、依赖、显存还是 checkpoint 加载问题。

======

## 全量 Stage B 上报 W&B

tokens wrapper 全量模式默认 REPORT_TO=wandb，pilot 默认 none；可显式覆盖。
Trainer 将 loss、grad_norm、learning_rate 和四个 htm_*_local 指标一起上报。
HTM 指标是日志 rank 的 microbatch 均值；总 loss 与它们的加权和不要求完全相等。
下面只启动新的全量训练，复用 Stage A predictor，不恢复 pilot。

```bash
conda activate video
cd /mnt/data_nvme1/wenbin/luoling/cvpr/code/ours/QVIC-MF-Train
git pull --ff-only
python -c 'import wandb; print("wandb", wandb.__version__)'
# 若上一行提示未安装，先在 video 环境运行：python -m pip install wandb
# 尚未登录时执行；在交互提示输入 API key，不要写入命令或仓库：
wandb login

export CUDA_VISIBLE_DEVICES=0,1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HTM_INIT_CHECKPOINT=/mnt/sdb1/wenbin/luoling/QVIC-MF-checkpoint/htm_stage_a_2xh100/checkpoint-5000
export OUTPUT_DIR="/mnt/sdb1/wenbin/luoling/QVIC-MF-checkpoint/htm_tokens_full_$(date +%Y%m%d_%H%M%S)"
export REPORT_TO=wandb WANDB_PROJECT=qvic-htm-stage-b WANDB_MODE=online
# 使用团队空间时，可另外设置 WANDB_ENTITY 为实际账号或团队名称。
export RUN_NAME="$(basename "$OUTPUT_DIR")"
mkdir -p logs/htm_checks
FULL_LOG="logs/htm_checks/$RUN_NAME.log"
(
set -o pipefail
MODE=full RESUME=0 bash scripts/train/train_htm_stage_b_tokens_2xh100.sh \
  2>&1 | tee "$FULL_LOG"
TRAIN_EXIT=$?
echo "training_exit=$TRAIN_EXIT"
echo "output_dir=$OUTPUT_DIR"
)
```

启动日志应显示 report_to=wandb 及 W&B run 链接。HF 离线变量仅影响模型下载，
W&B online 上报需要训练机器能访问 W&B。这里的命令不自动启动训练。
