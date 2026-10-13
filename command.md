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
