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
检查不要求每个张量都变化。参数变化不能单独证明 QA 梯度来源或准确率恢复；
低精度舍入也可能使小更新在保存权重中不可见。
