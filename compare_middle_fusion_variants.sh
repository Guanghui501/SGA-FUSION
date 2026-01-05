#!/bin/bash
# 中期融合变体对比实验脚本
# 用于评估带注意力的中期融合对训练精度的实际影响

set -e

# 配置
DATASET="jarvis"
PROPERTY="formation_energy_peratom"
EPOCHS=100
BATCH_SIZE=32
LEARNING_RATE=1e-3

# 输出目录
OUTPUT_DIR="./middle_fusion_comparison_results"
mkdir -p $OUTPUT_DIR

echo "========================================================================"
echo "🧪 中期融合变体对比实验"
echo "========================================================================"
echo "数据集: $DATASET"
echo "属性: $PROPERTY"
echo "训练轮数: $EPOCHS"
echo "========================================================================"

# 实验1: 基线 - 原始中期融合
echo ""
echo "▶️  实验1: 基线 - 原始中期融合"
echo "--------------------------------------------------------------------"
python SGA-fusion/train_with_cross_modal_attention.py \
    --dataset $DATASET \
    --property $PROPERTY \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --use_middle_fusion True \
    --use_middle_fusion_attention False \
    --middle_fusion_layers "2" \
    --middle_fusion_num_heads 4 \
    --use_fine_grained_attention False \
    --use_cross_modal True \
    --output_dir "${OUTPUT_DIR}/exp1_baseline_middle" \
    2>&1 | tee "${OUTPUT_DIR}/exp1_baseline.log"

# 实验2: 纯注意力模式
echo ""
echo "▶️  实验2: 带注意力的中期融合 - 纯注意力模式"
echo "--------------------------------------------------------------------"
python SGA-fusion/train_with_cross_modal_attention.py \
    --dataset $DATASET \
    --property $PROPERTY \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --use_middle_fusion True \
    --use_middle_fusion_attention True \
    --middle_fusion_attention_use_gate False \
    --middle_fusion_layers "2" \
    --middle_fusion_num_heads 4 \
    --middle_fusion_hidden_dim 128 \
    --middle_fusion_dropout 0.15 \
    --use_fine_grained_attention False \
    --use_cross_modal True \
    --output_dir "${OUTPUT_DIR}/exp2_attention_only" \
    2>&1 | tee "${OUTPUT_DIR}/exp2_attention_only.log"

# 实验3: 混合模式 (推荐)
echo ""
echo "▶️  实验3: 带注意力的中期融合 - 混合模式 (注意力+门控)"
echo "--------------------------------------------------------------------"
python SGA-fusion/train_with_cross_modal_attention.py \
    --dataset $DATASET \
    --property $PROPERTY \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --use_middle_fusion True \
    --use_middle_fusion_attention True \
    --middle_fusion_attention_use_gate True \
    --middle_fusion_layers "2" \
    --middle_fusion_num_heads 4 \
    --middle_fusion_hidden_dim 128 \
    --middle_fusion_dropout 0.15 \
    --use_fine_grained_attention False \
    --use_cross_modal True \
    --output_dir "${OUTPUT_DIR}/exp3_hybrid_mode" \
    2>&1 | tee "${OUTPUT_DIR}/exp3_hybrid.log"

# 实验4: 更多注意力头 (8头)
echo ""
echo "▶️  实验4: 混合模式 + 更多注意力头 (8头)"
echo "--------------------------------------------------------------------"
python SGA-fusion/train_with_cross_modal_attention.py \
    --dataset $DATASET \
    --property $PROPERTY \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --use_middle_fusion True \
    --use_middle_fusion_attention True \
    --middle_fusion_attention_use_gate True \
    --middle_fusion_layers "2" \
    --middle_fusion_num_heads 8 \
    --middle_fusion_hidden_dim 128 \
    --middle_fusion_dropout 0.15 \
    --use_fine_grained_attention False \
    --use_cross_modal True \
    --output_dir "${OUTPUT_DIR}/exp4_8heads" \
    2>&1 | tee "${OUTPUT_DIR}/exp4_8heads.log"

# 生成结果汇总
echo ""
echo "========================================================================"
echo "📊 生成结果汇总..."
echo "========================================================================"

python - <<EOF
import re
import json
from pathlib import Path

results_dir = Path("$OUTPUT_DIR")
experiments = [
    ("exp1_baseline.log", "基线: 原始中期融合"),
    ("exp2_attention_only.log", "纯注意力模式"),
    ("exp3_hybrid.log", "混合模式 (推荐)"),
    ("exp4_8heads.log", "混合模式 + 8头"),
]

print("\n" + "="*80)
print("📊 中期融合变体对比实验结果汇总")
print("="*80)
print(f"{'实验':<30} {'验证MAE':<15} {'测试MAE':<15} {'训练时间':<15}")
print("-"*80)

for log_file, exp_name in experiments:
    log_path = results_dir / log_file
    if not log_path.exists():
        print(f"{exp_name:<30} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
        continue

    with open(log_path, 'r') as f:
        content = f.read()

    # 提取最佳验证MAE
    val_mae_match = re.search(r'Best validation MAE: ([\d.]+)', content)
    val_mae = float(val_mae_match.group(1)) if val_mae_match else "N/A"

    # 提取测试MAE
    test_mae_match = re.search(r'Test MAE: ([\d.]+)', content)
    test_mae = float(test_mae_match.group(1)) if test_mae_match else "N/A"

    # 提取训练时间
    time_match = re.search(r'Training time: ([\d.]+)', content)
    train_time = f"{float(time_match.group(1)):.1f}s" if time_match else "N/A"

    print(f"{exp_name:<30} {str(val_mae):<15} {str(test_mae):<15} {train_time:<15}")

print("="*80)
print("\n💡 分析:")
print("  - 比较验证MAE和测试MAE，评估精度变化")
print("  - 比较训练时间，评估计算开销")
print("  - 推荐使用混合模式 (exp3) 作为平衡选择")
print("\n结果已保存至: $OUTPUT_DIR/")
EOF

echo ""
echo "✅ 对比实验完成！"
echo "结果目录: $OUTPUT_DIR"
