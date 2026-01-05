# 中期融合注意力机制 (Middle Fusion with Attention)

## 概述

本功能为SGA-FUSION模型添加了**带注意力机制的中期融合模块**，在保持中期融合高精度的同时，增加了细粒度的可解释性。

### 核心优势

✅ **高精度** - 保持中期融合的性能优势
✅ **可解释性** - 提供原子-文本对应关系的注意力权重
✅ **灵活性** - 支持纯注意力或混合模式（注意力+门控）
✅ **轻量级** - 相比细粒度融合，计算开销更小

---

## 实现原理

### 1. 核心组件

**`MiddleFusionWithAttention`** 类（位于 `models/alignn.py:254-453`）

```python
class MiddleFusionWithAttention(nn.Module):
    """
    增强版中期融合模块，结合：
    - 中期融合的效率（在ALIGNN中间层注入文本）
    - 细粒度注意力的可解释性（原子-文本对应关系）
    """
```

### 2. 工作流程

```
┌─────────────────────────────────────────────────────────────┐
│                ALIGNN层更新（第0,1,2,...层）                   │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼ 在指定层（如第2层）
┌─────────────────────────────────────────────────────────────┐
│  中期融合注意力模块                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 1. 将节点特征转换为批处理格式                           │   │
│  │    [total_atoms, dim] → [batch, max_atoms, dim]      │   │
│  │                                                       │   │
│  │ 2. 多头注意力计算                                      │   │
│  │    Q = node_feat @ W_q                               │   │
│  │    K = text_tokens @ W_k                             │   │
│  │    V = text_tokens @ W_v                             │   │
│  │    Attention = softmax(QK^T / √d) V                  │   │
│  │                                                       │   │
│  │ 3. 可选门控机制（混合模式）                            │   │
│  │    gate = sigmoid(Linear([node; context]))           │   │
│  │    output = node + gate * context                    │   │
│  │                                                       │   │
│  │ 4. 转换回DGL格式并返回注意力权重                       │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│         继续后续ALIGNN层和GCN层的更新                          │
└─────────────────────────────────────────────────────────────┘
```

### 3. 注意力权重格式

返回的注意力权重：
```python
middle_fusion_attention_weights = {
    'layer_2': torch.Tensor([batch, num_heads, num_atoms, seq_len])
    # shape说明：
    # - batch: 批次大小
    # - num_heads: 注意力头数（默认4）
    # - num_atoms: 该晶体结构中的原子数
    # - seq_len: 文本token序列长度
}
```

---

## 使用方法

### 方法1：配置文件启用

在训练配置中添加：

```python
config = ALIGNNConfig(
    # ... 其他配置 ...

    # ✅ 启用中期融合
    use_middle_fusion=True,

    # ✅ 启用注意力机制（关键！）
    use_middle_fusion_attention=True,

    # 融合层配置
    middle_fusion_layers="2",  # 在第2层ALIGNN后注入文本

    # 注意力配置
    middle_fusion_hidden_dim=128,
    middle_fusion_num_heads=4,
    middle_fusion_dropout=0.1,

    # 混合模式（推荐）
    middle_fusion_attention_use_gate=True,  # 注意力 + 门控
    middle_fusion_attention_use_projection=True,
)
```

### 方法2：命令行参数启用

```bash
python train_with_cross_modal_attention.py \
    --dataset jarvis \
    --property formation_energy \
    --use_middle_fusion True \
    --use_middle_fusion_attention True \
    --middle_fusion_layers "2" \
    --middle_fusion_num_heads 4 \
    --middle_fusion_attention_use_gate True
```

### 方法3：推理时提取注意力权重

```python
import torch
from models.alignn import ALIGNN, ALIGNNConfig

# 加载模型
config = ALIGNNConfig(
    use_middle_fusion=True,
    use_middle_fusion_attention=True,
    # ... 其他配置 ...
)
model = ALIGNN(config)
model.load_state_dict(torch.load('checkpoint.pt'))
model.eval()

# 推理并提取注意力权重
with torch.no_grad():
    output = model(
        (g_batch, lg_batch, [text]),
        return_attention=True  # 🔑 关键参数
    )

# 提取注意力权重
middle_attn = output['middle_fusion_attention_weights']
for layer_name, attn_weights in middle_attn.items():
    print(f"{layer_name}: {attn_weights.shape}")
    # layer_2: torch.Size([1, 4, 32, 15])
    #          (batch=1, heads=4, atoms=32, tokens=15)
```

---

## 可视化与分析

### 1. 运行演示脚本

```bash
python demo_middle_fusion_attention.py
```

输出：
- 注意力热力图（PNG图片）
- 原子-文本对应关系分析

### 2. 自定义可视化

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(attn_weights, text, atom_symbols):
    """
    可视化注意力权重

    Args:
        attn_weights: [batch, heads, atoms, tokens]
        text: 文本描述
        atom_symbols: 原子符号列表
    """
    # 平均所有头
    attn = attn_weights[0].mean(dim=0).cpu().numpy()

    tokens = text.split()

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        attn[:, :len(tokens)],
        xticklabels=tokens,
        yticklabels=atom_symbols,
        cmap='YlOrRd'
    )
    plt.xlabel('Text Tokens')
    plt.ylabel('Atoms')
    plt.title('Middle Fusion Attention Weights')
    plt.show()
```

### 3. 分析原子-文本对应关系

```python
def analyze_correspondences(attn_weights, text, atom_symbols, top_k=3):
    """找到每个原子最关注的文本tokens"""
    attn = attn_weights[0].mean(dim=0).cpu().numpy()
    tokens = text.split()

    for i, atom in enumerate(atom_symbols):
        top_indices = attn[i].argsort()[-top_k:][::-1]
        print(f"原子 {atom}_{i} 最关注:")
        for idx in top_indices:
            print(f"  - '{tokens[idx]}' (权重: {attn[i, idx]:.4f})")
```

---

## 配置参数详解

### 核心参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_middle_fusion_attention` | bool | False | 是否启用注意力版本的中期融合 |
| `middle_fusion_attention_use_gate` | bool | True | 是否使用混合模式（注意力+门控） |
| `middle_fusion_attention_use_projection` | bool | True | 是否投影文本token到隐藏维度 |

### 继承的中期融合参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `middle_fusion_layers` | str | "2" | 融合层索引（逗号分隔，如"2,3"） |
| `middle_fusion_hidden_dim` | int | 128 | 注意力隐藏维度 |
| `middle_fusion_num_heads` | int | 2 | 注意力头数（推荐4-8） |
| `middle_fusion_dropout` | float | 0.1 | Dropout率 |

---

## 模式对比

### 模式1：纯注意力模式
```python
use_middle_fusion_attention=True
middle_fusion_attention_use_gate=False
```
- 优点：完全基于注意力，可解释性最强
- 缺点：可能不如混合模式稳定

### 模式2：混合模式（推荐）
```python
use_middle_fusion_attention=True
middle_fusion_attention_use_gate=True
```
- 优点：结合注意力和门控机制，精度和可解释性兼顾
- 缺点：参数略多

### 模式3：原始中期融合
```python
use_middle_fusion=True
use_middle_fusion_attention=False
```
- 优点：参数少，训练快
- 缺点：无法获取注意力权重

---

## 与其他融合方式的对比

| 融合方式 | 精度 | 可解释性 | 计算开销 | 适用场景 |
|---------|------|---------|---------|---------|
| **中期融合（原始）** | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐ | 追求高精度 |
| **中期融合注意力** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | **兼顾精度和可解释性** |
| **细粒度融合** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 追求最强可解释性 |
| **跨模态融合** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | 全局语义融合 |

---

## 常见问题

### Q1: 中期融合注意力与细粒度融合有什么区别？

**A:**
- **位置不同**: 中期融合在ALIGNN中间层注入，细粒度融合在readout之前
- **开销不同**: 中期融合更轻量（注意力头数少，只在指定层）
- **精度不同**: 中期融合精度通常更高
- **可解释性**: 细粒度融合提供完整的双向注意力，中期融合只提供atom-to-text

### Q2: 如何选择融合层？

**A:**
- 推荐在**中间层**（如第2层，共4层ALIGNN时）
- 太早（第0层）：原子特征未充分编码
- 太晚（第3层）：接近readout，效果类似后期融合

### Q3: 混合模式和纯注意力模式哪个好？

**A:**
- **混合模式（推荐）**: 精度更高，训练更稳定
- **纯注意力模式**: 如果你只关心可解释性，可以尝试

### Q4: 可以同时启用中期融合注意力和细粒度融合吗？

**A:**
- 技术上可以，但不推荐（计算开销大，可能过拟合）
- 建议选择一种主要融合方式

---

## 引用

如果您在研究中使用了本功能，请引用：

```bibtex
@article{sga-fusion-middle-attention,
  title={Middle Fusion with Attention for Interpretable Multi-Modal Materials Property Prediction},
  author={Your Team},
  journal={arXiv preprint},
  year={2026}
}
```

---

## 更新日志

### v1.0.0 (2026-01-05)
- ✅ 实现 `MiddleFusionWithAttention` 类
- ✅ 添加配置选项 `use_middle_fusion_attention`
- ✅ 支持注意力权重提取和可视化
- ✅ 提供演示脚本和完整文档

---

## 联系方式

如有问题或建议，请通过以下方式联系：
- GitHub Issues
- Email: your-email@example.com

---

**Happy coding! 🚀**
