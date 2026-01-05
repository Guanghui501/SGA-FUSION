#!/usr/bin/env python
"""
演示脚本：带注意力的中期融合（Middle Fusion with Attention）
=============================================================

功能：
1. 展示如何启用带注意力的中期融合
2. 提取并可视化文本-原子对应关系
3. 结合中期融合的高精度和细粒度融合的可解释性

作者：Claude Code
日期：2026-01-05
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from jarvis.core.atoms import Atoms
from jarvis.db.figshare import data as jdata
import sys
import os

# 添加路径以导入自定义模块
sys.path.append(os.path.join(os.path.dirname(__file__), 'SGA-fusion'))

from models.alignn import ALIGNN, ALIGNNConfig
from models.utils import prepare_line_graph_batch


def visualize_middle_fusion_attention(attention_weights, text, atom_symbols, layer_name, save_path=None):
    """
    可视化中期融合的注意力权重（文本-原子对应关系）

    Args:
        attention_weights: [batch, num_heads, num_atoms, seq_len] 注意力权重张量
        text: 文本描述字符串
        atom_symbols: 原子符号列表 (e.g., ['Si', 'O', 'O', ...])
        layer_name: 层名称 (e.g., 'layer_2')
        save_path: 保存路径（可选）
    """
    # 提取单个样本（假设batch_size=1）
    attn = attention_weights[0]  # [num_heads, num_atoms, seq_len]

    # 对所有注意力头求平均
    attn_avg = attn.mean(dim=0).cpu().numpy()  # [num_atoms, seq_len]

    # 分词（简单分割）
    tokens = text.split()

    # 确保维度匹配（可能有padding）
    num_atoms = len(atom_symbols)
    num_tokens = len(tokens)
    attn_viz = attn_avg[:num_atoms, :num_tokens]

    # 创建图形
    fig, ax = plt.subplots(figsize=(max(12, num_tokens * 0.5), max(8, num_atoms * 0.3)))

    # 绘制热力图
    sns.heatmap(
        attn_viz,
        xticklabels=tokens,
        yticklabels=atom_symbols,
        cmap='YlOrRd',
        cbar_kws={'label': 'Attention Weight'},
        linewidths=0.5,
        linecolor='gray',
        ax=ax
    )

    ax.set_xlabel('Text Tokens', fontsize=12, fontweight='bold')
    ax.set_ylabel('Atoms', fontsize=12, fontweight='bold')
    ax.set_title(f'Middle Fusion Attention Weights - {layer_name}\n'
                 f'Text: "{text}"',
                 fontsize=14, fontweight='bold', pad=20)

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 注意力热力图已保存至: {save_path}")

    plt.show()


def analyze_atom_text_correspondence(attention_weights, text, atom_symbols, top_k=3):
    """
    分析每个原子最关注的文本tokens

    Args:
        attention_weights: [batch, num_heads, num_atoms, seq_len]
        text: 文本描述
        atom_symbols: 原子符号列表
        top_k: 显示每个原子最关注的前k个词

    Returns:
        对应关系字典
    """
    # 提取单个样本并平均所有头
    attn = attention_weights[0].mean(dim=0).cpu().numpy()  # [num_atoms, seq_len]

    tokens = text.split()
    num_atoms = len(atom_symbols)
    num_tokens = len(tokens)

    attn_viz = attn[:num_atoms, :num_tokens]

    print("\n" + "="*80)
    print("📊 原子-文本对应关系分析")
    print("="*80)

    correspondences = {}
    for i, atom in enumerate(atom_symbols):
        # 获取该原子对所有token的注意力权重
        atom_attn = attn_viz[i]

        # 找到top-k最关注的词
        top_indices = np.argsort(atom_attn)[-top_k:][::-1]
        top_tokens = [(tokens[idx], atom_attn[idx]) for idx in top_indices]

        correspondences[f"{atom}_{i}"] = top_tokens

        print(f"\n原子 #{i} ({atom}):")
        for rank, (token, weight) in enumerate(top_tokens, 1):
            print(f"  {rank}. '{token}' (权重: {weight:.4f})")

    print("\n" + "="*80)

    return correspondences


def main():
    """主函数：演示带注意力的中期融合"""

    print("\n" + "="*80)
    print("🚀 演示：带注意力的中期融合 (Middle Fusion with Attention)")
    print("="*80)

    # ========== 1. 配置模型 ==========
    print("\n[1/5] 配置模型...")
    config = ALIGNNConfig(
        name="alignn",
        alignn_layers=4,
        gcn_layers=4,
        hidden_features=256,

        # ✅ 启用带注意力的中期融合
        use_middle_fusion=True,
        use_middle_fusion_attention=True,  # 🔑 关键配置！
        middle_fusion_layers="2",  # 在第2层ALIGNN后注入文本
        middle_fusion_hidden_dim=128,
        middle_fusion_num_heads=4,
        middle_fusion_dropout=0.1,
        middle_fusion_attention_use_gate=True,  # 混合模式：注意力+门控
        middle_fusion_attention_use_projection=True,

        # 禁用其他融合（纯中期融合）
        use_fine_grained_attention=False,
        use_cross_modal_attention=True,

        # 后期融合配置
        late_fusion_type="adaptive",
        late_fusion_output_dim=64,

        output_features=1
    )

    model = ALIGNN(config)
    model.eval()

    print("✅ 模型配置完成")
    print(f"   - 中期融合注意力: 已启用")
    print(f"   - 融合层: {config.middle_fusion_layers}")
    print(f"   - 注意力头数: {config.middle_fusion_num_heads}")
    print(f"   - 混合模式（注意力+门控）: {config.middle_fusion_attention_use_gate}")

    # ========== 2. 准备示例数据 ==========
    print("\n[2/5] 准备示例数据...")

    # 示例：从JARVIS数据库加载一个晶体结构
    try:
        dft_3d = jdata('dft_3d')
        example = dft_3d[0]  # 获取第一个结构
        atoms = Atoms.from_dict(example['atoms'])
        text = example.get('description', "A crystalline material with high band gap")
    except:
        # 如果无法访问数据库，使用硬编码示例
        from jarvis.core.lattice import Lattice
        lattice = Lattice([[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]])
        atoms = Atoms(
            lattice_mat=lattice.matrix,
            coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
            elements=["Si", "O"],
            cartesian=False
        )
        text = "Silicon oxide semiconductor material"

    print(f"✅ 数据准备完成")
    print(f"   - 材料: {atoms.composition.reduced_formula}")
    print(f"   - 原子数: {atoms.num_atoms}")
    print(f"   - 文本: '{text}'")

    # ========== 3. 构建图数据 ==========
    print("\n[3/5] 构建图数据...")
    from jarvis.core.graphs import Graph

    # 构建晶体图
    graph = Graph.atom_dgl_multigraph(atoms, cutoff=8.0, compute_line_graph=True)
    g = graph['g']
    lg = graph['lg']

    # 批处理（batch_size=1）
    import dgl
    g_batch = dgl.batch([g])
    lg_batch = dgl.batch([lg])

    print(f"✅ 图数据构建完成")
    print(f"   - 节点数: {g.number_of_nodes()}")
    print(f"   - 边数: {g.number_of_edges()}")

    # ========== 4. 前向传播并提取注意力权重 ==========
    print("\n[4/5] 前向传播并提取注意力权重...")

    with torch.no_grad():
        output = model(
            (g_batch, lg_batch, [text]),
            return_attention=True  # 🔑 返回注意力权重
        )

    prediction = output['predictions']
    middle_fusion_attn = output.get('middle_fusion_attention_weights', {})

    print(f"✅ 前向传播完成")
    print(f"   - 预测值: {prediction.item():.4f}")
    print(f"   - 中期融合注意力层: {list(middle_fusion_attn.keys())}")

    # ========== 5. 可视化注意力权重 ==========
    print("\n[5/5] 可视化注意力权重...")

    if middle_fusion_attn:
        # 获取原子符号
        atom_symbols = [str(elem) for elem in atoms.elements]

        for layer_name, attn_weights in middle_fusion_attn.items():
            print(f"\n处理层: {layer_name}")
            print(f"   - 注意力权重形状: {attn_weights.shape}")

            # 分析对应关系
            correspondences = analyze_atom_text_correspondence(
                attn_weights, text, atom_symbols, top_k=3
            )

            # 可视化热力图
            save_path = f"middle_fusion_attention_{layer_name}.png"
            visualize_middle_fusion_attention(
                attn_weights, text, atom_symbols, layer_name, save_path
            )
    else:
        print("⚠️  未找到中期融合注意力权重")
        print("   请确保配置中启用了 use_middle_fusion_attention=True")

    print("\n" + "="*80)
    print("✅ 演示完成！")
    print("="*80)
    print("\n💡 关键要点：")
    print("   1. 中期融合注意力在ALIGNN中间层注入文本信息")
    print("   2. 保持了中期融合的高精度")
    print("   3. 增加了细粒度的可解释性（原子-文本对应关系）")
    print("   4. 可以通过热力图直观地看到每个原子关注哪些文本词汇")
    print("\n📊 输出文件：")
    print("   - middle_fusion_attention_layer_*.png (注意力热力图)")


if __name__ == "__main__":
    main()
