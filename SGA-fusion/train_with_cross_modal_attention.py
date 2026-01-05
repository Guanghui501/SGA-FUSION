#!/usr/bin/env python
"""
完整训练脚本 - 使用跨模态注意力的 CrysMMNet

这个脚本展示了如何训练集成跨模态注意力机制的 CrysMMNet 模型。
支持 JARVIS-DFT 和 Material Project 数据集。

用法示例:
    # JARVIS 数据集训练
    python train_with_cross_modal_attention.py \
        --dataset jarvis \
        --property formation_energy \
        --use_cross_modal True \
        --num_heads 4

    # Material Project 数据集训练
    python train_with_cross_modal_attention.py \
        --dataset mp \
        --property band_gap \
        --use_cross_modal True \
        --num_heads 8
"""

import os
import sys
import csv
import time
import json
import argparse
import numpy as np
import pickle as pkl
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from jarvis.core.atoms import Atoms
from jarvis.db.jsonutils import loadjson, dumpjson

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'crysmmnet-main/src'))

from data import get_train_val_loaders
from train import train_dgl
from config import TrainingConfig
from models.alignn import ALIGNN, ALIGNNConfig

from transformers import AutoTokenizer, AutoModel
from tokenizers.normalizers import BertNormalizer


# ==================== 辅助函数 ====================

def str2bool(v):
    """将字符串转换为布尔值（用于argparse）"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('布尔值应为 yes/no, true/false, t/f, y/n, 1/0')


# ==================== 配置参数 ====================

def get_parser():
    """构建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description='训练 CrysMMNet (带跨模态注意力)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 数据集参数
    parser.add_argument('--root_dir', type=str, default='../dataset/',
                        help='数据集根目录（相对于当前目录或绝对路径）')
    parser.add_argument('--dataset', type=str, default='jarvis',
                        choices=['jarvis', 'mp', 'class', 'toy'],
                        help='数据集名称: jarvis, mp, class (分类), toy')
    parser.add_argument('--property', type=str, default='formation_energy',
                        help='预测的性质 (回归: formation_energy, band_gap; 分类: syn, metal_oxide等)')

    # 预处理数据参数
    parser.add_argument('--use_preprocessed', type=str2bool, default=False,
                        help='是否使用预处理的图数据（大幅加快加载速度）')
    parser.add_argument('--preprocessed_dir', type=str, default='preprocessed_data',
                        help='预处理数据目录')

    # 数据划分参数
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='测试集比例')
    parser.add_argument('--n_train', type=int, default=None,
                        help='训练样本数（如果指定则覆盖train_ratio）')
    parser.add_argument('--n_val', type=int, default=None,
                        help='验证样本数')
    parser.add_argument('--n_test', type=int, default=None,
                        help='测试样本数')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='权重衰减')
    parser.add_argument('--warmup_steps', type=int, default=2000,
                        help='学习率warmup步数')

    # 模型参数
    parser.add_argument('--alignn_layers', type=int, default=4,
                        help='ALIGNN层数')
    parser.add_argument('--gcn_layers', type=int, default=4,
                        help='GCN层数')
    parser.add_argument('--hidden_features', type=int, default=256,
                        help='隐藏层特征维度')
    parser.add_argument('--graph_dropout', type=float, default=0.0,
                        help='ALIGNN/GCN层的dropout率（用于正则化）')

    # 跨模态注意力参数（晚期融合）
    parser.add_argument('--use_cross_modal', type=str2bool, default=True,
                        help='是否使用跨模态注意力（晚期融合）')
    parser.add_argument('--cross_modal_hidden_dim', type=int, default=256,
                        help='跨模态注意力隐藏层维度')
    parser.add_argument('--cross_modal_num_heads', type=int, default=4,
                        choices=[1, 2, 4, 8],
                        help='跨模态注意力头数')
    parser.add_argument('--cross_modal_dropout', type=float, default=0.1,
                        help='跨模态注意力dropout率')

    # 后期融合方式参数 ⭐ NEW!
    parser.add_argument('--late_fusion_type', type=str, default='concat',
                        choices=['concat', 'gated', 'bilinear', 'adaptive', 'tucker'],
                        help='后期融合方式: concat(拼接), gated(门控), bilinear(双线性), adaptive(自适应), tucker(Tucker分解)')
    parser.add_argument('--late_fusion_rank', type=int, default=16,
                        help='双线性/Tucker融合的秩（低秩分解参数，越大表达能力越强但参数越多）')
    parser.add_argument('--late_fusion_output_dim', type=int, default=64,
                        help='后期融合输出维度（对于非concat融合）')

    # 中期融合参数
    parser.add_argument('--use_middle_fusion', type=str2bool, default=False,
                        help='是否使用中期融合（在编码过程中注入文本信息）')
    parser.add_argument('--middle_fusion_layers', type=str, default='2',
                        help='中期融合注入的层索引（逗号分隔，如 "2" 或 "2,3"）')
    parser.add_argument('--middle_fusion_hidden_dim', type=int, default=128,
                        help='中期融合隐藏层维度')
    parser.add_argument('--middle_fusion_num_heads', type=int, default=2,
                        choices=[1, 2, 4],
                        help='中期融合注意力头数')
    parser.add_argument('--middle_fusion_dropout', type=float, default=0.1,
                        help='中期融合dropout率')
    parser.add_argument('--middle_fusion_use_gate_norm', type=str2bool, default=False,
                        help='中期融合是否使用Gate LayerNorm（用于特征尺度平衡）')
    parser.add_argument('--middle_fusion_use_learnable_scale', type=str2bool, default=False,
                        help='中期融合是否使用可学习缩放因子（自动学习最优文本特征权重）')
    parser.add_argument('--middle_fusion_initial_scale', type=float, default=1.0,
                        help='中期融合可学习缩放因子的初始值（建议基于诊断结果设置，如12.0）')

    # 细粒度注意力参数（原子-文本token级别）⭐ NEW!
    parser.add_argument('--use_fine_grained_attention', type=str2bool, default=False,
                        help='是否使用细粒度注意力（原子-文本token级别）')
    parser.add_argument('--fine_grained_hidden_dim', type=int, default=256,
                        help='细粒度注意力隐藏层维度')
    parser.add_argument('--fine_grained_num_heads', type=int, default=8,
                        choices=[1, 2, 4, 8],
                        help='细粒度注意力头数')
    parser.add_argument('--fine_grained_dropout', type=float, default=0.1,
                        help='细粒度注意力dropout率')
    parser.add_argument('--fine_grained_use_projection', type=str2bool, default=True,
                        help='细粒度注意力是否使用投影层')
    parser.add_argument('--mask_stopwords', type=int, default=0,
                        help='是否在训练时mask停用词 (0/1)')
    parser.add_argument('--remove_stopwords', type=int, default=0,
                        help='是否在BERT编码前删除停用词 (0/1)')
    parser.add_argument('--stopwords_dir', type=str, default='',
                        help='停用词目录路径 (默认: ./stopwords/en/)')

    # 对比学习参数
    parser.add_argument('--use_contrastive', type=str2bool, default=False,
                        help='是否使用对比学习损失')
    parser.add_argument('--contrastive_weight', type=float, default=0.1,
                        help='对比学习损失的权重（相对于主任务损失）')
    parser.add_argument('--contrastive_temperature', type=float, default=0.1,
                        help='对比学习的温度参数')

    # 融合策略参数 ⭐ NEW! (方案1和方案2)
    parser.add_argument('--cross_modal_attention_type', type=str, default='bidirectional',
                        choices=['bidirectional', 'unidirectional'],
                        help='跨模态注意力类型: bidirectional (双向), unidirectional (单向，文本→图)')
    parser.add_argument('--fusion_strategy', type=str, default='gated',
                        choices=['average', 'concat', 'gated'],
                        help='融合策略: average (平均), concat (拼接), gated (门控)')
    parser.add_argument('--gated_fusion_type', type=str, default='dual_gate',
                        choices=['single_gate', 'dual_gate', 'attention'],
                        help='门控融合类型: single_gate, dual_gate (推荐), attention')
    parser.add_argument('--gated_fusion_hidden_dim', type=int, default=128,
                        help='门控融合隐藏层维度')
    parser.add_argument('--gated_fusion_dropout', type=float, default=0.1,
                        help='门控融合dropout率')

    # Gated Cross-Attention参数 ⭐ NEW! (方案三 - 质量感知自适应融合)
    # 注意: 这些参数暂时被注释，因为 Gated Cross-Attention 功能尚未在 ALIGNNConfig 中实现
    # parser.add_argument('--use_gated_cross_attention', type=str2bool, default=False,
    #                     help='是否使用Gated Cross-Attention（质量感知自适应融合）')
    # parser.add_argument('--gated_attention_hidden_dim', type=int, default=256,
    #                     help='Gated Cross-Attention隐藏层维度')
    # parser.add_argument('--gated_attention_num_heads', type=int, default=4,
    #                     choices=[1, 2, 4, 8],
    #                     help='Gated Cross-Attention注意力头数')
    # parser.add_argument('--gated_attention_dropout', type=float, default=0.1,
    #                     help='Gated Cross-Attention dropout率')
    # parser.add_argument('--gated_quality_hidden_dim', type=int, default=128,
    #                     help='质量门控网络隐藏层维度')

    # 其他参数
    parser.add_argument('--output_dir', type=str, default='./output/',
                        help='输出目录')
    parser.add_argument('--config_file', type=str, default=None,
                        help='配置文件路径（可选）')
    parser.add_argument('--resume', type=int, default=0,
                        help='是否从checkpoint恢复训练 (0/1)')
    parser.add_argument('--random_seed', type=int, default=123,
                        help='随机种子')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='数据加载workers数量')
    parser.add_argument('--early_stopping_patience', type=int, default=None,
                        help='Early stopping耐心值（无改善的epoch数）')

    # 分类任务参数
    parser.add_argument('--classification', type=int, default=0,
                        help='是否为分类任务 (0/1)')
    parser.add_argument('--classification_threshold', type=float, default=0.5,
                        help='分类阈值（用于启用分类模式）')

    return parser


# ==================== 数据集路径配置 ====================

def get_dataset_paths(root_dir, dataset, property_name):
    """根据数据集和性质获取数据路径"""

    if dataset.lower() == 'jarvis':
        # JARVIS-DFT 数据集
        property_map = {
            'formation_energy': 'formation_energy_peratom',
            'fe': 'formation_energy_peratom',
            'total_energy': 'optb88vdw_total_energy',
            'opt_bandgap': 'optb88vdw_bandgap',
            'mbj_bandgap': 'mbj_bandgap',
            'bulk_modulus': 'bulk_modulus_kv',
            'bulk_modulus_kv': 'bulk_modulus_kv',
            'shear_modulus': 'shear_modulus_gv',
            'shear_modulus_gv': 'shear_modulus_gv',
        }

        prop_folder = property_map.get(property_name, property_name)
        cif_dir = os.path.join(root_dir, f'jarvis/{prop_folder}/cif/')
        id_prop_file = os.path.join(root_dir, f'jarvis/{prop_folder}/description.csv')

    elif dataset.lower() == 'mp':
        # Material Project 数据集
        if property_name in ['formation_energy', 'band_gap']:
            cif_dir = os.path.join(root_dir, 'mp_2018_new/')
            id_prop_file = os.path.join(root_dir, 'mp_2018_new/mat_text.csv')
        elif property_name in ['bulk', 'shear', 'bulk_modulus', 'shear_modulus']:
            cif_dir = os.path.join(root_dir, 'mp_2018_small/cif/')
            id_prop_file = os.path.join(root_dir, 'mp_2018_small/description.csv')
        else:
            raise ValueError(f"Unsupported property for MP dataset: {property_name}")

    elif dataset.lower() == 'class':
        # 分类数据集（类似jarvis结构）
        # 例如：class/syn, class/metal_oxide, 等
        cif_dir = os.path.join(root_dir, f'class/{property_name}/cif/')
        id_prop_file = os.path.join(root_dir, f'class/{property_name}/description.csv')

    elif dataset.lower() == 'toy':
        # 玩具数据集（用于测试）
        cif_dir = os.path.join(root_dir, 'toy/cif/')
        id_prop_file = os.path.join(root_dir, 'toy/description.csv')

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return cif_dir, id_prop_file


# ==================== 数据加载 ====================

def load_preprocessed_dataset(preprocessed_dir, dataset, property_name):
    """加载预处理的数据集

    Args:
        preprocessed_dir: 预处理数据目录
        dataset: 数据集名称
        property_name: 属性名称

    Returns:
        train_data, val_data, test_data: 三个列表，每个元素是 (graph, line_graph, text, target)
    """
    import pickle

    print(f"\n{'='*60}")
    print(f"加载预处理数据集: {dataset} - {property_name}")
    print(f"预处理目录: {preprocessed_dir}")
    print(f"{'='*60}\n")

    # 加载三个数据集（使用dataset名称，与preprocess_graphs.py一致）
    splits = {}
    for split_name in ['train', 'val', 'test']:
        pkl_file = os.path.join(
            preprocessed_dir,
            f"{dataset.lower()}_{property_name}_{split_name}.pkl"
        )

        if not os.path.exists(pkl_file):
            raise FileNotFoundError(
                f"找不到预处理文件: {pkl_file}\n"
                f"请先运行: python preprocess_graphs.py --dataset {dataset} --property {property_name}"
            )

        print(f"加载 {split_name} 集: {pkl_file}")
        with open(pkl_file, 'rb') as f:
            samples = pickle.load(f)

        # 转换为训练所需的字典格式（与load_dataset保持一致）
        data = []
        for sample in samples:
            # 从图中恢复 atoms 信息
            # 注意：我们直接使用预构建的图，不需要 atoms.to_dict()
            info = {
                "graph": sample['graph'][0],      # atom graph (预构建)
                "line_graph": sample['line_graph'],  # line graph (预构建)
                "jid": sample['id'],
                "text": sample['text'],          # 已规范化的文本
                "target": sample['target']
            }
            data.append(info)

        splits[split_name] = data
        print(f"  ✓ 加载了 {len(data)} 个样本")

    print(f"\n成功加载预处理数据!")
    print(f"  训练集: {len(splits['train'])} 样本")
    print(f"  验证集: {len(splits['val'])} 样本")
    print(f"  测试集: {len(splits['test'])} 样本\n")

    return splits['train'], splits['val'], splits['test']


def load_dataset(cif_dir, id_prop_file, dataset, property_name):
    """加载数据集"""
    print(f"\n{'='*60}")
    print(f"加载数据集: {dataset} - {property_name}")
    print(f"CIF目录: {cif_dir}")
    print(f"描述文件: {id_prop_file}")
    print(f"{'='*60}\n")

    # 读取CSV文件
    with open(id_prop_file, 'r') as f:
        reader = csv.reader(f)
        headings = next(reader)
        data = [row for row in reader]

    print(f"总样本数: {len(data)}")

    # 文本归一化器
    norm = BertNormalizer(lowercase=False, strip_accents=True,
                         clean_text=True, handle_chinese_chars=True)

    # 加载词汇映射 - 智能路径查找
    # 尝试多个可能的路径
    possible_paths = [
        'vocab_mappings.txt',  # 当前目录
        './vocab_mappings.txt',
        os.path.join(os.path.dirname(__file__), 'vocab_mappings.txt'),  # 脚本所在目录
        os.path.join(os.path.dirname(__file__), 'crysmmnet-main/src/vocab_mappings.txt'),  # 从根目录
        '../vocab_mappings.txt',  # 上级目录
        '../../vocab_mappings.txt',
    ]

    vocab_file = None
    for path in possible_paths:
        if os.path.exists(path):
            vocab_file = path
            break

    if vocab_file is None:
        raise FileNotFoundError(
            "无法找到 vocab_mappings.txt 文件。请确保：\n"
            "1. 文件存在于 crysmmnet-main/src/ 目录\n"
            "2. 当前工作目录正确\n"
            f"尝试过的路径: {possible_paths}"
        )

    print(f"使用词汇映射文件: {vocab_file}")
    with open(vocab_file, 'r') as f:
        mappings = f.read().strip().split('\n')
    mappings = {m[0]: m[2:] for m in mappings}

    def normalize(text):
        text = [norm.normalize_str(s) for s in text.split('\n')]
        out = []
        for s in text:
            norm_s = ''
            for c in s:
                norm_s += mappings.get(c, ' ')
            out.append(norm_s)
        return '\n'.join(out)

    # 构建数据集
    dataset_array = []
    skipped = 0

    for j in tqdm(range(len(data)), desc="加载数据"):
        try:
            if dataset.lower() == 'mp':
                if property_name == 'formation_energy':
                    id, composition, target, _, crys_desc_full, _ = data[j]
                elif property_name == 'band_gap':
                    id, composition, _, target, crys_desc_full, _ = data[j]
                elif property_name == 'shear':
                    id, composition, target, _, crys_desc_full, _ = data[j]
                elif property_name in ['bulk', 'bulk_modulus']:
                    id, composition, _, target, crys_desc_full, _ = data[j]
            elif dataset.lower() == 'jarvis':
                id, composition, target, crys_desc_full, _ = data[j]
            elif dataset.lower() == 'class':
                # 分类数据集格式：id, target, description
                id, target, crys_desc_full = data[j]
                composition = ''  # 分类任务不需要composition
            elif dataset.lower() == 'toy':
                id, composition, target, crys_desc_full, _ = data[j]

            # 读取CIF文件
            file_path = os.path.join(cif_dir, f'{id}.cif')
            if not os.path.exists(file_path):
                skipped += 1
                continue

            atoms = Atoms.from_cif(file_path)

            # 构建样本
            info = {
                "atoms": atoms.to_dict(),
                "jid": id,
                "text": crys_desc_full,
                "target": float(target)
            }

            # MP数据集的特殊处理
            if dataset.lower() == 'mp' and property_name in ['shear', 'bulk', 'bulk_modulus', 'shear_modulus']:
                info["target"] = np.log10(float(target))

            dataset_array.append(info)

        except Exception as e:
            skipped += 1
            if skipped <= 5:  # 只显示前5个错误
                print(f"跳过样本 {id}: {e}")

    print(f"\n成功加载: {len(dataset_array)} 样本")
    print(f"跳过: {skipped} 样本\n")

    return dataset_array


# ==================== 配置生成 ====================

def create_config(args):
    """根据命令行参数创建训练配置"""

    # 导入 ALIGNNConfig
    from models.alignn import ALIGNNConfig

    # 数据集名称映射：用户友好名称 -> TrainingConfig 期望的名称
    dataset_mapping = {
        'jarvis': 'user_data',
        'mp': 'user_data',
        'class': 'user_data',
        'toy': 'user_data',
        'user_data': 'user_data',
    }

    # 获取实际的数据集名称
    actual_dataset = dataset_mapping.get(args.dataset.lower(), 'user_data')

    # 创建模型配置对象
    model_config = ALIGNNConfig(
        name="alignn",
        alignn_layers=args.alignn_layers,
        gcn_layers=args.gcn_layers,
        atom_input_features=92,
        edge_input_features=80,
        triplet_input_features=40,
        embedding_features=64,
        hidden_features=args.hidden_features,
        output_features=1,
        # Graph层 dropout（正则化）
        graph_dropout=args.graph_dropout,
        # 跨模态注意力配置（晚期融合）
        use_cross_modal_attention=args.use_cross_modal,
        cross_modal_hidden_dim=args.cross_modal_hidden_dim,
        cross_modal_num_heads=args.cross_modal_num_heads,
        cross_modal_dropout=args.cross_modal_dropout,
        # 后期融合方式配置 ⭐ NEW!
        late_fusion_type=args.late_fusion_type,
        late_fusion_rank=args.late_fusion_rank,
        late_fusion_output_dim=args.late_fusion_output_dim,
        # 中期融合配置
        use_middle_fusion=args.use_middle_fusion,
        middle_fusion_layers=args.middle_fusion_layers,
        middle_fusion_hidden_dim=args.middle_fusion_hidden_dim,
        middle_fusion_num_heads=args.middle_fusion_num_heads,
        middle_fusion_dropout=args.middle_fusion_dropout,
        middle_fusion_use_gate_norm=args.middle_fusion_use_gate_norm,
        middle_fusion_use_learnable_scale=args.middle_fusion_use_learnable_scale,
        middle_fusion_initial_scale=args.middle_fusion_initial_scale,
        # 对比学习配置
        use_contrastive_loss=args.use_contrastive,
        contrastive_loss_weight=args.contrastive_weight,
        contrastive_temperature=args.contrastive_temperature,
        # 细粒度注意力配置（原子-文本token级别）⭐ NEW!
        use_fine_grained_attention=args.use_fine_grained_attention,
        fine_grained_hidden_dim=args.fine_grained_hidden_dim,
        fine_grained_num_heads=args.fine_grained_num_heads,
        fine_grained_dropout=args.fine_grained_dropout,
        fine_grained_use_projection=args.fine_grained_use_projection,
        mask_stopwords=bool(args.mask_stopwords),
        remove_stopwords=bool(args.remove_stopwords),
        stopwords_dir=args.stopwords_dir,
        link="identity",
        zero_inflated=False,
        classification=False
    )

    config = {
        "version": "cross_modal_attention_v1",
        "dataset": actual_dataset,  # 使用映射后的名称
        "target": "target",
        "atom_features": "cgcnn",
        "neighbor_strategy": "k-nearest",
        "id_tag": "jid",
        "random_seed": args.random_seed,
        "classification_threshold": args.classification_threshold if args.classification else None,

        # 数据划分
        "n_train": args.n_train,
        "n_val": args.n_val,
        "n_test": args.n_test,
        "train_ratio": args.train_ratio if args.n_train is None else None,
        "val_ratio": args.val_ratio if args.n_val is None else None,
        "test_ratio": args.test_ratio if args.n_test is None else None,

        "target_multiplication_factor": None,

        # 训练参数
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "learning_rate": args.learning_rate,
        "filename": f"{args.dataset}_{args.property}",
        "warmup_steps": args.warmup_steps,
        "criterion": "bce" if args.classification else "mse",
        "optimizer": "adamw",
        "scheduler": "onecycle",

        "pin_memory": False,
        "save_dataloader": False,
        "write_checkpoint": True,
        "write_predictions": True,
        "store_outputs": True,
        "progress": True,
        "log_tensorboard": False,
        "standard_scalar_and_pca": False,
        "use_canonize": True,
        "num_workers": args.num_workers,
        "cutoff": 8.0,
        "max_neighbors": 12,
        "keep_data_order": False,
        "distributed": False,
        "n_early_stopping": args.early_stopping_patience,
        "output_dir": args.output_dir,

        # 模型配置对象（而不是字典）
        "model": model_config
    }

    return config


# ==================== 主训练函数 ====================

def main():
    """主训练流程"""

    # 解析参数
    parser = get_parser()
    args = parser.parse_args()

    # 打印配置
    print("\n" + "="*80)
    print("CrysMMNet 训练 - 跨模态注意力机制")
    print("="*80)
    print(f"\n数据集配置:")
    print(f"  数据集: {args.dataset}")
    print(f"  性质: {args.property}")
    print(f"  根目录: {args.root_dir}")

    print(f"\n训练配置:")
    print(f"  批次大小: {args.batch_size}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  学习率: {args.learning_rate}")
    print(f"  权重衰减: {args.weight_decay}")
    print(f"  任务类型: {'分类' if args.classification else '回归'}")
    if args.classification:
        print(f"  分类阈值: {args.classification_threshold}")
        print(f"  损失函数: BCE (Binary Cross Entropy)")

    print(f"\n模型配置:")
    print(f"  ALIGNN层数: {args.alignn_layers}")
    print(f"  GCN层数: {args.gcn_layers}")
    print(f"  隐藏层维度: {args.hidden_features}")
    print(f"  Graph层Dropout: {args.graph_dropout}")

    print(f"\n跨模态注意力配置（晚期融合）:")
    print(f"  启用: {args.use_cross_modal}")
    if args.use_cross_modal:
        print(f"  隐藏维度: {args.cross_modal_hidden_dim}")
        print(f"  注意力头数: {args.cross_modal_num_heads}")
        print(f"  Dropout率: {args.cross_modal_dropout}")

    print(f"\n后期融合方式配置:")
    print(f"  融合类型: {args.late_fusion_type}")
    fusion_desc = {
        'concat': '简单拼接（基线）',
        'gated': '门控融合（自适应权重）',
        'bilinear': '双线性池化（二阶交互）',
        'adaptive': '自适应融合（多策略组合）',
        'tucker': 'Tucker分解（高阶张量）'
    }
    print(f"  说明: {fusion_desc.get(args.late_fusion_type, '未知')}")
    if args.late_fusion_type in ['bilinear', 'tucker']:
        print(f"  低秩分解Rank: {args.late_fusion_rank}")
    if args.late_fusion_type != 'concat':
        print(f"  融合输出维度: {args.late_fusion_output_dim}")

    print(f"\n中期融合配置:")
    print(f"  启用: {args.use_middle_fusion}")
    if args.use_middle_fusion:
        print(f"  融合层: {args.middle_fusion_layers}")
        print(f"  隐藏维度: {args.middle_fusion_hidden_dim}")
        print(f"  注意力头数: {args.middle_fusion_num_heads}")
        print(f"  Dropout率: {args.middle_fusion_dropout}")
        print(f"  Gate LayerNorm: {args.middle_fusion_use_gate_norm}")
        print(f"  可学习缩放: {args.middle_fusion_use_learnable_scale}")
        if args.middle_fusion_use_learnable_scale:
            print(f"  初始缩放值: {args.middle_fusion_initial_scale}")

    print(f"\n细粒度注意力配置（原子-文本token级别）:")
    print(f"  启用: {args.use_fine_grained_attention}")
    if args.use_fine_grained_attention:
        print(f"  隐藏维度: {args.fine_grained_hidden_dim}")
        print(f"  注意力头数: {args.fine_grained_num_heads}")
        print(f"  Dropout率: {args.fine_grained_dropout}")
        print(f"  使用投影: {args.fine_grained_use_projection}")
        print(f"  停用词Masking: {bool(args.mask_stopwords)}")
        print(f"  停用词删除(BERT编码前): {bool(args.remove_stopwords)}")
        if args.mask_stopwords or args.remove_stopwords:
            print(f"  停用词目录: {args.stopwords_dir if args.stopwords_dir else '默认 (./stopwords/en/)'}")

    print(f"\n对比学习配置:")
    print(f"  启用: {args.use_contrastive}")
    if args.use_contrastive:
        print(f"  损失权重: {args.contrastive_weight}")
        print(f"  温度参数: {args.contrastive_temperature}")

    print(f"\n输出目录: {args.output_dir}")
    print("="*80 + "\n")

    # 创建输出目录
    output_dir = os.path.join(args.output_dir, f"{args.property}/")
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据集
    if args.use_preprocessed:
        # 使用预处理数据（快速加载）
        print(f"\n⚡ 使用预处理数据加载模式")
        print(f"预处理目录: {args.preprocessed_dir}\n")

        train_data, val_data, test_data = load_preprocessed_dataset(
            args.preprocessed_dir,
            args.dataset,
            args.property
        )
        dataset_array = (train_data, val_data, test_data)

    else:
        # 从原始 CIF 文件加载（慢速）
        print(f"\n🐢 从原始 CIF 文件加载模式（较慢）")
        print(f"提示: 使用 --use_preprocessed True 可以大幅加快加载速度\n")

        # 获取数据路径
        cif_dir, id_prop_file = get_dataset_paths(args.root_dir, args.dataset, args.property)

        # 检查路径是否存在
        if not os.path.exists(cif_dir):
            print(f"\n❌ 错误: CIF目录不存在: {cif_dir}")
            print(f"\n提示:")
            print(f"  1. 检查 --root_dir 参数是否正确")
            print(f"  2. 当前工作目录: {os.getcwd()}")
            print(f"  3. 如果在 src 目录下运行，使用: --root_dir ../dataset/")
            print(f"  4. 如果在项目根目录运行，使用: --root_dir ./crysmmnet-main/dataset/")
            raise FileNotFoundError(f"CIF目录不存在: {cif_dir}")
        if not os.path.exists(id_prop_file):
            print(f"\n❌ 错误: 描述文件不存在: {id_prop_file}")
            print(f"\n提示: 请确保数据集已正确下载并解压")
            raise FileNotFoundError(f"描述文件不存在: {id_prop_file}")

        # 加载数据集
        dataset_array = load_dataset(cif_dir, id_prop_file, args.dataset, args.property)

    # 创建配置
    config_dict = create_config(args)
    config_dict['output_dir'] = output_dir

    # 提取模型配置（ALIGNNConfig）用于保存到checkpoint
    model_config = config_dict['model']

    # 保存配置（将 ALIGNNConfig 对象转换为字典以便 JSON 序列化）
    config_file = os.path.join(output_dir, 'config.json')
    config_dict_serializable = config_dict.copy()

    # 转换 model 配置对象为字典
    if hasattr(config_dict['model'], 'dict'):
        # Pydantic v1
        config_dict_serializable['model'] = config_dict['model'].dict()
    elif hasattr(config_dict['model'], 'model_dump'):
        # Pydantic v2
        config_dict_serializable['model'] = config_dict['model'].model_dump()
    else:
        # 尝试使用 __dict__
        config_dict_serializable['model'] = config_dict['model'].__dict__

    with open(config_file, 'w') as f:
        json.dump(config_dict_serializable, f, indent=4)
    print(f"配置已保存到: {config_file}\n")

    # 转换为TrainingConfig对象
    try:
        config = TrainingConfig(**config_dict)
    except Exception as e:
        print(f"配置验证失败: {e}")
        return

    # 创建数据加载器
    print("创建数据加载器...")
    train_loader, val_loader, test_loader, prepare_batch = get_train_val_loaders(
        dataset_array=dataset_array,
        target=config.target,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        batch_size=config.batch_size,
        atom_features=config.atom_features,
        neighbor_strategy=config.neighbor_strategy,
        id_tag=config.id_tag,
        pin_memory=config.pin_memory,
        workers=config.num_workers,
        save_dataloader=config.save_dataloader,
        use_canonize=config.use_canonize,
        filename=config.filename,
        cutoff=config.cutoff,
        max_neighbors=config.max_neighbors,
        target_multiplication_factor=config.target_multiplication_factor,
        standard_scalar_and_pca=config.standard_scalar_and_pca,
        keep_data_order=config.keep_data_order,
        output_dir=config.output_dir
    )

    print(f"\n数据集大小:")
    print(f"  训练集: {len(train_loader.dataset)}")
    print(f"  验证集: {len(val_loader.dataset)}")
    print(f"  测试集: {len(test_loader.dataset)}")
    print()

    # 开始训练
    print("="*80)
    print("开始训练...")
    print("="*80 + "\n")

    start_time = time.time()

    train_dgl(
        config=config,
        train_val_test_loaders=[train_loader, val_loader, test_loader, prepare_batch],
        resume=args.resume,
        model_config=model_config  # Pass model config for checkpoint saving
    )

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("\n" + "="*80)
    print(f"训练完成！")
    print(f"总用时: {elapsed_time/3600:.2f} 小时 ({elapsed_time/60:.2f} 分钟)")
    print(f"结果保存在: {output_dir}")
    print("="*80 + "\n")


# ==================== 快捷训练函数 ====================

def train_jarvis_formation_energy():
    """JARVIS 形成能训练的快捷函数"""
    sys.argv = [
        'train_with_cross_modal_attention.py',
        '--dataset', 'jarvis',
        '--property', 'formation_energy',
        '--use_cross_modal', 'True',
        '--cross_modal_num_heads', '4',
        '--epochs', '1000',
        '--batch_size', '64'
    ]
    main()


def train_mp_bandgap():
    """Material Project 带隙训练的快捷函数"""
    sys.argv = [
        'train_with_cross_modal_attention.py',
        '--dataset', 'mp',
        '--property', 'band_gap',
        '--use_cross_modal', 'True',
        '--cross_modal_num_heads', '8',
        '--n_train', '60000',
        '--n_val', '5000',
        '--n_test', '4132',
        '--epochs', '1000',
        '--batch_size', '64'
    ]
    main()


def train_without_cross_modal():
    """不使用跨模态注意力训练（对比实验）"""
    sys.argv = [
        'train_with_cross_modal_attention.py',
        '--dataset', 'jarvis',
        '--property', 'formation_energy',
        '--use_cross_modal', 'False',  # 禁用跨模态注意力
        '--epochs', '1000',
        '--batch_size', '64'
    ]
    main()


# ==================== 入口 ====================

if __name__ == "__main__":
    main()
