"""Atomistic LIne Graph Neural Network.

A prototype crystal line graph network dgl implementation.
"""
from typing import Tuple, Union

import dgl
import dgl.function as fn
import numpy as np
import torch
from dgl.nn import AvgPooling
# from dgl.nn.functional import edge_softmax
from pydantic.typing import Literal
from torch import nn
from torch.nn import functional as F
from models.utils import RBFExpansion
from utils import BaseSettings

from transformers import AutoTokenizer
from transformers import AutoModel
from tokenizers.normalizers import BertNormalizer

"""**VoCab Mapping and Normalizer**"""

f = open('vocab_mappings.txt', 'r')
mappings = f.read().strip().split('\n')
f.close()

mappings = {m[0]: m[2:] for m in mappings}

norm = BertNormalizer(lowercase=False, strip_accents=True, clean_text=True, handle_chinese_chars=True)

def normalize(text):
    text = [norm.normalize_str(s) for s in text.split('\n')]
    out = []
    for s in text:
        norm_s = ''
        for c in s:
            norm_s += mappings.get(c, ' ')
        out.append(norm_s)
    return '\n'.join(out)

"""**Custom Dataset**"""

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tokenizer = AutoTokenizer.from_pretrained('m3rg-iitd/matscibert',model_max_length=512)
text_model = AutoModel.from_pretrained('m3rg-iitd/matscibert')
text_model.to(device)


class ProjectionHead(nn.Module):
    def __init__(self,embedding_dim,projection_dim=64,dropout=0.1):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class ContrastiveLoss(nn.Module):
    """Contrastive loss for aligning graph and text representations.

    Implements InfoNCE loss to encourage corresponding graph-text pairs
    to have similar representations while pushing non-corresponding pairs apart.
    """

    def __init__(self, temperature=0.1):
        """Initialize contrastive loss.

        Args:
            temperature: Temperature parameter for scaling similarities.
                        Lower values make the distribution more peaked.
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, graph_features, text_features):
        """Compute bidirectional contrastive loss.

        Args:
            graph_features: Graph representations [batch_size, feature_dim]
            text_features: Text representations [batch_size, feature_dim]

        Returns:
            loss: Contrastive loss value
        """
        batch_size = graph_features.size(0)

        # L2 normalize features for cosine similarity
        graph_features = F.normalize(graph_features, dim=1)
        text_features = F.normalize(text_features, dim=1)

        # Compute similarity matrix [batch_size, batch_size]
        # similarity[i,j] = cosine similarity between graph_i and text_j
        similarity_matrix = torch.matmul(graph_features, text_features.T) / self.temperature

        # Labels: diagonal elements are positive pairs
        labels = torch.arange(batch_size, device=graph_features.device)

        # Graph-to-Text loss: for each graph, find its corresponding text
        loss_g2t = F.cross_entropy(similarity_matrix, labels)

        # Text-to-Graph loss: for each text, find its corresponding graph
        loss_t2g = F.cross_entropy(similarity_matrix.T, labels)

        # Average bidirectional loss
        loss = (loss_g2t + loss_t2g) / 2.0

        return loss


class MiddleFusionModule(nn.Module):
    """Middle fusion module for injecting text information into graph encoding.

    This module performs cross-modal fusion during the intermediate layers of
    graph encoding, allowing text features to modulate node representations.

    Uses a simple gated fusion mechanism that works with DGL batched graphs.
    """

    def __init__(self, node_dim=64, text_dim=64, hidden_dim=128, num_heads=2, dropout=0.1,
                 use_gate_norm=False, use_learnable_scale=False, initial_scale=1.0):
        """Initialize middle fusion module.

        Args:
            node_dim: Dimension of graph node features
            text_dim: Dimension of text features
            hidden_dim: Hidden dimension for fusion
            num_heads: Number of attention heads (for future compatibility)
            dropout: Dropout rate
            use_gate_norm: Whether to use LayerNorm before gate (for feature scale balance)
            use_learnable_scale: Whether to use learnable scaling factor for text features
            initial_scale: Initial value for learnable scaling factor (default: 1.0)
        """
        super().__init__()
        self.node_dim = node_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.use_gate_norm = use_gate_norm
        self.use_learnable_scale = use_learnable_scale

        # Text transformation
        self.text_transform = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_dim)
        )

        # === 新增：可学习的缩放因子 ===
        if use_learnable_scale:
            self.text_scale = nn.Parameter(torch.tensor(initial_scale, dtype=torch.float32))
            print(f"✅ 启用可学习缩放因子，初始值: {initial_scale:.2f}")
        else:
            self.register_buffer('text_scale', torch.tensor(1.0, dtype=torch.float32))

        # === 新增：Gate 输入归一化（用于特征尺度平衡）===
        if use_gate_norm:
            self.gate_norm = nn.LayerNorm(node_dim * 2)
            print(f"✅ 启用 Gate LayerNorm")

        # Gate mechanism to control text influence
        self.gate = nn.Sequential(
            nn.Linear(node_dim + node_dim, node_dim),
            nn.Sigmoid()
        )

        self.layer_norm = nn.LayerNorm(node_dim)
        self.dropout = nn.Dropout(dropout)

        # 初始化存储变量
        self.stored_alphas = None

    def forward(self, node_feat, text_feat, batch_num_nodes=None):
        """Apply middle fusion using gated mechanism.

        Args:
            node_feat: Node features [total_nodes, node_dim] (for batched graphs)
                      or [batch_size, node_dim] (for pooled features)
            text_feat: Text features [batch_size, text_dim]
            batch_num_nodes: List of number of nodes in each graph (optional)

        Returns:
            Enhanced node features with same shape as input
        """
        # 调试输出
        #print(f"\n  🔍 MiddleFusionModule.forward 调试:")
        #print(f"     node_feat.shape: {node_feat.shape}")
        #print(f"     text_feat.shape: {text_feat.shape}")
        #print(f"     batch_num_nodes: {batch_num_nodes}")

        batch_size = text_feat.size(0)
        num_nodes = node_feat.size(0)

        # Transform text features
        text_transformed = self.text_transform(text_feat)  # [batch_size, node_dim]

        # === 应用可学习缩放 ===
        text_transformed = text_transformed * self.text_scale

        #print(f"     text_transformed.shape: {text_transformed.shape}")

        # Case 1: Batched graphs (total_nodes != batch_size)
        if num_nodes != batch_size:
            # Broadcast text features to all nodes
            # Simple approach: repeat text features proportionally to nodes per graph
            if batch_num_nodes is not None:
                # Use provided batch information
                text_expanded = []
                for i, num in enumerate(batch_num_nodes):
                    text_expanded.append(text_transformed[i].unsqueeze(0).repeat(num, 1))
                text_broadcasted = torch.cat(text_expanded, dim=0)  # [total_nodes, node_dim]
            else:
                # Fallback: use average pooling of text and broadcast
                text_pooled = text_transformed.mean(dim=0, keepdim=True)  # [1, node_dim]
                text_broadcasted = text_pooled.repeat(num_nodes, 1)  # [total_nodes, node_dim]

        # Case 2: Already pooled features (one per graph)
        else:
            text_broadcasted = text_transformed  # [batch_size, node_dim]
        
        #print(f"     text_broadcasted.shape: {text_broadcasted.shape}")
        # Gated fusion
        gate_input = torch.cat([node_feat, text_broadcasted], dim=-1)  # [*, node_dim*2]

        # === 应用 Gate LayerNorm（用于特征尺度平衡）===
        if self.use_gate_norm:
            gate_input = self.gate_norm(gate_input)

        #print(f"     gate_input.shape: {gate_input.shape}")
        gate_values = self.gate(gate_input)  # [*, node_dim]
        
        # [关键修改] 存储 gate_values (alpha) 供外部提取分析
        # 使用 .detach() 避免占用梯度计算图显存
        self.stored_alphas = gate_values.mean(dim=1).detach().cpu()

        # Apply gating and residual connection
        enhanced = node_feat + gate_values * text_broadcasted
        enhanced = self.layer_norm(enhanced)
        enhanced = self.dropout(enhanced)

        return enhanced


class MiddleFusionWithAttention(nn.Module):
    """Enhanced middle fusion module with attention mechanism for interpretability.

    This module combines the efficiency of middle fusion with fine-grained
    atom-to-text attention weights for interpretability. It allows you to:
    1. Maintain high accuracy of middle fusion
    2. Visualize which text tokens each atom attends to
    3. Extract attention weights for explainability analysis

    Key differences from original MiddleFusionModule:
    - Uses multi-head attention instead of simple broadcasting
    - Computes atom-to-text attention weights
    - Supports both attention-only and hybrid (attention + gate) modes
    """

    def __init__(self, node_dim=64, text_token_dim=768, hidden_dim=128,
                 num_heads=4, dropout=0.1, use_gate=True, use_projection=True):
        """Initialize middle fusion with attention.

        Args:
            node_dim: Dimension of graph node features
            text_token_dim: Dimension of text token features (768 for BERT)
            hidden_dim: Hidden dimension for attention computation
            num_heads: Number of attention heads (default: 4 for efficiency)
            dropout: Dropout rate
            use_gate: Whether to use additional gating mechanism (hybrid mode)
            use_projection: Whether to project token features to hidden_dim
        """
        super().__init__()
        self.node_dim = node_dim
        self.text_token_dim = text_token_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_gate = use_gate
        self.use_projection = use_projection

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Input projections
        if use_projection:
            self.node_proj_in = nn.Linear(node_dim, hidden_dim)
            self.token_proj_in = nn.Linear(text_token_dim, hidden_dim)

        # Atom-to-Token attention (atoms query text tokens)
        self.a2t_query = nn.Linear(hidden_dim if use_projection else node_dim, hidden_dim)
        self.a2t_key = nn.Linear(hidden_dim if use_projection else text_token_dim, hidden_dim)
        self.a2t_value = nn.Linear(hidden_dim if use_projection else text_token_dim, hidden_dim)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, node_dim)

        # Optional gating mechanism for hybrid mode
        if use_gate:
            self.gate = nn.Sequential(
                nn.Linear(node_dim * 2, node_dim),
                nn.Sigmoid()
            )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(node_dim)
        self.scale = self.head_dim ** -0.5

        # Store attention weights for visualization
        self.last_attention_weights = None

    def split_heads(self, x):
        """Split the last dimension into (num_heads, head_dim).

        Args:
            x: [batch_size, seq_len, hidden_dim]
        Returns:
            [batch_size, num_heads, seq_len, head_dim]
        """
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # (batch, heads, seq, head_dim)

    def forward(self, node_feat, text_tokens, batch_num_nodes=None,
                token_mask=None, return_attention=False):
        """Apply middle fusion with attention mechanism.

        Args:
            node_feat: Node features [total_nodes, node_dim] (for batched graphs)
                      or [batch_size, node_dim] (for pooled features)
            text_tokens: Text token features [batch_size, seq_len, text_token_dim]
            batch_num_nodes: List of number of nodes in each graph (required for batched graphs)
            token_mask: Optional mask for padded tokens [batch_size, seq_len]
            return_attention: Whether to return attention weights

        Returns:
            enhanced_nodes: Enhanced node features with same shape as input
            attention_weights: (optional) Attention weights [batch, num_heads, num_atoms, seq_len]
        """
        batch_size = text_tokens.size(0)
        seq_len = text_tokens.size(1)
        num_nodes = node_feat.size(0)

        # Store original features for residual connection
        node_feat_orig = node_feat

        # === Step 1: Convert DGL batched format to standard batch format ===
        if num_nodes != batch_size:
            # Batched graphs: need to convert to [batch_size, max_atoms, node_dim]
            if batch_num_nodes is None:
                raise ValueError("batch_num_nodes is required for batched graphs")

            max_atoms = max(batch_num_nodes)
            node_feat_batched = torch.zeros(
                batch_size, max_atoms, self.node_dim,
                dtype=node_feat.dtype, device=node_feat.device
            )

            # Create node mask
            node_mask = torch.zeros(batch_size, max_atoms, dtype=torch.bool, device=node_feat.device)

            # Fill in actual node features
            start_idx = 0
            for i, num in enumerate(batch_num_nodes):
                node_feat_batched[i, :num] = node_feat[start_idx:start_idx + num]
                node_mask[i, :num] = True
                start_idx += num

            num_atoms = max_atoms
        else:
            # Already pooled: [batch_size, node_dim] -> [batch_size, 1, node_dim]
            node_feat_batched = node_feat.unsqueeze(1)
            node_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=node_feat.device)
            num_atoms = 1

        # === Step 2: Optional input projection ===
        if self.use_projection:
            node_feat_projected = self.node_proj_in(node_feat_batched)  # [batch, num_atoms, hidden]
            token_feat_projected = self.token_proj_in(text_tokens)      # [batch, seq_len, hidden]
        else:
            node_feat_projected = node_feat_batched
            token_feat_projected = text_tokens

        # === Step 3: Atom-to-Token Attention ===
        Q_a2t = self.a2t_query(node_feat_projected)  # [batch, num_atoms, hidden]
        K_a2t = self.a2t_key(token_feat_projected)   # [batch, seq_len, hidden]
        V_a2t = self.a2t_value(token_feat_projected) # [batch, seq_len, hidden]

        # Multi-head attention
        Q_a2t = self.split_heads(Q_a2t)  # [batch, heads, num_atoms, head_dim]
        K_a2t = self.split_heads(K_a2t)  # [batch, heads, seq_len, head_dim]
        V_a2t = self.split_heads(V_a2t)  # [batch, heads, seq_len, head_dim]

        # Attention scores: [batch, heads, num_atoms, seq_len]
        attn_scores = torch.matmul(Q_a2t, K_a2t.transpose(-2, -1)) * self.scale

        # Apply token mask if provided (mask out padding tokens)
        if token_mask is not None:
            # token_mask: [batch, seq_len] -> [batch, 1, 1, seq_len]
            token_mask_expanded = token_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(~token_mask_expanded, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch, heads, num_atoms, seq_len]

        # Store attention weights for visualization
        if return_attention:
            self.last_attention_weights = attn_weights.detach()

        attn_weights_dropped = self.dropout(attn_weights)

        # Apply attention: [batch, heads, num_atoms, head_dim]
        context = torch.matmul(attn_weights_dropped, V_a2t)
        context = context.permute(0, 2, 1, 3).contiguous()  # [batch, num_atoms, heads, head_dim]
        context = context.view(batch_size, num_atoms, self.hidden_dim)
        context = self.output_proj(context)  # [batch, num_atoms, node_dim]

        # === Step 4: Optional gating mechanism (hybrid mode) ===
        if self.use_gate:
            gate_input = torch.cat([node_feat_batched, context], dim=-1)
            gate_values = self.gate(gate_input)  # [batch, num_atoms, node_dim]
            enhanced_batched = node_feat_batched + gate_values * context
        else:
            # Pure attention mode
            enhanced_batched = node_feat_batched + context

        # === Step 5: Layer norm ===
        enhanced_batched = self.layer_norm(enhanced_batched)
        enhanced_batched = self.dropout(enhanced_batched)

        # === Step 6: Convert back to DGL batched format ===
        if num_nodes != batch_size:
            # Extract actual nodes back to [total_nodes, node_dim]
            enhanced = torch.zeros_like(node_feat_orig)
            start_idx = 0
            for i, num in enumerate(batch_num_nodes):
                enhanced[start_idx:start_idx + num] = enhanced_batched[i, :num]
                start_idx += num
        else:
            # Remove the extra dimension: [batch_size, 1, node_dim] -> [batch_size, node_dim]
            enhanced = enhanced_batched.squeeze(1)

        if return_attention:
            return enhanced, attn_weights
        else:
            return enhanced


class CrossModalAttention(nn.Module):
    """Cross-modal attention between graph and text features.

    This module enables bidirectional attention mechanism where:
    - Graph features attend to text features
    - Text features attend to graph features
    Both modalities are enhanced through this interaction.
    """

    def __init__(self, graph_dim=256, text_dim=64, hidden_dim=256, num_heads=4, dropout=0.1):
        """Initialize cross-modal attention.

        Args:
            graph_dim: Dimension of graph features
            text_dim: Dimension of text features
            hidden_dim: Hidden dimension for attention computation
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Graph-to-Text attention (graph queries text)
        self.g2t_query = nn.Linear(graph_dim, hidden_dim)
        self.g2t_key = nn.Linear(text_dim, hidden_dim)
        self.g2t_value = nn.Linear(text_dim, hidden_dim)

        # Text-to-Graph attention (text queries graph)
        self.t2g_query = nn.Linear(text_dim, hidden_dim)
        self.t2g_key = nn.Linear(graph_dim, hidden_dim)
        self.t2g_value = nn.Linear(graph_dim, hidden_dim)

        # Output projections
        self.graph_output = nn.Linear(hidden_dim, graph_dim)
        self.text_output = nn.Linear(hidden_dim, text_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm_graph = nn.LayerNorm(graph_dim)
        self.layer_norm_text = nn.LayerNorm(text_dim)

        self.scale = self.head_dim ** -0.5

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, head_dim)."""
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # (batch, heads, seq, head_dim)

    def forward(self, graph_feat, text_feat, return_attention=False):
        """Forward pass of cross-modal attention.

        Args:
            graph_feat: Graph features [batch_size, graph_dim]
            text_feat: Text features [batch_size, text_dim]
            return_attention: 是否返回注意力权重（用于可解释性）

        Returns:
            enhanced_graph: Graph features enhanced by text [batch_size, graph_dim]
            enhanced_text: Text features enhanced by graph [batch_size, text_dim]
            attention_weights: (可选) 注意力权重字典
        """
        batch_size = graph_feat.size(0)

        # 存储注意力权重（用于可解释性）
        attention_weights = {} if return_attention else None

        # Add sequence dimension if needed
        if graph_feat.dim() == 2:
            graph_feat_seq = graph_feat.unsqueeze(1)  # [batch, 1, graph_dim]
        else:
            graph_feat_seq = graph_feat

        if text_feat.dim() == 2:
            text_feat_seq = text_feat.unsqueeze(1)  # [batch, 1, text_dim]
        else:
            text_feat_seq = text_feat

        # Graph-to-Text Attention: Graph attends to Text
        Q_g2t = self.g2t_query(graph_feat_seq)  # [batch, 1, hidden]
        K_g2t = self.g2t_key(text_feat_seq)     # [batch, 1, hidden]
        V_g2t = self.g2t_value(text_feat_seq)   # [batch, 1, hidden]

        # Multi-head attention
        Q_g2t = self.split_heads(Q_g2t, batch_size)
        K_g2t = self.split_heads(K_g2t, batch_size)
        V_g2t = self.split_heads(V_g2t, batch_size)

        attn_g2t = torch.matmul(Q_g2t, K_g2t.transpose(-2, -1)) * self.scale
        attn_g2t = F.softmax(attn_g2t, dim=-1)
        if return_attention:
            attention_weights['graph_to_text'] = attn_g2t.detach()  # [batch, heads, 1, 1]
        attn_g2t = self.dropout(attn_g2t)

        context_g2t = torch.matmul(attn_g2t, V_g2t)
        context_g2t = context_g2t.permute(0, 2, 1, 3).contiguous()
        context_g2t = context_g2t.view(batch_size, 1, self.hidden_dim)
        context_g2t = self.graph_output(context_g2t).squeeze(1)  # [batch, graph_dim]

        # Text-to-Graph Attention: Text attends to Graph
        Q_t2g = self.t2g_query(text_feat_seq)   # [batch, 1, hidden]
        K_t2g = self.t2g_key(graph_feat_seq)    # [batch, 1, hidden]
        V_t2g = self.t2g_value(graph_feat_seq)  # [batch, 1, hidden]

        # Multi-head attention
        Q_t2g = self.split_heads(Q_t2g, batch_size)
        K_t2g = self.split_heads(K_t2g, batch_size)
        V_t2g = self.split_heads(V_t2g, batch_size)

        attn_t2g = torch.matmul(Q_t2g, K_t2g.transpose(-2, -1)) * self.scale
        attn_t2g = F.softmax(attn_t2g, dim=-1)
        if return_attention:
            attention_weights['text_to_graph'] = attn_t2g.detach()  # [batch, heads, 1, 1]
        attn_t2g = self.dropout(attn_t2g)

        context_t2g = torch.matmul(attn_t2g, V_t2g)
        context_t2g = context_t2g.permute(0, 2, 1, 3).contiguous()
        context_t2g = context_t2g.view(batch_size, 1, self.hidden_dim)
        context_t2g = self.text_output(context_t2g).squeeze(1)  # [batch, text_dim]

        # Residual connection and layer normalization
        enhanced_graph = self.layer_norm_graph(graph_feat + context_g2t)
        enhanced_text = self.layer_norm_text(text_feat + context_t2g)

        if return_attention:
            return enhanced_graph, enhanced_text, attention_weights
        else:
            return enhanced_graph, enhanced_text


class GatedFusion(nn.Module):
    """门控融合模块 - 学习动态权重来平衡两种模态

    相比简单concat，门控融合可以：
    1. 自适应地控制每个模态的贡献
    2. 对不同样本学习不同的融合权重
    3. 提供更好的特征交互
    """

    def __init__(self, graph_dim=64, text_dim=64, output_dim=64, dropout=0.1):
        super().__init__()
        self.graph_dim = graph_dim
        self.text_dim = text_dim

        # 门控网络：学习每个模态的重要性权重
        self.gate_graph = nn.Sequential(
            nn.Linear(graph_dim, graph_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(graph_dim // 2, 1),
            nn.Sigmoid()
        )

        self.gate_text = nn.Sequential(
            nn.Linear(text_dim, text_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(text_dim // 2, 1),
            nn.Sigmoid()
        )

        # 特征变换
        self.graph_transform = nn.Linear(graph_dim, output_dim)
        self.text_transform = nn.Linear(text_dim, output_dim)

        # 融合后的变换
        self.fusion_transform = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, graph_feat, text_feat):
        """
        Args:
            graph_feat: [batch, graph_dim]
            text_feat: [batch, text_dim]
        Returns:
            fused: [batch, output_dim]
        """
        # 计算门控权重
        gate_g = self.gate_graph(graph_feat)  # [batch, 1]
        gate_t = self.gate_text(text_feat)    # [batch, 1]

        # 归一化门控权重（确保和为1）
        gate_sum = gate_g + gate_t + 1e-8
        gate_g = gate_g / gate_sum
        gate_t = gate_t / gate_sum

        # 特征变换
        graph_transformed = self.graph_transform(graph_feat)  # [batch, output_dim]
        text_transformed = self.text_transform(text_feat)     # [batch, output_dim]

        # 门控加权融合
        fused = gate_g * graph_transformed + gate_t * text_transformed

        # 最终变换
        fused = self.fusion_transform(fused)

        return fused


class BilinearFusion(nn.Module):
    """双线性池化融合 - 捕捉特征之间的二阶交互

    相比简单concat，双线性融合可以：
    1. 捕捉跨模态的特征交互（外积）
    2. 学习更丰富的联合表示
    3. 在视觉-语言任务中效果显著
    """

    def __init__(self, graph_dim=64, text_dim=64, output_dim=64, rank=16, dropout=0.1):
        super().__init__()
        # 使用低秩分解减少参数量：W = U * V^T
        # 完整双线性: (g^T W t) 需要 graph_dim × text_dim × output_dim 参数
        # 低秩分解: g^T U * V^T t 只需要 (graph_dim + text_dim) × rank × output_dim
        self.rank = rank

        self.graph_proj = nn.Linear(graph_dim, rank * output_dim, bias=False)
        self.text_proj = nn.Linear(text_dim, rank * output_dim, bias=False)

        self.output_dim = output_dim
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, graph_feat, text_feat):
        """
        Args:
            graph_feat: [batch, graph_dim]
            text_feat: [batch, text_dim]
        Returns:
            fused: [batch, output_dim]
        """
        batch_size = graph_feat.size(0)

        # 投影到低秩空间
        graph_proj = self.graph_proj(graph_feat)  # [batch, rank * output_dim]
        text_proj = self.text_proj(text_feat)     # [batch, rank * output_dim]

        # Reshape
        graph_proj = graph_proj.view(batch_size, self.rank, self.output_dim)
        text_proj = text_proj.view(batch_size, self.rank, self.output_dim)

        # 双线性交互（逐元素乘积然后求和）
        fused = torch.sum(graph_proj * text_proj, dim=1)  # [batch, output_dim]

        # 归一化和dropout
        fused = self.layer_norm(fused)
        fused = self.dropout(fused)

        return fused


class AdaptiveFusion(nn.Module):
    """自适应融合 - 基于样本内容动态选择融合策略

    相比简单concat，自适应融合可以：
    1. 对不同样本使用不同的融合策略
    2. 结合加法、乘法、门控等多种融合方式
    3. 通过注意力机制自动选择最佳策略
    """

    def __init__(self, graph_dim=64, text_dim=64, output_dim=64, dropout=0.1):
        super().__init__()

        # 特征对齐
        self.graph_align = nn.Linear(graph_dim, output_dim)
        self.text_align = nn.Linear(text_dim, output_dim)

        # 多种融合方式的权重预测器
        fusion_input_dim = output_dim * 2  # concat of graph and text
        self.fusion_selector = nn.Sequential(
            nn.Linear(fusion_input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, 3),  # 3种融合策略：加法、乘法、门控
            nn.Softmax(dim=-1)
        )

        # 门控融合分支
        self.gate_net = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.Sigmoid()
        )

        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph_feat, text_feat):
        """
        Args:
            graph_feat: [batch, graph_dim]
            text_feat: [batch, text_dim]
        Returns:
            fused: [batch, output_dim]
        """
        # 特征对齐到相同维度
        graph_aligned = self.graph_align(graph_feat)  # [batch, output_dim]
        text_aligned = self.text_align(text_feat)     # [batch, output_dim]

        # 预测融合策略权重
        concat_feat = torch.cat([graph_aligned, text_aligned], dim=-1)
        fusion_weights = self.fusion_selector(concat_feat)  # [batch, 3]

        # 三种融合方式
        # 1. 加法融合
        fusion_add = graph_aligned + text_aligned

        # 2. 哈达玛积（逐元素乘法）
        fusion_mul = graph_aligned * text_aligned

        # 3. 门控融合
        gate = self.gate_net(concat_feat)
        fusion_gate = gate * graph_aligned + (1 - gate) * text_aligned

        # 加权组合三种融合方式
        fused = (fusion_weights[:, 0:1] * fusion_add +
                 fusion_weights[:, 1:2] * fusion_mul +
                 fusion_weights[:, 2:3] * fusion_gate)

        # 归一化
        fused = self.layer_norm(fused)
        fused = self.dropout(fused)

        return fused


class TuckerFusion(nn.Module):
    """Tucker分解融合 - 高效的高阶张量分解

    相比简单concat，Tucker融合可以：
    1. 捕捉高阶特征交互
    2. 参数量更少但表达能力更强
    3. 在多模态学习中表现优异
    """

    def __init__(self, graph_dim=64, text_dim=64, output_dim=64, rank=8, dropout=0.1):
        super().__init__()
        self.rank = rank

        # Tucker分解的三个因子矩阵
        self.graph_factor = nn.Linear(graph_dim, rank, bias=False)
        self.text_factor = nn.Linear(text_dim, rank, bias=False)
        self.core_factor = nn.Linear(rank * rank, output_dim)

        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph_feat, text_feat):
        """
        Args:
            graph_feat: [batch, graph_dim]
            text_feat: [batch, text_dim]
        Returns:
            fused: [batch, output_dim]
        """
        batch_size = graph_feat.size(0)

        # 因子分解
        graph_compressed = self.graph_factor(graph_feat)  # [batch, rank]
        text_compressed = self.text_factor(text_feat)     # [batch, rank]

        # 外积形成核心张量
        # [batch, rank, 1] × [batch, 1, rank] = [batch, rank, rank]
        core_tensor = torch.bmm(
            graph_compressed.unsqueeze(2),
            text_compressed.unsqueeze(1)
        )  # [batch, rank, rank]

        # Flatten并投影到输出维度
        core_flat = core_tensor.view(batch_size, -1)  # [batch, rank*rank]
        fused = self.core_factor(core_flat)  # [batch, output_dim]

        # 归一化
        fused = self.layer_norm(fused)
        fused = self.dropout(fused)

        return fused


class FineGrainedCrossModalAttention(nn.Module):
    """Fine-grained cross-modal attention between atoms and text tokens.

    This module enables atom-level and token-level attention:
    - Each atom attends to all text tokens
    - Each text token attends to all atoms

    This provides detailed interpretability by showing which atoms
    attend to which words in the text description.
    """

    def __init__(self, node_dim=256, token_dim=768, hidden_dim=256,
                 num_heads=8, dropout=0.1, use_projection=True):
        """Initialize fine-grained cross-modal attention.

        Args:
            node_dim: Dimension of node (atom) features
            token_dim: Dimension of text token features (e.g., 768 for BERT)
            hidden_dim: Hidden dimension for attention computation
            num_heads: Number of attention heads
            dropout: Dropout rate
            use_projection: Whether to project inputs to same dimension
        """
        super().__init__()
        self.node_dim = node_dim
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.use_projection = use_projection

        # Input projections (optional, to match dimensions)
        if use_projection:
            self.node_proj_in = nn.Linear(node_dim, hidden_dim)
            self.token_proj_in = nn.Linear(token_dim, hidden_dim)

        # Atom-to-Token attention (atoms query tokens)
        self.a2t_query = nn.Linear(hidden_dim if use_projection else node_dim, hidden_dim)
        self.a2t_key = nn.Linear(hidden_dim if use_projection else token_dim, hidden_dim)
        self.a2t_value = nn.Linear(hidden_dim if use_projection else token_dim, hidden_dim)

        # Token-to-Atom attention (tokens query atoms)
        self.t2a_query = nn.Linear(hidden_dim if use_projection else token_dim, hidden_dim)
        self.t2a_key = nn.Linear(hidden_dim if use_projection else node_dim, hidden_dim)
        self.t2a_value = nn.Linear(hidden_dim if use_projection else node_dim, hidden_dim)

        # Output projections
        self.node_output = nn.Linear(hidden_dim, node_dim)
        self.token_output = nn.Linear(hidden_dim, token_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm_node = nn.LayerNorm(node_dim)
        self.layer_norm_token = nn.LayerNorm(token_dim)

        self.scale = self.head_dim ** -0.5

    def split_heads(self, x):
        """Split the last dimension into (num_heads, head_dim).

        Args:
            x: [batch_size, seq_len, hidden_dim]
        Returns:
            [batch_size, num_heads, seq_len, head_dim]
        """
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # (batch, heads, seq, head_dim)

    def forward(self, node_feat, token_feat, node_mask=None, token_mask=None,
                return_attention=False):
        """Forward pass of fine-grained cross-modal attention.

        Args:
            node_feat: Node features [batch_size, num_atoms, node_dim]
            token_feat: Token features [batch_size, seq_len, token_dim]
            node_mask: Optional mask for padded nodes [batch_size, num_atoms]
            token_mask: Optional mask for padded tokens [batch_size, seq_len]
            return_attention: Whether to return attention weights

        Returns:
            enhanced_nodes: Enhanced node features [batch_size, num_atoms, node_dim]
            enhanced_tokens: Enhanced token features [batch_size, seq_len, token_dim]
            attention_weights: (optional) Dict with 'atom_to_text' and 'text_to_atom'
        """
        batch_size = node_feat.size(0)
        num_atoms = node_feat.size(1)
        seq_len = token_feat.size(1)

        # Store original features for residual connection
        node_feat_orig = node_feat
        token_feat_orig = token_feat

        # Optional input projection
        if self.use_projection:
            node_feat = self.node_proj_in(node_feat)  # [batch, num_atoms, hidden]
            token_feat = self.token_proj_in(token_feat)  # [batch, seq_len, hidden]

        attention_weights = {} if return_attention else None

        # ============ Atom-to-Token Attention ============
        # Atoms attend to tokens: which words does each atom focus on?
        Q_a2t = self.a2t_query(node_feat)   # [batch, num_atoms, hidden]
        K_a2t = self.a2t_key(token_feat)    # [batch, seq_len, hidden]
        V_a2t = self.a2t_value(token_feat)  # [batch, seq_len, hidden]

        # Multi-head attention
        Q_a2t = self.split_heads(Q_a2t)  # [batch, heads, num_atoms, head_dim]
        K_a2t = self.split_heads(K_a2t)  # [batch, heads, seq_len, head_dim]
        V_a2t = self.split_heads(V_a2t)  # [batch, heads, seq_len, head_dim]

        # Attention scores: [batch, heads, num_atoms, seq_len]
        attn_a2t = torch.matmul(Q_a2t, K_a2t.transpose(-2, -1)) * self.scale

        # Apply token mask if provided (mask out padding tokens)
        if token_mask is not None:
            # token_mask: [batch, seq_len] -> [batch, 1, 1, seq_len]
            token_mask_expanded = token_mask.unsqueeze(1).unsqueeze(2)
            attn_a2t = attn_a2t.masked_fill(~token_mask_expanded, float('-inf'))

        attn_a2t = F.softmax(attn_a2t, dim=-1)

        if return_attention:
            # Store attention weights: [batch, heads, num_atoms, seq_len]
            attention_weights['atom_to_text'] = attn_a2t.detach()

        attn_a2t = self.dropout(attn_a2t)

        # Apply attention: [batch, heads, num_atoms, head_dim]
        context_a2t = torch.matmul(attn_a2t, V_a2t)
        context_a2t = context_a2t.permute(0, 2, 1, 3).contiguous()
        context_a2t = context_a2t.view(batch_size, num_atoms, self.hidden_dim)
        context_a2t = self.node_output(context_a2t)  # [batch, num_atoms, node_dim]

        # ============ Token-to-Atom Attention ============
        # Tokens attend to atoms: which atoms does each word focus on?
        Q_t2a = self.t2a_query(token_feat)  # [batch, seq_len, hidden]
        K_t2a = self.t2a_key(node_feat)     # [batch, num_atoms, hidden]
        V_t2a = self.t2a_value(node_feat)   # [batch, num_atoms, hidden]

        # Multi-head attention
        Q_t2a = self.split_heads(Q_t2a)  # [batch, heads, seq_len, head_dim]
        K_t2a = self.split_heads(K_t2a)  # [batch, heads, num_atoms, head_dim]
        V_t2a = self.split_heads(V_t2a)  # [batch, heads, num_atoms, head_dim]

        # Attention scores: [batch, heads, seq_len, num_atoms]
        attn_t2a = torch.matmul(Q_t2a, K_t2a.transpose(-2, -1)) * self.scale

        # Apply node mask if provided (mask out padding atoms)
        if node_mask is not None:
            # node_mask: [batch, num_atoms] -> [batch, 1, 1, num_atoms]
            node_mask_expanded = node_mask.unsqueeze(1).unsqueeze(2)
            attn_t2a = attn_t2a.masked_fill(~node_mask_expanded, float('-inf'))

        attn_t2a = F.softmax(attn_t2a, dim=-1)

        if return_attention:
            # Store attention weights: [batch, heads, seq_len, num_atoms]
            attention_weights['text_to_atom'] = attn_t2a.detach()

        attn_t2a = self.dropout(attn_t2a)

        # Apply attention: [batch, heads, seq_len, head_dim]
        context_t2a = torch.matmul(attn_t2a, V_t2a)
        context_t2a = context_t2a.permute(0, 2, 1, 3).contiguous()
        context_t2a = context_t2a.view(batch_size, seq_len, self.hidden_dim)
        context_t2a = self.token_output(context_t2a)  # [batch, seq_len, token_dim]

        # Residual connection and layer normalization
        enhanced_nodes = self.layer_norm_node(node_feat_orig + context_a2t)
        enhanced_tokens = self.layer_norm_token(token_feat_orig + context_t2a)

        if return_attention:
            return enhanced_nodes, enhanced_tokens, attention_weights
        else:
            return enhanced_nodes, enhanced_tokens


class ALIGNNConfig(BaseSettings):
    """Hyperparameter schema for jarvisdgl.models.alignn."""

    name: Literal["alignn"]
    alignn_layers: int = 4
    gcn_layers: int = 4
    atom_input_features: int = 92
    edge_input_features: int = 80
    triplet_input_features: int = 40
    embedding_features: int = 64
    hidden_features: int = 256
    # fc_layers: int = 1
    # fc_features: int = 64
    output_features: int = 1

    # Dropout for graph layers (regularization)
    graph_dropout: float = 0.0  # Dropout for ALIGNN and GCN layers

    # Cross-modal attention settings (late fusion)
    use_cross_modal_attention: bool = True
    cross_modal_hidden_dim: int = 256
    cross_modal_num_heads: int = 4
    cross_modal_dropout: float = 0.1

    # Late fusion type (后期融合方式)
    late_fusion_type: Literal["concat", "gated", "bilinear", "adaptive", "tucker"] = "concat"
    late_fusion_rank: int = 16  # For bilinear and tucker fusion (rank for low-rank decomposition)
    late_fusion_output_dim: int = 64  # Output dimension after fusion

    # Fine-grained attention settings (NEW!)
    use_fine_grained_attention: bool = False  # Enable fine-grained atom-token attention
    fine_grained_hidden_dim: int = 256
    fine_grained_num_heads: int = 8
    fine_grained_dropout: float = 0.1
    fine_grained_use_projection: bool = True  # Project inputs to same dimension
    mask_stopwords: bool = False
    remove_stopwords: bool = False
    stopwords_dir: str = ""

    # Middle fusion settings
    use_middle_fusion: bool = False
    middle_fusion_layers: str = "2"  # Comma-separated layer indices, e.g., "2" or "2,3" for ALIGNN layers
    middle_fusion_hidden_dim: int = 128
    middle_fusion_num_heads: int = 2
    middle_fusion_dropout: float = 0.1
    middle_fusion_use_gate_norm: bool = False  # Use LayerNorm before gate (for feature scale balance)
    middle_fusion_use_learnable_scale: bool = False  # Use learnable scaling factor for text features
    middle_fusion_initial_scale: float = 1.0  # Initial value for learnable scaling (use 12.0 based on diagnostics)

    # Enhanced middle fusion with attention (for interpretability)
    use_middle_fusion_attention: bool = False  # Use attention-based middle fusion instead of simple gating
    middle_fusion_attention_use_gate: bool = True  # Hybrid mode: attention + gate
    middle_fusion_attention_use_projection: bool = True  # Project text tokens to hidden_dim

    # Contrastive learning settings
    use_contrastive_loss: bool = False
    contrastive_loss_weight: float = 0.1
    contrastive_temperature: float = 0.1

    # if link == log, apply `exp` to final outputs
    # to constrain predictions to be positive
    link: Literal["identity", "log", "logit"] = "identity"
    zero_inflated: bool = False
    classification: bool = False

    class Config:
        """Configure model settings behavior."""

        env_prefix = "jv_model"


class EdgeGatedGraphConv(nn.Module):
    """Edge gated graph convolution from arxiv:1711.07553.

    see also arxiv:2003.0098.

    This is similar to CGCNN, but edge features only go into
    the soft attention / edge gating function, and the primary
    node update function is W cat(u, v) + b
    """

    def __init__(
        self, input_features: int, output_features: int, residual: bool = True, dropout: float = 0.0
    ):
        """Initialize parameters for ALIGNN update."""
        super().__init__()
        self.residual = residual
        # CGCNN-Conv operates on augmented edge features
        # z_ij = cat(v_i, v_j, u_ij)
        # m_ij = σ(z_ij W_f + b_f) ⊙ g_s(z_ij W_s + b_s)
        # coalesce parameters for W_f and W_s
        # but -- split them up along feature dimension
        self.src_gate = nn.Linear(input_features, output_features)
        self.dst_gate = nn.Linear(input_features, output_features)
        self.edge_gate = nn.Linear(input_features, output_features)
        self.bn_edges = nn.BatchNorm1d(output_features)

        self.src_update = nn.Linear(input_features, output_features)
        self.dst_update = nn.Linear(input_features, output_features)
        self.bn_nodes = nn.BatchNorm1d(output_features)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        g: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Edge-gated graph convolution.

        h_i^l+1 = ReLU(U h_i + sum_{j->i} eta_{ij} ⊙ V h_j)
        """
        g = g.local_var()

        # instead of concatenating (u || v || e) and applying one weight matrix
        # split the weight matrix into three, apply, then sum
        # see https://docs.dgl.ai/guide/message-efficient.html
        # but split them on feature dimensions to update u, v, e separately
        # m = BatchNorm(Linear(cat(u, v, e)))

        # compute edge updates, equivalent to:
        # Softplus(Linear(u || v || e))
        g.ndata["e_src"] = self.src_gate(node_feats)
        g.ndata["e_dst"] = self.dst_gate(node_feats)
        g.apply_edges(fn.u_add_v("e_src", "e_dst", "e_nodes"))
        m = g.edata.pop("e_nodes") + self.edge_gate(edge_feats)

        g.edata["sigma"] = torch.sigmoid(m)
        g.ndata["Bh"] = self.dst_update(node_feats)
        g.update_all(
            fn.u_mul_e("Bh", "sigma", "m"), fn.sum("m", "sum_sigma_h")
        )
        g.update_all(fn.copy_e("sigma", "m"), fn.sum("m", "sum_sigma"))
        g.ndata["h"] = g.ndata["sum_sigma_h"] / (g.ndata["sum_sigma"] + 1e-6)
        x = self.src_update(node_feats) + g.ndata.pop("h")

        # softmax version seems to perform slightly worse
        # that the sigmoid-gated version
        # compute node updates
        # Linear(u) + edge_gates ⊙ Linear(v)
        # g.edata["gate"] = edge_softmax(g, y)
        # g.ndata["h_dst"] = self.dst_update(node_feats)
        # g.update_all(fn.u_mul_e("h_dst", "gate", "m"), fn.sum("m", "h"))
        # x = self.src_update(node_feats) + g.ndata.pop("h")

        # node and edge updates
        x = F.silu(self.bn_nodes(x))
        y = F.silu(self.bn_edges(m))

        # Apply dropout for regularization
        x = self.dropout(x)
        y = self.dropout(y)

        if self.residual:
            x = node_feats + x
            y = edge_feats + y

        return x, y


class ALIGNNConv(nn.Module):
    """Line graph update."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.0,
    ):
        """Set up ALIGNN parameters."""
        super().__init__()
        self.node_update = EdgeGatedGraphConv(in_features, out_features, dropout=dropout)
        self.edge_update = EdgeGatedGraphConv(out_features, out_features, dropout=dropout)

    def forward(self,g: dgl.DGLGraph,lg: dgl.DGLGraph,x: torch.Tensor,y: torch.Tensor,z: torch.Tensor,):
        """Node and Edge updates for ALIGNN layer.

        x: node input features
        y: edge input features
        z: edge pair input features
        """
        g = g.local_var()
        lg = lg.local_var()
        # Edge-gated graph convolution update on crystal graph
        x, m = self.node_update(g, x, y)

        # Edge-gated graph convolution update on crystal graph
        y, z = self.edge_update(lg, m, z)

        return x, y, z


class MLPLayer(nn.Module):
    """Multilayer perceptron layer helper."""

    def __init__(self, in_features: int, out_features: int):
        """Linear, Batchnorm, SiLU layer."""
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.SiLU(),
        )

    def forward(self, x):
        """Linear, Batchnorm, silu layer."""
        return self.layer(x)


class ALIGNN(nn.Module):
    """Atomistic Line graph network.

    Chain alternating gated graph convolution updates on crystal graph
    and atomistic line graph.
    """

    def __init__(self, config: ALIGNNConfig = ALIGNNConfig(name="alignn")):
        """Initialize class with number of input features, conv layers."""
        super().__init__()
        # print(config)
        self.classification = config.classification

        self.atom_embedding = MLPLayer(config.atom_input_features, config.hidden_features)

        self.edge_embedding = nn.Sequential(
            RBFExpansion(vmin=0,vmax=8.0,bins=config.edge_input_features),
            MLPLayer(config.edge_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )
        self.angle_embedding = nn.Sequential(
            RBFExpansion(vmin=-1,vmax=1.0,bins=config.triplet_input_features),
            MLPLayer(config.triplet_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )

        self.alignn_layers = nn.ModuleList(
            [
                ALIGNNConv(config.hidden_features, config.hidden_features, dropout=config.graph_dropout)
                for idx in range(config.alignn_layers)
            ]
        )
        self.gcn_layers = nn.ModuleList(
            [
                EdgeGatedGraphConv(config.hidden_features, config.hidden_features, dropout=config.graph_dropout)
                for idx in range(config.gcn_layers)
            ]
        )

        self.readout = AvgPooling()

        self.graph_projection = ProjectionHead(embedding_dim=config.hidden_features)
        self.text_projection = ProjectionHead(embedding_dim=768)

        # Middle fusion modules
        self.use_middle_fusion = config.use_middle_fusion
        self.use_middle_fusion_attention = config.use_middle_fusion_attention
        self.middle_fusion_modules = nn.ModuleDict()
        if self.use_middle_fusion:
            # Parse middle_fusion_layers string to get layer indices
            fusion_layers = [int(x.strip()) for x in config.middle_fusion_layers.split(',')]

            # Choose between attention-based or simple gating-based middle fusion
            if self.use_middle_fusion_attention:
                print(f"\n{'='*80}")
                print(f"🔗 中期融合配置: 带注意力的中期融合 (可解释性增强)")
                print(f"{'='*80}")
                print(f"融合层: {fusion_layers}")
                print(f"隐藏维度: {config.middle_fusion_hidden_dim}")
                print(f"注意力头数: {config.middle_fusion_num_heads}")
                print(f"混合模式 (注意力+门控): {config.middle_fusion_attention_use_gate}")
                print(f"{'='*80}\n")

                for layer_idx in fusion_layers:
                    self.middle_fusion_modules[f'layer_{layer_idx}'] = MiddleFusionWithAttention(
                        node_dim=config.hidden_features,
                        text_token_dim=768,  # BERT token dimension (before projection)
                        hidden_dim=config.middle_fusion_hidden_dim,
                        num_heads=config.middle_fusion_num_heads,
                        dropout=config.middle_fusion_dropout,
                        use_gate=config.middle_fusion_attention_use_gate,
                        use_projection=config.middle_fusion_attention_use_projection
                    )
            else:
                # Original simple gating-based middle fusion
                for layer_idx in fusion_layers:
                    self.middle_fusion_modules[f'layer_{layer_idx}'] = MiddleFusionModule(
                        node_dim=config.hidden_features,
                        text_dim=64,  # After text_projection
                        hidden_dim=config.middle_fusion_hidden_dim,
                        num_heads=config.middle_fusion_num_heads,
                        dropout=config.middle_fusion_dropout,
                        use_gate_norm=config.middle_fusion_use_gate_norm,
                        use_learnable_scale=config.middle_fusion_use_learnable_scale,
                        initial_scale=config.middle_fusion_initial_scale
                    )
            self.middle_fusion_layer_indices = fusion_layers

        # Fine-grained cross-modal attention module (atom-token level)
        self.use_fine_grained_attention = config.use_fine_grained_attention
        if self.use_fine_grained_attention:
            self.fine_grained_attention = FineGrainedCrossModalAttention(
                node_dim=config.hidden_features,  # Node features from ALIGNN layers
                token_dim=768,  # BERT token dimension
                hidden_dim=config.fine_grained_hidden_dim,
                num_heads=config.fine_grained_num_heads,
                dropout=config.fine_grained_dropout,
                use_projection=config.fine_grained_use_projection
            )

        # Cross-modal attention module (global level, for backward compatibility)
        self.use_cross_modal_attention = config.use_cross_modal_attention
        if self.use_cross_modal_attention:
            self.cross_modal_attention = CrossModalAttention(
                graph_dim=64,  # After graph_projection
                text_dim=64,   # After text_projection
                hidden_dim=config.cross_modal_hidden_dim,
                num_heads=config.cross_modal_num_heads,
                dropout=config.cross_modal_dropout
            )

        # Late fusion module
        self.late_fusion_type = config.late_fusion_type
        print(f"\n{'='*80}")
        print(f"🔗 后期融合配置")
        print(f"{'='*80}")
        print(f"融合类型: {config.late_fusion_type}")

        if config.late_fusion_type == "concat":
            # 原始concat融合
            self.fusion_module = None
            self.fc1 = nn.Linear(128, 64)  # Concat: 64 + 64 = 128-dim
            self.fc = nn.Linear(64, config.output_features)
            print(f"参数: 简单拼接，输入128维 -> 64维 -> {config.output_features}维")

        elif config.late_fusion_type == "gated":
            # 门控融合
            self.fusion_module = GatedFusion(
                graph_dim=64,
                text_dim=64,
                output_dim=config.late_fusion_output_dim,
                dropout=config.cross_modal_dropout
            )
            self.fc = nn.Linear(config.late_fusion_output_dim, config.output_features)
            print(f"参数: 门控融合，输出维度 {config.late_fusion_output_dim}")

        elif config.late_fusion_type == "bilinear":
            # 双线性融合
            self.fusion_module = BilinearFusion(
                graph_dim=64,
                text_dim=64,
                output_dim=config.late_fusion_output_dim,
                rank=config.late_fusion_rank,
                dropout=config.cross_modal_dropout
            )
            self.fc = nn.Linear(config.late_fusion_output_dim, config.output_features)
            print(f"参数: 双线性融合，Rank={config.late_fusion_rank}, 输出维度 {config.late_fusion_output_dim}")

        elif config.late_fusion_type == "adaptive":
            # 自适应融合
            self.fusion_module = AdaptiveFusion(
                graph_dim=64,
                text_dim=64,
                output_dim=config.late_fusion_output_dim,
                dropout=config.cross_modal_dropout
            )
            self.fc = nn.Linear(config.late_fusion_output_dim, config.output_features)
            print(f"参数: 自适应融合（加法+乘法+门控），输出维度 {config.late_fusion_output_dim}")

        elif config.late_fusion_type == "tucker":
            # Tucker分解融合
            self.fusion_module = TuckerFusion(
                graph_dim=64,
                text_dim=64,
                output_dim=config.late_fusion_output_dim,
                rank=config.late_fusion_rank,
                dropout=config.cross_modal_dropout
            )
            self.fc = nn.Linear(config.late_fusion_output_dim, config.output_features)
            print(f"参数: Tucker分解融合，Rank={config.late_fusion_rank}, 输出维度 {config.late_fusion_output_dim}")

        else:
            raise ValueError(f"Unknown late_fusion_type: {config.late_fusion_type}")

        print(f"{'='*80}\n")

        # Contrastive learning module
        self.use_contrastive_loss = config.use_contrastive_loss
        if self.use_contrastive_loss:
            self.contrastive_loss_fn = ContrastiveLoss(temperature=config.contrastive_temperature)
            self.contrastive_loss_weight = config.contrastive_loss_weight

        self.link = None
        self.link_name = config.link
        if config.link == "identity":
            self.link = lambda x: x
        elif config.link == "log":
            self.link = torch.exp
            avg_gap = 0.7  # magic number -- average bandgap in dft_3d
            self.fc.bias.data = torch.tensor(
                np.log(avg_gap), dtype=torch.float
            )
        elif config.link == "logit":
            self.link = torch.sigmoid

    def forward(self, g: Union[Tuple[dgl.DGLGraph, dgl.DGLGraph], dgl.DGLGraph],
               return_features=False, return_attention=False, return_intermediate_features=False):
        """ALIGNN : start with `atom_features`.

        Args:
            g: Graph(s) and text input
            return_features: If True, return dict with predictions and intermediate features
            return_attention: If True, include attention weights in returned dict (for interpretability)
            return_intermediate_features: If True, return features at each fusion stage (for ablation studies)

        Returns:
            If return_features=False: predictions [batch_size]
            If return_features=True or return_attention=True: dict with predictions and features/attention
            If return_intermediate_features=True: dict with features at each fusion stage

        x: atom features (g.ndata)
        y: bond features (g.edata and lg.ndata)
        z: angle features (lg.edata)
        """
        if len(self.alignn_layers) > 0:
            # g, lg = g
            g, lg, text = g
            lg = lg.local_var()

            # angle features (fixed)
            z = self.angle_embedding(lg.edata.pop("h"))

        g = g.local_var()


        # Text Encoding
        norm_sents = [normalize(s) for s in text]
        encodings = tokenizer(norm_sents, return_tensors='pt', padding=True, truncation=True)
        if torch.cuda.is_available():
            encodings.to(device)
        with torch.no_grad():
            last_hidden_state = text_model(**encodings)[0]  # [batch, seq_len, 768]

        # For fine-grained attention: keep all tokens
        text_tokens = last_hidden_state  # [batch, seq_len, 768]
        attention_mask = encodings['attention_mask']  # [batch, seq_len]

        # For backward compatibility: CLS token + projection
        cls_emb = last_hidden_state[:, 0, :]  # [batch, 768]
        text_emb = self.text_projection(cls_emb)  # [batch, 64]

        # Store base text features for ablation studies
        text_emb_base = text_emb.clone() if return_intermediate_features else None

        # initial node features: atom feature network...
        x = g.ndata.pop("atom_features")
        x = self.atom_embedding(x)

        # initial bond features
        bondlength = torch.norm(g.edata.pop("r"), dim=1)
        y = self.edge_embedding(bondlength)

        # ALIGNN updates: update node, edge, triplet features
        middle_fusion_attention_weights = {}  # Store attention weights from middle fusion
        for idx, alignn_layer in enumerate(self.alignn_layers):
            x, y, z = alignn_layer(g, lg, x, y, z)

            # Apply middle fusion if configured for this layer
            if self.use_middle_fusion and idx in self.middle_fusion_layer_indices:
                # Get batch information for proper text broadcasting
                batch_num_nodes = g.batch_num_nodes().tolist()

                if self.use_middle_fusion_attention:
                    # Attention-based middle fusion (with interpretability)
                    if return_attention:
                        x, attn_weights = self.middle_fusion_modules[f'layer_{idx}'](
                            x, text_tokens, batch_num_nodes,
                            token_mask=attention_mask.bool(),
                            return_attention=True
                        )
                        middle_fusion_attention_weights[f'layer_{idx}'] = attn_weights
                    else:
                        x = self.middle_fusion_modules[f'layer_{idx}'](
                            x, text_tokens, batch_num_nodes,
                            token_mask=attention_mask.bool(),
                            return_attention=False
                        )
                else:
                    # Original simple gating-based middle fusion
                    x = self.middle_fusion_modules[f'layer_{idx}'](x, text_emb, batch_num_nodes)

        # Save features after middle fusion (if requested for ablation studies)
        graph_emb_after_middle = None
        if return_intermediate_features and self.use_middle_fusion:
            # Need to do readout + projection to get graph-level features
            temp_graph_emb = self.readout(g, x)
            graph_emb_after_middle = self.graph_projection(temp_graph_emb).clone()

        # gated GCN updates: update node, edge features
        for gcn_layer in self.gcn_layers:
            x, y = gcn_layer(g, x, y)

        # Save BASE graph features BEFORE any attention (for ablation studies)
        # This must be done before fine-grained attention modifies x
        graph_emb_base = None
        if return_intermediate_features:
            temp_graph_emb = self.readout(g, x)
            graph_emb_base = self.graph_projection(temp_graph_emb).clone()

        # Fine-grained cross-modal attention (before readout)
        fine_grained_attention_weights = None
        if self.use_fine_grained_attention:
            # Convert node features from DGL format to batched format
            # DGL batches graphs by concatenating: x is [total_atoms, node_dim]
            # We need [batch_size, max_atoms, node_dim] for attention
            batch_num_nodes = g.batch_num_nodes().tolist()  # List of num_atoms per graph
            batch_size = len(batch_num_nodes)
            max_atoms = max(batch_num_nodes)
            node_dim = x.size(1)

            # Split and pad node features
            node_features_batched = torch.zeros(batch_size, max_atoms, node_dim,
                                                 device=x.device, dtype=x.dtype)
            node_mask = torch.zeros(batch_size, max_atoms, device=x.device, dtype=torch.bool)

            offset = 0
            for i, num_nodes in enumerate(batch_num_nodes):
                node_features_batched[i, :num_nodes] = x[offset:offset+num_nodes]
                node_mask[i, :num_nodes] = True
                offset += num_nodes

            # Apply fine-grained attention
            if return_attention:
                enhanced_nodes, enhanced_tokens, fine_grained_attention_weights = \
                    self.fine_grained_attention(
                        node_features_batched,
                        text_tokens,
                        node_mask=node_mask,
                        token_mask=attention_mask.bool(),
                        return_attention=True
                    )
            else:
                enhanced_nodes, enhanced_tokens = self.fine_grained_attention(
                    node_features_batched,
                    text_tokens,
                    node_mask=node_mask,
                    token_mask=attention_mask.bool()
                )

            # Convert back to DGL format: [batch, max_atoms, node_dim] -> [total_atoms, node_dim]
            x_enhanced = torch.zeros_like(x)
            offset = 0
            for i, num_nodes in enumerate(batch_num_nodes):
                x_enhanced[offset:offset+num_nodes] = enhanced_nodes[i, :num_nodes]
                offset += num_nodes

            x = x_enhanced  # Use enhanced node features

        # Save features after fine-grained attention (if requested for ablation studies)
        graph_emb_after_fine = None
        text_emb_after_fine = None
        if return_intermediate_features and self.use_fine_grained_attention:
            # Graph features after fine-grained attention
            temp_graph_emb = self.readout(g, x)
            graph_emb_after_fine = self.graph_projection(temp_graph_emb).clone()

            # Text features after fine-grained attention
            # Use the enhanced_tokens pooled representation
            text_emb_after_fine = enhanced_tokens[:, 0, :]  # CLS token
            text_emb_after_fine = self.text_projection(text_emb_after_fine).clone()

        # norm-activation-pool-classify
        graph_emb = self.readout(g, x)
        h = self.graph_projection(graph_emb)
        # Note: graph_emb_base was already saved earlier (before fine-grained attention)
        # to ensure it captures true "base" features without any attention applied

        # Multi-Modal Representation Fusion
        attention_weights = None
        if self.use_cross_modal_attention:
            # Cross-modal attention fusion
            if return_attention:
                enhanced_graph, enhanced_text, attention_weights = self.cross_modal_attention(
                    h, text_emb, return_attention=True
                )
            else:
                enhanced_graph, enhanced_text = self.cross_modal_attention(h, text_emb)

            # Late fusion using configured fusion module
            if self.late_fusion_type == "concat":
                # Original concat fusion
                h = torch.cat([enhanced_graph, enhanced_text], dim=-1)  # [batch, 128]
                h = F.relu(self.fc1(h))
                out = self.fc(h)
            else:
                # Use advanced fusion modules
                fused = self.fusion_module(enhanced_graph, enhanced_text)
                out = self.fc(fused)
        else:
            # No cross-modal attention - use simple fusion
            if self.late_fusion_type == "concat":
                # Original concat fusion
                h = torch.cat((h, text_emb), 1)
                h = F.relu(self.fc1(h))
                out = self.fc(h)
            else:
                # Use advanced fusion modules
                fused = self.fusion_module(h, text_emb)
                out = self.fc(fused)

        if self.link:
            out = self.link(out)

        if self.classification:
            # out = torch.round(torch.sigmoid(out))
            #out = self.softmax(out)
            # For binary classification, use sigmoid
            out = torch.sigmoid(out)

        predictions = torch.squeeze(out)

        # Return intermediate features if requested (for contrastive learning or interpretability)
        if return_features or self.use_contrastive_loss or return_attention or return_intermediate_features:
            output_dict = {
                'predictions': predictions,
                'graph_features': h if not self.use_cross_modal_attention else enhanced_graph,
                'text_features': text_emb if not self.use_cross_modal_attention else enhanced_text,
            }

            # Add intermediate features for ablation studies
            if return_intermediate_features:
                output_dict['graph_base'] = graph_emb_base
                output_dict['text_base'] = text_emb_base
                if self.use_middle_fusion and graph_emb_after_middle is not None:
                    output_dict['graph_middle'] = graph_emb_after_middle
                if self.use_fine_grained_attention and graph_emb_after_fine is not None:
                    output_dict['graph_fine'] = graph_emb_after_fine
                    output_dict['text_fine'] = text_emb_after_fine
                if self.use_cross_modal_attention:
                    output_dict['graph_cross'] = enhanced_graph
                    output_dict['text_cross'] = enhanced_text

            # Add attention weights if requested (for interpretability)
            if return_attention:
                # Global attention weights (backward compatibility)
                if attention_weights is not None:
                    output_dict['attention_weights'] = attention_weights
                # Fine-grained attention weights (new!)
                if fine_grained_attention_weights is not None:
                    output_dict['fine_grained_attention_weights'] = fine_grained_attention_weights
                # Middle fusion attention weights (new! for interpretability)
                if middle_fusion_attention_weights:
                    output_dict['middle_fusion_attention_weights'] = middle_fusion_attention_weights

            # Compute contrastive loss if enabled
            if self.use_contrastive_loss and self.training:
                graph_feat = h if not self.use_cross_modal_attention else enhanced_graph
                text_feat = text_emb if not self.use_cross_modal_attention else enhanced_text
                contrastive_loss = self.contrastive_loss_fn(graph_feat, text_feat)
                output_dict['contrastive_loss'] = contrastive_loss

            return output_dict if return_features or return_attention or return_intermediate_features or (self.use_contrastive_loss and self.training) else predictions

        return predictions
