#!/usr/bin/env python
"""
Fine-Grained Cross-Modal Attention Analysis Demo

This script demonstrates how to use the fine-grained attention mechanism
to analyze atom-level and token-level interactions in material property prediction.

Usage:
    python demo_fine_grained_attention.py \
        --model_path /path/to/checkpoint.pt \
        --cif_path /path/to/structure.cif \
        --text "Material description..." \
        --save_dir ./results
"""

import argparse
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from jarvis.core.atoms import Atoms
from jarvis.core.graphs import Graph
from alignn.graphs import cgcnn_features as features
from transformers import BertTokenizer

from band.models.alignn import ALIGNN, ALIGNNConfig
from band.interpretability_enhanced import EnhancedInterpretabilityAnalyzer


def load_model_with_fine_grained_attention(checkpoint_path, device='cuda'):
    """Load model with fine-grained attention enabled."""

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create config with fine-grained attention enabled
    config = ALIGNNConfig(
        name="alignn",
        alignn_layers=4,
        gcn_layers=4,
        atom_input_features=92,
        hidden_features=256,
        output_features=1,

        # Enable cross-modal features
        use_cross_modal_attention=True,
        cross_modal_hidden_dim=256,
        cross_modal_num_heads=4,

        # Enable middle fusion
        use_middle_fusion=True,
        middle_fusion_layers="2",

        # ⭐ Enable fine-grained attention (NEW!)
        use_fine_grained_attention=True,
        fine_grained_hidden_dim=256,
        fine_grained_num_heads=8,
        fine_grained_dropout=0.1,
        fine_grained_use_projection=True
    )

    # Create model
    model = ALIGNN(config)

    # Load weights
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model = model.to(device)
    model.eval()

    print("✅ Model loaded with fine-grained attention enabled")
    print(f"   - Fine-grained attention heads: {config.fine_grained_num_heads}")
    print(f"   - Hidden dimension: {config.fine_grained_hidden_dim}")

    return model, config


def cif_to_graph(cif_path, cutoff=8.0, max_neighbors=12):
    """Convert CIF file to DGL graph."""

    # Read structure
    atoms = Atoms.from_poscar(cif_path)

    # Create graph using JARVIS
    g, lg = Graph.atom_dgl_multigraph(
        atoms=atoms,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        atom_features="atomic_number",
        compute_line_graph=True,
        use_canonize=True
    )

    # Convert atomic_number to cgcnn features
    z = g.ndata.pop("atom_features")
    g.ndata["atomic_number"] = z
    z = z.type(torch.LongTensor).squeeze()
    f = torch.tensor(features[z], dtype=torch.float32)
    g.ndata["atom_features"] = f

    return g, lg, atoms


def analyze_with_fine_grained_attention(
    model,
    g,
    lg,
    text,
    atoms_object,
    save_dir=None
):
    """Perform analysis with fine-grained attention."""

    device = next(model.parameters()).device
    g = g.to(device)
    lg = lg.to(device)

    # Get prediction with attention weights
    with torch.no_grad():
        output = model(
            [g, lg, [text]],
            return_features=True,
            return_attention=True
        )

    prediction = output['predictions'].cpu().item()

    # Extract fine-grained attention weights
    fg_attn = output.get('fine_grained_attention_weights', None)

    if fg_attn is None:
        print("❌ Fine-grained attention weights not found!")
        print("   Make sure the model has use_fine_grained_attention=True")
        return

    print(f"\n✅ Prediction: {prediction:.4f}")
    print(f"\n✅ Fine-grained attention extracted:")
    print(f"   - atom_to_text shape: {fg_attn['atom_to_text'].shape}")
    print(f"   - text_to_atom shape: {fg_attn['text_to_atom'].shape}")

    # Tokenize text to get token strings
    tokenizer = BertTokenizer.from_pretrained('m3rg-iitd/matscibert')
    tokens = tokenizer.tokenize(text)
    tokens = ['[CLS]'] + tokens + ['[SEP]']  # Add special tokens

    # Truncate or pad to match attention shape
    seq_len = fg_attn['atom_to_text'].shape[-1]
    if len(tokens) > seq_len:
        tokens = tokens[:seq_len]
    elif len(tokens) < seq_len:
        tokens = tokens + ['[PAD]'] * (seq_len - len(tokens))

    # Create analyzer for visualization
    analyzer = EnhancedInterpretabilityAnalyzer(model, device=device)

    # Visualize fine-grained attention
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / 'fine_grained_attention.png'
    else:
        save_path = 'fine_grained_attention.png'

    analysis = analyzer.visualize_fine_grained_attention(
        attention_weights=fg_attn,
        atoms_object=atoms_object,
        text_tokens=tokens,
        save_path=save_path,
        top_k_atoms=10,
        top_k_words=15,
        show_all_heads=False
    )

    # Print analysis results
    print(f"\n{'='*80}")
    print("📊 Fine-Grained Attention Analysis")
    print(f"{'='*80}\n")

    if analysis and 'overall_top_words' in analysis:
        print("🔤 Top 10 Most Important Words (overall):")
        print(f"{'Rank':<6} {'Word':<20} {'Importance':<10}")
        print("-" * 40)
        for rank, (word, importance) in enumerate(analysis['overall_top_words'][:10], 1):
            print(f"{rank:<6} {word:<20} {importance:.4f}")

    print()

    if analysis and 'overall_top_atoms' in analysis:
        print("⚛️  Top 10 Most Important Atoms (overall):")
        print(f"{'Rank':<6} {'Atom':<20} {'Importance':<10}")
        print("-" * 40)
        for rank, (atom, importance) in enumerate(analysis['overall_top_atoms'][:10], 1):
            print(f"{rank:<6} {atom:<20} {importance:.4f}")

    print()

    # Print some specific atom-word pairs
    if analysis and 'atom_top_words' in analysis:
        print("🔍 Top Words for Each Atom:")
        print("-" * 60)
        for atom_id, top_words in list(analysis['atom_top_words'].items())[:5]:
            print(f"\n{atom_id}:")
            for word, importance in top_words[:5]:
                print(f"  - {word:<20} {importance:.4f}")

    print(f"\n{'='*80}\n")

    return analysis


def main():
    parser = argparse.ArgumentParser(description='Fine-Grained Attention Analysis')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--cif_path', type=str, required=True,
                        help='Path to CIF file')
    parser.add_argument('--text', type=str, required=True,
                        help='Text description of the material')
    parser.add_argument('--save_dir', type=str, default='./fine_grained_results',
                        help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print("🔬 Fine-Grained Cross-Modal Attention Analysis")
    print(f"{'='*80}\n")

    # Load model
    print("📦 Loading model...")
    model, config = load_model_with_fine_grained_attention(
        args.model_path,
        device=args.device
    )

    # Load structure
    print(f"\n📂 Loading structure from: {args.cif_path}")
    g, lg, atoms_object = cif_to_graph(args.cif_path)
    print(f"   - Number of atoms: {atoms_object.num_atoms}")
    print(f"   - Formula: {atoms_object.composition.reduced_formula}")

    # Analyze
    print(f"\n🔍 Analyzing with text:")
    print(f'   "{args.text[:100]}..."')

    analysis = analyze_with_fine_grained_attention(
        model=model,
        g=g,
        lg=lg,
        text=args.text,
        atoms_object=atoms_object,
        save_dir=args.save_dir
    )

    print(f"✅ Analysis complete! Results saved to: {args.save_dir}\n")


if __name__ == '__main__':
    main()
