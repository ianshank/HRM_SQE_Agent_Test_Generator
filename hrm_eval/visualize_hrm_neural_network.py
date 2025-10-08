"""
HRM Neural Network Architecture Visualization
Detailed matplotlib diagram of the hierarchical transformer architecture.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle, Polygon
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

def create_neural_network_visualization():
    """Create detailed HRM neural network architecture diagram."""
    
    fig, ax = plt.subplots(figsize=(16, 20))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 24)
    ax.axis('off')
    
    # Title
    fig.suptitle('HRM v9 Neural Network Architecture', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Color scheme
    input_color = '#E3F2FD'
    embed_color = '#BBDEFB'
    encoder_color = '#90CAF9'
    decoder_color = '#64B5F6'
    output_color = '#42A5F5'
    rl_color = '#FF9800'
    
    # ===== Input Layer =====
    y_pos = 22
    draw_layer(ax, 'Input Layer', y_pos, input_color, 
               ['Tokenized Puzzle State', 'Vocab Size: 12', 'Sequence Length: Variable'])
    
    # Arrow
    draw_arrow(ax, 5, y_pos - 0.5, 5, y_pos - 1.5)
    
    # ===== Puzzle Embedding Layer =====
    y_pos = 20
    draw_layer(ax, 'Puzzle Embedding', y_pos, embed_color,
               ['Embedding Dim: 128', 'Learnable Embeddings', 'Token → Dense Vector'])
    
    # Arrow
    draw_arrow(ax, 5, y_pos - 0.5, 5, y_pos - 1.5)
    
    # ===== Positional Encoding =====
    y_pos = 18
    draw_layer(ax, 'Positional Encoding', y_pos, embed_color,
               ['Sinusoidal Encoding', 'Position-aware', 'Added to Embeddings'])
    
    # Arrow
    draw_arrow(ax, 5, y_pos - 0.5, 5, y_pos - 1.5)
    
    # ===== Hierarchical Transformer Stack =====
    y_pos = 16
    
    # Transformer Block 1
    draw_transformer_block(ax, y_pos, 'Transformer Block 1', 
                          encoder_color, decoder_color)
    
    y_pos -= 3
    draw_arrow(ax, 5, y_pos + 2.5, 5, y_pos + 0.5)
    
    # Transformer Block 2
    draw_transformer_block(ax, y_pos, 'Transformer Block 2',
                          encoder_color, decoder_color)
    
    y_pos -= 3
    draw_arrow(ax, 5, y_pos + 2.5, 5, y_pos + 0.5)
    
    # Transformer Block N (indication)
    ax.text(5, y_pos + 1.5, '⋮', ha='center', va='center', 
            fontsize=30, fontweight='bold', color='gray')
    ax.text(5, y_pos + 0.8, 'N Transformer Blocks', ha='center', 
            fontsize=10, style='italic', color='gray')
    
    y_pos -= 1
    
    # ===== Output Processing =====
    y_pos = 5
    
    # Layer Norm
    draw_layer(ax, 'Layer Normalization', y_pos, output_color,
               ['Stabilize Training', 'Normalize Activations'])
    
    draw_arrow(ax, 5, y_pos - 0.5, 5, y_pos - 1.5)
    y_pos -= 2
    
    # Dense Layer
    draw_layer(ax, 'Dense Layer', y_pos, output_color,
               ['Hidden Size: 256', 'ReLU Activation'])
    
    # Split into two paths
    draw_arrow(ax, 5, y_pos - 0.5, 3, y_pos - 2, color='blue')
    draw_arrow(ax, 5, y_pos - 0.5, 7, y_pos - 2, color='orange')
    
    y_pos -= 3
    
    # ===== Dual Output Heads =====
    
    # Action Head (Left)
    draw_output_head(ax, 2.5, y_pos, 'Action Prediction Head', 
                    output_color,
                    ['Output: Token Probabilities', 
                     'Vocab Size: 12',
                     'Softmax Activation'])
    
    # RL Q-Head (Right)
    draw_output_head(ax, 7.5, y_pos, 'RL Q-Value Head',
                    rl_color,
                    ['Output: Q-Values',
                     'Action Space: 2',
                     'Linear Activation'])
    
    # ===== Add technical specifications box =====
    add_specs_box(ax)
    
    # ===== Add architecture details =====
    add_architecture_details(ax)
    
    return fig

def draw_layer(ax, title, y_pos, color, details):
    """Draw a neural network layer."""
    # Main box
    box = FancyBboxPatch((1, y_pos - 0.4), 8, 1.2,
                         boxstyle="round,pad=0.1",
                         facecolor=color, edgecolor='black', linewidth=2)
    ax.add_patch(box)
    
    # Title
    ax.text(5, y_pos + 0.5, title, ha='center', va='center',
            fontsize=13, fontweight='bold')
    
    # Details
    detail_y = y_pos + 0.1
    for detail in details:
        ax.text(5, detail_y, detail, ha='center', va='center',
                fontsize=9, style='italic')
        detail_y -= 0.25

def draw_transformer_block(ax, y_pos, title, encoder_color, decoder_color):
    """Draw a detailed transformer block."""
    # Outer box
    outer_box = FancyBboxPatch((0.5, y_pos - 0.5), 9, 2.2,
                               boxstyle="round,pad=0.1",
                               facecolor='white', edgecolor='black', 
                               linewidth=3, linestyle='--', alpha=0.3)
    ax.add_patch(outer_box)
    
    # Title
    ax.text(5, y_pos + 1.5, title, ha='center', va='top',
            fontsize=12, fontweight='bold')
    
    # Multi-Head Attention
    attention_box = FancyBboxPatch((1, y_pos + 0.3), 8, 0.7,
                                   boxstyle="round,pad=0.05",
                                   facecolor=encoder_color, 
                                   edgecolor='black', linewidth=2)
    ax.add_patch(attention_box)
    ax.text(5, y_pos + 0.65, 'Multi-Head Self-Attention', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(5, y_pos + 0.4, 'Heads: 8 | d_k: 64 | d_v: 64',
            ha='center', va='center', fontsize=8, style='italic')
    
    # Add & Norm
    norm1_box = Rectangle((2.5, y_pos - 0.1), 5, 0.3,
                          facecolor=decoder_color, edgecolor='black', linewidth=1)
    ax.add_patch(norm1_box)
    ax.text(5, y_pos + 0.05, 'Add & Norm', ha='center', va='center',
            fontsize=9, fontweight='bold')
    
    # Feed-Forward
    ff_box = FancyBboxPatch((1, y_pos - 0.6), 8, 0.4,
                            boxstyle="round,pad=0.05",
                            facecolor=decoder_color, edgecolor='black', linewidth=2)
    ax.add_patch(ff_box)
    ax.text(5, y_pos - 0.4, 'Feed-Forward Network (FFN)', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(5, y_pos - 0.55, 'd_ff: 512 | Dropout: 0.1',
            ha='center', va='center', fontsize=8, style='italic')

def draw_output_head(ax, x_center, y_pos, title, color, details):
    """Draw an output head."""
    box = FancyBboxPatch((x_center - 1.5, y_pos - 0.4), 3, 1.2,
                         boxstyle="round,pad=0.1",
                         facecolor=color, edgecolor='black', linewidth=2)
    ax.add_patch(box)
    
    # Title
    ax.text(x_center, y_pos + 0.5, title, ha='center', va='center',
            fontsize=11, fontweight='bold', wrap=True)
    
    # Details
    detail_y = y_pos + 0.1
    for detail in details:
        ax.text(x_center, detail_y, detail, ha='center', va='center',
                fontsize=8, style='italic')
        detail_y -= 0.25

def draw_arrow(ax, x1, y1, x2, y2, color='black'):
    """Draw an arrow between layers."""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=20,
                           color=color, linewidth=2)
    ax.add_patch(arrow)

def add_specs_box(ax):
    """Add technical specifications box."""
    # Box
    spec_box = FancyBboxPatch((0.2, 0.2), 4.5, 1.3,
                             boxstyle="round,pad=0.1",
                             facecolor='#FFF9C4', edgecolor='black', linewidth=2)
    ax.add_patch(spec_box)
    
    # Title
    ax.text(2.45, 1.35, 'Model Specifications', ha='center', va='top',
            fontsize=11, fontweight='bold')
    
    # Specs
    specs = [
        'Parameters: ~2M',
        'Layers: 6 Transformer Blocks',
        'Attention Heads: 8',
        'Hidden Size: 128',
        'FFN Size: 512',
        'Dropout: 0.1',
        'Training Steps: 7,566',
    ]
    
    y_spec = 1.1
    for spec in specs:
        ax.text(0.4, y_spec, spec, ha='left', va='center',
                fontsize=8, family='monospace')
        y_spec -= 0.15

def add_architecture_details(ax):
    """Add architecture details box."""
    # Box
    detail_box = FancyBboxPatch((5.3, 0.2), 4.5, 1.3,
                               boxstyle="round,pad=0.1",
                               facecolor='#E8F5E9', edgecolor='black', linewidth=2)
    ax.add_patch(detail_box)
    
    # Title
    ax.text(7.55, 1.35, 'Architecture Features', ha='center', va='top',
            fontsize=11, fontweight='bold')
    
    # Features
    features = [
        '[x] Hierarchical Structure',
        '[x] Self-Attention Mechanism',
        '[x] Residual Connections',
        '[x] Layer Normalization',
        '[x] Positional Encoding',
        '[x] Dual Output (Action + Q)',
        '[x] Reinforcement Learning',
    ]
    
    y_feat = 1.1
    for feat in features:
        ax.text(5.5, y_feat, feat, ha='left', va='center',
                fontsize=8)
        y_feat -= 0.15

def add_attention_visualization(ax):
    """Add attention mechanism visualization (optional detail)."""
    pass

def main():
    """Main execution."""
    print("Generating HRM Neural Network Visualization...")
    
    fig = create_neural_network_visualization()
    
    # Add footer
    fig.text(0.5, 0.01, 
             'HRM v9 Optimized Architecture | Checkpoint: Step 7566 | Status: Production Ready',
             ha='center', fontsize=10, style='italic', color='gray')
    
    # Save figure
    output_png = 'test_results/hrm_neural_network.png'
    output_pdf = 'test_results/hrm_neural_network.pdf'
    
    fig.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[DONE] PNG saved to: {output_png}")
    
    fig.savefig(output_pdf, format='pdf', bbox_inches='tight', facecolor='white')
    print(f"[DONE] PDF saved to: {output_pdf}")
    
    plt.show()
    
    print("\n" + "="*80)
    print("NEURAL NETWORK VISUALIZATION COMPLETE")
    print("="*80)
    print(f"PNG: {output_png} (300 DPI)")
    print(f"PDF: {output_pdf} (Vector)")
    print("="*80)

if __name__ == "__main__":
    main()
