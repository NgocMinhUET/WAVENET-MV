"""
Create AdaMixNet Architecture Detail Figure
Tạo hình ảnh chi tiết cho AdaMixNet component
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, Arrow

# Set IEEE paper style
plt.rcParams.update({
    'font.size': 9,
    'font.family': 'serif',
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'lines.linewidth': 1.5,
    'grid.linewidth': 0.5,
    'figure.dpi': 300
})

def create_adamixnet_detail():
    """Tạo sơ đồ chi tiết AdaMixNet architecture"""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # Input - Wavelet coefficients
    input_rect = FancyBboxPatch(
        (0.5, 4.5), 2.5, 1.5,
        boxstyle="round,pad=0.1",
        facecolor="#E8F4FD",
        edgecolor="black",
        linewidth=2
    )
    ax.add_patch(input_rect)
    ax.text(1.75, 5.25, "Wavelet Coefficients\n[H_LL, H_LH, H_HL, H_HH]\n256×H×W", 
            ha="center", va="center", fontsize=10, fontweight="bold")
    
    # Split into 4 branches
    branches = [
        {"name": "H_LL\n64×H×W", "pos": (4, 7.5), "color": "#FFE6CC"},
        {"name": "H_LH\n64×H×W", "pos": (4, 5.5), "color": "#CCFFCC"},
        {"name": "H_HL\n64×H×W", "pos": (4, 3.5), "color": "#FFCCCC"},
        {"name": "H_HH\n64×H×W", "pos": (4, 1.5), "color": "#CCCCFF"}
    ]
    
    for i, branch in enumerate(branches):
        rect = FancyBboxPatch(
            branch["pos"], 2, 1,
            boxstyle="round,pad=0.05",
            facecolor=branch["color"],
            edgecolor="black",
            linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(branch["pos"][0] + 1, branch["pos"][1] + 0.5,
                branch["name"], ha="center", va="center", fontsize=9, fontweight="bold")
    
    # Parallel processing branches
    conv_layers = [
        {"name": "Conv3×3\n(64→64)", "pos": (7, 7.5), "color": "#FFE6CC"},
        {"name": "Conv3×3\n(64→64)", "pos": (7, 5.5), "color": "#CCFFCC"},
        {"name": "Conv3×3\n(64→64)", "pos": (7, 3.5), "color": "#FFCCCC"},
        {"name": "Conv3×3\n(64→64)", "pos": (7, 1.5), "color": "#CCCCFF"}
    ]
    
    for conv in conv_layers:
        rect = FancyBboxPatch(
            conv["pos"], 2, 1,
            boxstyle="round,pad=0.05",
            facecolor=conv["color"],
            edgecolor="black",
            linewidth=1
        )
        ax.add_patch(rect)
        ax.text(conv["pos"][0] + 1, conv["pos"][1] + 0.5,
                conv["name"], ha="center", va="center", fontsize=9, fontweight="bold")
    
    # ReLU activations
    relu_layers = [
        {"name": "ReLU", "pos": (9.5, 7.5), "color": "#FFF2CC"},
        {"name": "ReLU", "pos": (9.5, 5.5), "color": "#FFF2CC"},
        {"name": "ReLU", "pos": (9.5, 3.5), "color": "#FFF2CC"},
        {"name": "ReLU", "pos": (9.5, 1.5), "color": "#FFF2CC"}
    ]
    
    for relu in relu_layers:
        rect = FancyBboxPatch(
            relu["pos"], 1.5, 1,
            boxstyle="round,pad=0.05",
            facecolor=relu["color"],
            edgecolor="black",
            linewidth=1
        )
        ax.add_patch(rect)
        ax.text(relu["pos"][0] + 0.75, relu["pos"][1] + 0.5,
                relu["name"], ha="center", va="center", fontsize=9, fontweight="bold")
    
    # Global Average Pooling for attention
    gap_rect = FancyBboxPatch(
        (1, 8.5), 2.5, 1,
        boxstyle="round,pad=0.1",
        facecolor="#DDA0DD",
        edgecolor="black",
        linewidth=1.5
    )
    ax.add_patch(gap_rect)
    ax.text(2.25, 9, "Global Average Pooling\n256×1×1", ha="center", va="center", 
            fontsize=9, fontweight="bold")
    
    # Fully Connected layers for attention
    fc_layers = [
        {"name": "FC\n(256→128)", "pos": (4.5, 8.5), "color": "#DDA0DD"},
        {"name": "ReLU", "pos": (7, 8.5), "color": "#DDA0DD"},
        {"name": "FC\n(128→4)", "pos": (9, 8.5), "color": "#DDA0DD"},
        {"name": "Softmax", "pos": (11.5, 8.5), "color": "#DDA0DD"}
    ]
    
    for fc in fc_layers:
        rect = FancyBboxPatch(
            fc["pos"], 1.5, 1,
            boxstyle="round,pad=0.05",
            facecolor=fc["color"],
            edgecolor="black",
            linewidth=1
        )
        ax.add_patch(rect)
        ax.text(fc["pos"][0] + 0.75, fc["pos"][1] + 0.5,
                fc["name"], ha="center", va="center", fontsize=8, fontweight="bold")
    
    # Attention weights
    attention_rect = FancyBboxPatch(
        (13.5, 8.5), 2, 1,
        boxstyle="round,pad=0.1",
        facecolor="#FFD700",
        edgecolor="black",
        linewidth=2
    )
    ax.add_patch(attention_rect)
    ax.text(14.5, 9, "Attention Weights\n[α₁, α₂, α₃, α₄]", ha="center", va="center", 
            fontsize=9, fontweight="bold")
    
    # Weighted combination
    weighted_features = [
        {"name": "α₁ ⊙ F₁\n32×H×W", "pos": (12, 7.5), "color": "#FFE6CC"},
        {"name": "α₂ ⊙ F₂\n32×H×W", "pos": (12, 5.5), "color": "#CCFFCC"},
        {"name": "α₃ ⊙ F₃\n32×H×W", "pos": (12, 3.5), "color": "#FFCCCC"},
        {"name": "α₄ ⊙ F₄\n32×H×W", "pos": (12, 1.5), "color": "#CCCCFF"}
    ]
    
    for wf in weighted_features:
        rect = FancyBboxPatch(
            wf["pos"], 2, 1,
            boxstyle="round,pad=0.05",
            facecolor=wf["color"],
            edgecolor="black",
            linewidth=1
        )
        ax.add_patch(rect)
        ax.text(wf["pos"][0] + 1, wf["pos"][1] + 0.5,
                wf["name"], ha="center", va="center", fontsize=9, fontweight="bold")
    
    # Final summation
    sum_rect = FancyBboxPatch(
        (15, 4.5), 2.5, 1.5,
        boxstyle="round,pad=0.1",
        facecolor="#90EE90",
        edgecolor="black",
        linewidth=2
    )
    ax.add_patch(sum_rect)
    ax.text(16.25, 5.25, "Σ\nMixed Features\n128×H×W", ha="center", va="center", 
            fontsize=11, fontweight="bold")
    
    # Add arrows
    arrow_props = dict(arrowstyle='->', lw=2, color='black')
    
    # From input to branches
    for i in range(4):
        ax.annotate('', xy=(4, 2 + i*2), xytext=(3, 5.25), arrowprops=arrow_props)
    
    # Through processing pipeline
    for i in range(4):
        y_pos = 2 + i*2
        ax.annotate('', xy=(7, y_pos), xytext=(6, y_pos), arrowprops=arrow_props)
        ax.annotate('', xy=(9.5, y_pos), xytext=(9, y_pos), arrowprops=arrow_props)
        ax.annotate('', xy=(12, y_pos), xytext=(11, y_pos), arrowprops=arrow_props)
        ax.annotate('', xy=(15, 5.25), xytext=(14, y_pos), arrowprops=arrow_props)
    
    # Attention pathway
    ax.annotate('', xy=(4.5, 9), xytext=(3.5, 9), arrowprops=arrow_props)
    ax.annotate('', xy=(7, 9), xytext=(6, 9), arrowprops=arrow_props)
    ax.annotate('', xy=(9, 9), xytext=(8.5, 9), arrowprops=arrow_props)
    ax.annotate('', xy=(11.5, 9), xytext=(10.5, 9), arrowprops=arrow_props)
    ax.annotate('', xy=(13.5, 9), xytext=(13, 9), arrowprops=arrow_props)
    
    # From attention to features
    for i in range(4):
        ax.annotate('', xy=(13, 2 + i*2), xytext=(14.5, 8.5), arrowprops=arrow_props)
    
    # Add labels
    ax.text(1.75, 10.5, "Attention Computation", ha="center", va="center", 
            fontsize=12, fontweight="bold", color="purple")
    ax.text(8, 0.5, "Parallel Feature Processing", ha="center", va="center", 
            fontsize=12, fontweight="bold", color="blue")
    ax.text(13, 0.5, "Adaptive Mixing", ha="center", va="center", 
            fontsize=12, fontweight="bold", color="green")
    
    # Mathematical equation
    ax.text(9, 10.5, r"$\mathbf{Y} = \sum_{i=1}^{4} \alpha_i \odot \text{ReLU}(\text{Conv3×3}(\mathbf{W}_i))$", 
            ha="center", va="center", fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 11)
    ax.set_title("AdaMixNet: Adaptive Feature Mixing Architecture Detail", 
                fontsize=14, fontweight="bold", pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('fig_adamixnet_detail.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ AdaMixNet detail created: fig_adamixnet_detail.png")

if __name__ == '__main__':
    create_adamixnet_detail() 