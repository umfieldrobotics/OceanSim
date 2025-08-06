import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import json
import os
import math
import PIL.Image as Image

def plot_sonar_sdg(base, index=0):
    # File paths
    sonar_path = os.path.join(base, 'sonar_image', f'{index}.png')
    instance_seg_path = os.path.join(base, 'instance_segmentation', f'{index}.png')
    semantic_seg_path = os.path.join(base, 'semantic_segmentation', f'{index}.png')
    debug_path = os.path.join(base, 'debug', f'{index}.png')
    instance_rgb_path = os.path.join(base, 'instance_rgb', f'{index}.png')
    mapping_path = os.path.join(base, 'semantic_mapping.json')

    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    axs = np.atleast_2d(axs)

    # --- Raw Sonar Image ---
    ax = axs[0, 0]
    if os.path.exists(sonar_path):
        sonar_img = mpimg.imread(sonar_path)
        ax.imshow(sonar_img, cmap='gray')
        ax.set_title('Raw Sonar Image', fontsize=14, fontweight='bold')
    else:
        ax.set_title('Raw Sonar Image (Not Found)', fontsize=14, fontweight='bold')
    ax.axis('off')

    # --- Debug Image ---
    ax2 = axs[0, 1]
    if os.path.exists(debug_path):
        debug_img = mpimg.imread(debug_path)
        ax2.imshow(debug_img)
        ax2.set_title('Debug Image', fontsize=14, fontweight='bold')
    else:
        ax2.set_title('Debug Image (Not Found)', fontsize=14, fontweight='bold')
    ax2.axis('off')

    # --- Instance RGB Image ---
    ax3 = axs[0, 2]
    if os.path.exists(instance_rgb_path):
        rgb_img = mpimg.imread(instance_rgb_path)
        ax3.imshow(rgb_img)
        ax3.set_title('Instance RGB', fontsize=14, fontweight='bold')
    else:
        ax3.set_title('Instance RGB (Not Found)', fontsize=14, fontweight='bold')
    ax3.axis('off')

    # --- Instance Segmentation ---
    ax4 = axs[1, 0]
    if os.path.exists(instance_seg_path):
        inst_img = np.array(Image.open(instance_seg_path))
        instance_id = inst_img & 0xFF
        unique_ids = np.unique(instance_id)
        num_ids = len(unique_ids)
        import matplotlib
        from matplotlib.patches import Patch
        cmap = matplotlib.colormaps['tab20']
        colors = [cmap(i) for i in range(num_ids)]
        listed_cmap = matplotlib.colors.ListedColormap(colors)
        boundaries = np.append(unique_ids, unique_ids[-1]+1)
        norm = matplotlib.colors.BoundaryNorm(boundaries, listed_cmap.N)
        legend_elements = []
        for i, id_ in enumerate(unique_ids[:10]):
            if id_ == 0:
                legend_elements.append(Patch(facecolor=colors[i], edgecolor='k', label=f'No Instance'))
                continue
            legend_elements.append(Patch(facecolor=colors[i], edgecolor='k', label=f'ID {id_}'))
        ax4.legend(handles=legend_elements, loc='lower right', fontsize=8, borderaxespad=0.)
        ax4.imshow(instance_id, cmap=listed_cmap, norm=norm, interpolation='nearest')
        ax4.set_title('Instance Segmentation', fontsize=14, fontweight='bold')
    else:
        ax4.set_title('Instance Segmentation (Not Found)', fontsize=14, fontweight='bold')
    ax4.axis('off')

    # --- Semantic Segmentation ---
    ax5 = axs[1, 1]
    if os.path.exists(instance_seg_path):
        inst_img = np.array(Image.open(instance_seg_path))
        semantic_id = (inst_img >> 8) & 0xFF
        if os.path.exists(mapping_path):
            with open(mapping_path, 'r') as f:
                semantic_mapping = json.load(f)
            id_to_class = {v: k for k, v in semantic_mapping.items()}
            max_id = max(id_to_class.keys())
            num_classes = max_id + 1
        else:
            id_to_class = {}
            num_classes = int(semantic_id.max()) + 1
        import matplotlib
        cmap = matplotlib.colormaps['tab20']
        colors = [cmap(i % cmap.N) for i in range(num_classes)]
        unique_ids = np.unique(semantic_id)
        seg_colors = [colors[idx] for idx in unique_ids]
        listed_cmap = matplotlib.colors.ListedColormap(seg_colors)
        boundaries = np.append(unique_ids, unique_ids[-1]+1)
        norm = matplotlib.colors.BoundaryNorm(boundaries, listed_cmap.N)
        ax5.imshow(semantic_id, cmap=listed_cmap, norm=norm, interpolation='nearest')
        ax5.set_title('Semantic Segmentation', fontsize=14, fontweight='bold')
    else:
        ax5.set_title('Semantic Segmentation (Not Found)', fontsize=14, fontweight='bold')
    ax5.axis('off')

    # --- Semantic Mapping Table ---
    ax6 = axs[1, 2]
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r') as f:
            semantic_mapping = json.load(f)
        id_to_class = {v: k for k, v in semantic_mapping.items()}
        max_id = max(id_to_class.keys())
        num_classes = max_id + 1
        import matplotlib
        cmap = matplotlib.colormaps['tab20']
        colors = [cmap(i % cmap.N) for i in range(num_classes)]
        table_data = []
        cell_colours = []
        for idx in range(num_classes):
            if idx not in id_to_class:
                continue
            class_name = id_to_class[idx]
            table_data.append([class_name, idx])
            rgb = colors[idx][:3]
            cell_colours.append([rgb, [1, 1, 1]])
        n_cols = 2
        n_items = len(table_data)
        n_rows = math.ceil(n_items / n_cols)
        pad_len = n_rows * n_cols - n_items
        table_data_padded = table_data + [["", ""]] * pad_len
        cell_colours_padded = cell_colours + [[[1,1,1], [1,1,1]]] * pad_len
        table_grid = []
        colour_grid = []
        for row in range(n_rows):
            row_cells = []
            row_colours = []
            for col in range(n_cols):
                idx = row + col * n_rows
                row_cells.extend(table_data_padded[idx])
                row_colours.extend(cell_colours_padded[idx])
            table_grid.append(row_cells)
            colour_grid.append(row_colours)
        ax6.axis('off')
        table = ax6.table(
            cellText=table_grid,
            cellColours=colour_grid,
            loc='center',
            cellLoc='center',
            colLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        for (row, col), cell in table.get_celld().items():
            cell.set_linewidth(0.7)
            cell.set_edgecolor('gray')
        table.auto_set_column_width(list(range(n_cols*2)))
        ax6.set_title('Semantic Mapping Table', fontsize=14, fontweight='bold')
    else:
        ax6.set_title('Semantic Mapping Table (Not Found)', fontsize=14, fontweight='bold')
        ax6.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default=".", help="Base directory for the data set")
    parser.add_argument("--index", type=int, default=0, help="Frame index to visualize")
    args = parser.parse_args()
    plot_sonar_sdg(args.base, args.index)
