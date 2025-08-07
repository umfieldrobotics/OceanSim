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
    debug_path = os.path.join(base, 'debug', f'{index}.png')
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

    # --- Max Intensity Color Strip at ax3 ---
    ax3 = axs[0, 2]
    max_intensity_path = os.path.join(base, 'max_intensity', f'{index}.npy')
    if os.path.exists(max_intensity_path):
        max_intensity = np.load(max_intensity_path)
        # Create a 2D image by repeating the 1D array vertically
        color_strip = np.tile(max_intensity, (50, 1))  # 50 rows, adjust as needed
        im = ax3.imshow(color_strip, aspect='auto', cmap='viridis')
        ax3.set_title('Max Intensity Color Strip', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        ax3.set_xlabel('Pixel Index')
        ax3.set_ylabel('Repeated Rows')
    else:
        ax3.set_title('Max Intensity (Not Found)', fontsize=14, fontweight='bold')
    ax3.axis('on')

    # --- Sonar Parameters Table at ax3 ---
    ax3 = axs[0, 2]
    sonar_param_path = os.path.join(base, 'sonar_param.json')
    if os.path.exists(sonar_param_path):
        with open(sonar_param_path, 'r') as f:
            sonar_param_json = json.load(f)
        sonar_param = sonar_param_json.get('sonar_param', {})
        table_data = [[str(k), str(v)] for k, v in sonar_param.items()]
        col_labels = ['Parameter', 'Value']
        n_rows = len(table_data)
        # Alternating row colors
        row_colors = [['#f2f2f2', 'white'][i % 2] for i in range(n_rows)]
        cell_colours = [[row_colors[i], row_colors[i]] for i in range(n_rows)]
        ax3.axis('off')
        table = ax3.table(
            cellText=table_data,
            colLabels=col_labels,
            cellColours=cell_colours,
            loc='center',
            cellLoc='center',
            colLoc='center',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(13)
        table.scale(1.3, 1.3)
        # Bold header row and center align
        for (row, col), cell in table.get_celld().items():
            cell.set_linewidth(0.7)
            cell.set_edgecolor('gray')
            cell.set_fontsize(13)
            if row == 0:
                cell.set_text_props(ha='center', va='center', weight='bold')
            else:
                cell.set_text_props(ha='center', va='center')
            if row == 0 and col in [0, 1]:
                cell.set_facecolor('#d9ead3')  # header color
        table.auto_set_column_width([0, 1])
        ax3.set_title('Sonar Parameters', fontsize=15, fontweight='bold', pad=12)
    else:
        ax3.set_title('Sonar Parameters (Not Found)', fontsize=14, fontweight='bold')

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

    # --- Pointcloud Plot (Open3D) ---
    pcl_path = os.path.join(base, 'pcl', f'{index}.npy')
    try:
        import open3d as o3d
        import threading
        o3d_available = True
    except ImportError:
        o3d_available = False
    def show_o3d(pcl_path):
        if os.path.exists(pcl_path):
            pcl = np.load(pcl_path)
            if pcl.shape[1] >= 3 and o3d_available:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pcl[:, :3])
                origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
                o3d.visualization.draw_geometries([pcd, origin], window_name='Open3D Pointcloud')
    if os.path.exists(pcl_path) and o3d_available:
        t = threading.Thread(target=show_o3d, args=(pcl_path,))
        t.start()

    plt.tight_layout()
    plt.show()  # blocking, keeps matplotlib window open
    if os.path.exists(pcl_path) and o3d_available:
        t.join()  # wait for Open3D window to close if still open

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default=".", help="Base directory for the data set")
    parser.add_argument("--index", type=int, default=0, help="Frame index to visualize")
    args = parser.parse_args()
    plot_sonar_sdg(args.base, args.index)
