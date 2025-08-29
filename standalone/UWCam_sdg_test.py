import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import json
import cv2
import os
import matplotlib
import math
def load_kitti_label(label_path):
    objects = []
    with open(label_path, 'r') as f:
        for line in f:
            if line.strip() == '': continue
            data = line.strip().split()
            obj = {
                'type': data[0],
                'truncated': float(data[1]),
                'occluded': int(data[2]),
                'alpha': float(data[3]),
                'bbox': [float(x) for x in data[4:8]],
                'dimensions': [float(x) for x in data[8:11]],  # h, w, l
                'location': [float(x) for x in data[11:14]],  # x, y, z (user: z backward)
                'rotation_y': float(data[14]),
            }
            objects.append(obj)
    return objects



def plot_all_debug(label_path, rgb_path, debug_path, instance_seg_path, mapping_path):
    objects = load_kitti_label(label_path)
    fig, axs = plt.subplots(2, 3, figsize=(24, 12))
    axs = np.atleast_2d(axs)

    # --- BEV subplot ---
    ax = axs[0, 0]
    for obj in objects:
        loc = obj['location']
        dims = obj['dimensions']
        ry = obj['rotation_y']
        label = obj['type']
        x, y, z = loc
        z = -z  # convert to z-forward
        alpha = ry - np.arctan2(x, z)
        alpha = alpha % (2 * np.pi)
        alpha = alpha - 2 * np.pi if alpha > np.pi else alpha  # normalize to [-pi, pi]
        assert np.isclose(obj['alpha'], alpha, atol=1e-2) # 1e-2 because SDG saves :2f
        ax.plot(x, z, 'ro')
        ax.text(x, z, label, color='blue', fontsize=10, ha='center', va='bottom')
        arrow_length = 0.5
        dx = np.cos(ry) * arrow_length
        dz = np.sin(ry) * arrow_length
        ax.arrow(x, z, dx, dz, head_width=0.12, head_length=0.15, fc='m', ec='m', linewidth=2)
        h, w, l = dims
        # Correct order: front-right, front-left, rear-left, rear-right, close
        corners = np.array([
            [ l/2, -w/2],   # front-right
            [ l/2,  w/2],   # front-left
            [-l/2,  w/2],   # rear-left
            [-l/2, -w/2],   # rear-right
            [ l/2, -w/2],   # close rectangle
        ])
        R = np.array([
            [np.cos(ry), -np.sin(ry)],
            [np.sin(ry),  np.cos(ry)]
        ])
        bev_corners = (R @ corners.T).T + np.array([x, z])
        ax.plot(bev_corners[:,0], bev_corners[:,1], 'g-', linewidth=2)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-0.5, 5)
    ax.plot(0, 0, marker='^', color='black', markersize=14, label='Camera')
    ax.text(0, 0, 'Camera', color='black', fontsize=12, ha='right', va='bottom', fontweight='bold')
    cam_arrow_len = 1.0
    ax.arrow(0, 0, 0, cam_arrow_len, head_width=0.2, head_length=0.2, fc='navy', ec='navy', linewidth=3, zorder=5, label='Camera View')
    ax.text(0, cam_arrow_len+0.2, 'View', color='navy', fontsize=12, ha='center', va='bottom', fontweight='bold')
    ax.set_facecolor('#f7f7fa')
    ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
    ax.set_xlabel('x (right, m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('z (forward, m)', fontsize=12, fontweight='bold')
    ax.set_title('BEV', fontsize=14, fontweight='bold')
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='^', color='w', label='Camera', markerfacecolor='black', markersize=14),
        Line2D([0], [0], color='navy', lw=3, label='Camera View'),
        Line2D([0], [0], marker='o', color='w', label='Object Center', markerfacecolor='red', markersize=10),
        Line2D([0], [0], color='g', lw=2, label='Object BBox'),
        Line2D([0], [0], color='m', lw=2, label='Object Direction'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    # --- 2D BBox on RGB image subplot ---
    ax2 = axs[0, 1]
    img = mpimg.imread(rgb_path)
    ax2.imshow(img)
    for obj in objects:
        bbox = obj['bbox']  # [xmin, ymin, xmax, ymax]
        label = obj['type']
        occlusion = obj['occluded']
        truncation = obj['truncated']
        # Compose label with occlusion and truncation
        label_text = f"{label}\n({occlusion}, {truncation:.2f})"
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=2, edgecolor='lime', facecolor='none')
        ax2.add_patch(rect)
        ax2.text(
            bbox[0], bbox[1]-5, label_text,
            color='lime', fontsize=10, fontweight='bold', va='bottom', ha='left',
            bbox=dict(facecolor='black', alpha=0.3, edgecolor='none', pad=1)
        )
    ax2.set_title('2D Bounding Boxes (occlusion level, truncation ratio)', fontsize=14, fontweight='bold')
    ax2.axis('off')


    # --- Debug image subplot ---
    ax3 = axs[0, 2]
    debug_img = mpimg.imread(debug_path)
    ax3.imshow(debug_img)
    ax3.set_title('Debug Image', fontsize=14, fontweight='bold')
    ax3.axis('off')


    # --- Instance segmentation subplot (second row) ---
    ax4 = axs[1, 0]
    # Read 16-bit PNG
    import PIL.Image as Image
    inst_img = np.array(Image.open(instance_seg_path))

    instance_id = inst_img & 0xFF
    unique_ids = np.unique(instance_id)
    num_ids = len(unique_ids)

    import matplotlib
    from matplotlib.patches import Patch
    # Create a ListedColormap with as many colors as unique ids
    cmap = matplotlib.colormaps['tab20']
    colors = [cmap(i) for i in range(num_ids)]
    listed_cmap = matplotlib.colors.ListedColormap(colors)
    # Set boundaries so each unique id gets its own color
    boundaries = np.append(unique_ids, unique_ids[-1]+1)
    norm = matplotlib.colors.BoundaryNorm(boundaries, listed_cmap.N)
    # Legend for instance ids and their colors (first 10 for readability)
    legend_elements = []
    for i, id_ in enumerate(unique_ids[:10]):
        if id_ == 0:
            legend_elements.append(Patch(facecolor=colors[i], edgecolor='k', label=f'No Instance'))
            continue
        legend_elements.append(Patch(facecolor=colors[i], edgecolor='k', label=f'ID {id_}'))
    ax4.legend(handles=legend_elements, loc='lower right', fontsize=8, borderaxespad=0.)
    # Plot instance_id directly
    ax4.imshow(instance_id, cmap=listed_cmap, norm=norm, interpolation='nearest')
    ax4.set_title('Instance Segmentation', fontsize=14, fontweight='bold')
    ax4.axis('off')


    # --- Semantic segmentation subplot (second row) ---
    ax5 = axs[1, 1]
    semantic_id = (inst_img >> 8) & 0xFF
    # Read semantic mapping and build id-to-class mapping
    import matplotlib
    with open(mapping_path, 'r') as f:
        semantic_mapping = json.load(f)
    id_to_class = {v: k for k, v in semantic_mapping.items()}
    max_id = max(id_to_class.keys())
    num_classes = max_id + 1
    cmap = matplotlib.colormaps['tab20']
    colors = [cmap(i % cmap.N) for i in range(num_classes)]  # RGBA tuples
    # For semantic segmentation subplot
    unique_ids = np.unique(semantic_id)
    seg_colors = [colors[idx] for idx in unique_ids]
    listed_cmap = matplotlib.colors.ListedColormap(seg_colors)
    boundaries = np.append(unique_ids, unique_ids[-1]+1)
    norm = matplotlib.colors.BoundaryNorm(boundaries, listed_cmap.N)
    ax5.imshow(semantic_id, cmap=listed_cmap, norm=norm, interpolation='nearest')
    ax5.set_title('Semantic Segmentation', fontsize=14, fontweight='bold')
    ax5.axis('off')

    # --- Semantic mapping table subplot (second row, last column) ---
    table_data = []
    cell_colours = []
    for idx in range(num_classes):
        if idx not in id_to_class:
            continue
        class_name = id_to_class[idx]
        table_data.append([class_name, idx])
        rgb = colors[idx][:3]  # Only RGB, ignore alpha
        cell_colours.append([rgb, [1, 1, 1]])  # Color only class name cell
    n_cols = 2  # Number of columns you want
    n_items = len(table_data)
    n_rows = math.ceil(n_items / n_cols)
    # Pad table_data and cell_colours to fill the grid
    pad_len = n_rows * n_cols - n_items
    table_data_padded = table_data + [["", ""]] * pad_len
    cell_colours_padded = cell_colours + [[[1,1,1], [1,1,1]]] * pad_len
    # Reshape into grid (each row is a list of [class, id, class, id, ...])
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
    # Build column labels (no longer needed)
    # col_labels_multi = []
    # for i in range(n_cols):
    #     col_labels_multi += [f'Class Name {i+1}', f'ID {i+1}']
    ax6 = axs[1, 2]
    ax6.axis('off')
    table = ax6.table(
        cellText=table_grid,
        # colLabels=col_labels_multi,  # Remove column labels
        cellColours=colour_grid,
        loc='center',
        cellLoc='center',
        colLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    # Style grid as before (skip header row styling)
    for (row, col), cell in table.get_celld().items():
        # Highlight UNLABELLED row
        if row > 0 and any('UNLABELLED' == table_grid[row-1][c*2] for c in range(n_cols)):
            cell.set_edgecolor('red')
            cell.set_linewidth(2)
        cell.set_linewidth(0.7)
        cell.set_edgecolor('gray')
    table.auto_set_column_width(list(range(n_cols*2)))

    plt.tight_layout()
    plt.show()



def plot_all(label_path, rgb_path, instance_seg_path, mapping_path):
    objects = load_kitti_label(label_path)
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    axs = np.atleast_2d(axs)

    # --- BEV subplot ---
    ax = axs[0, 0]
    for obj in objects:
        loc = obj['location']
        dims = obj['dimensions']
        ry = obj['rotation_y']
        label = obj['type']
        x, y, z = loc
        z = -z  # convert to z-forward
        alpha = ry - np.arctan2(x, z)
        alpha = alpha % (2 * np.pi)
        alpha = alpha - 2 * np.pi if alpha > np.pi else alpha  # normalize to [-pi, pi]
        assert np.isclose(obj['alpha'], alpha, atol=1e-2) # 1e-2 because SDG saves :2f
        ax.plot(x, z, 'ro')
        ax.text(x, z, label, color='blue', fontsize=10, ha='center', va='bottom')
        arrow_length = 0.5
        dx = np.cos(ry) * arrow_length
        dz = np.sin(ry) * arrow_length
        ax.arrow(x, z, dx, dz, head_width=0.12, head_length=0.15, fc='m', ec='m', linewidth=2)
        h, w, l = dims
        # Correct order: front-right, front-left, rear-left, rear-right, close
        corners = np.array([
            [ l/2, -w/2],   # front-right
            [ l/2,  w/2],   # front-left
            [-l/2,  w/2],   # rear-left
            [-l/2, -w/2],   # rear-right
            [ l/2, -w/2],   # close rectangle
        ])
        R = np.array([
            [np.cos(ry), -np.sin(ry)],
            [np.sin(ry),  np.cos(ry)]
        ])
        bev_corners = (R @ corners.T).T + np.array([x, z])
        ax.plot(bev_corners[:,0], bev_corners[:,1], 'g-', linewidth=2)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-0.5, 5)
    ax.plot(0, 0, marker='^', color='black', markersize=14, label='Camera')
    ax.text(0, 0, 'Camera', color='black', fontsize=12, ha='right', va='bottom', fontweight='bold')
    cam_arrow_len = 1.0
    ax.arrow(0, 0, 0, cam_arrow_len, head_width=0.2, head_length=0.2, fc='navy', ec='navy', linewidth=3, zorder=5, label='Camera View')
    ax.text(0, cam_arrow_len+0.2, 'View', color='navy', fontsize=12, ha='center', va='bottom', fontweight='bold')
    ax.set_facecolor('#f7f7fa')
    ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
    ax.set_xlabel('x (right, m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('z (forward, m)', fontsize=12, fontweight='bold')
    ax.set_title('BEV', fontsize=14, fontweight='bold')
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='^', color='w', label='Camera', markerfacecolor='black', markersize=14),
        Line2D([0], [0], color='navy', lw=3, label='Camera View'),
        Line2D([0], [0], marker='o', color='w', label='Object Center', markerfacecolor='red', markersize=10),
        Line2D([0], [0], color='g', lw=2, label='Object BBox'),
        Line2D([0], [0], color='m', lw=2, label='Object Direction'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    # --- 2D BBox on RGB image subplot ---
    ax2 = axs[0, 1]
    img = mpimg.imread(rgb_path)
    ax2.imshow(img)
    for obj in objects:
        bbox = obj['bbox']  # [xmin, ymin, xmax, ymax]
        label = obj['type']
        occlusion = obj['occluded']
        truncation = obj['truncated']
        # Compose label with occlusion and truncation
        label_text = f"{label}\n({occlusion}, {truncation:.2f})"
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=2, edgecolor='lime', facecolor='none')
        ax2.add_patch(rect)
        ax2.text(
            bbox[0], bbox[1]-5, label_text,
            color='lime', fontsize=10, fontweight='bold', va='bottom', ha='left',
            bbox=dict(facecolor='black', alpha=0.3, edgecolor='none', pad=1)
        )
    ax2.set_title('2D Bounding Boxes (occlusion level, truncation ratio)', fontsize=14, fontweight='bold')
    ax2.axis('off')


    # --- Instance segmentation subplot (second row) ---
    ax4 = axs[1, 0]
    # Read 16-bit PNG
    import PIL.Image as Image
    inst_img = np.array(Image.open(instance_seg_path))

    instance_id = inst_img & 0xFF
    unique_ids = np.unique(instance_id)
    num_ids = len(unique_ids)

    import matplotlib
    from matplotlib.patches import Patch
    # Create a ListedColormap with as many colors as unique ids
    cmap = matplotlib.colormaps['tab20']
    colors = [cmap(i) for i in range(num_ids)]
    listed_cmap = matplotlib.colors.ListedColormap(colors)
    # Set boundaries so each unique id gets its own color
    boundaries = np.append(unique_ids, unique_ids[-1]+1)
    norm = matplotlib.colors.BoundaryNorm(boundaries, listed_cmap.N)
    # Legend for instance ids and their colors (first 10 for readability)
    legend_elements = []
    for i, id_ in enumerate(unique_ids[:10]):
        if id_ == 0:
            legend_elements.append(Patch(facecolor=colors[i], edgecolor='k', label=f'No Instance'))
            continue
        legend_elements.append(Patch(facecolor=colors[i], edgecolor='k', label=f'ID {id_}'))
    ax4.legend(handles=legend_elements, loc='lower right', fontsize=8, borderaxespad=0.)
    # Plot instance_id directly
    ax4.imshow(instance_id, cmap=listed_cmap, norm=norm, interpolation='nearest')
    ax4.set_title('Instance Segmentation', fontsize=14, fontweight='bold')
    ax4.axis('off')


    # --- Semantic segmentation subplot (second row) ---
    ax5 = axs[1, 1]
    semantic_id = (inst_img >> 8) & 0xFF
    # Read semantic mapping and build id-to-class mapping
    import matplotlib
    with open(mapping_path, 'r') as f:
        semantic_mapping = json.load(f)
    id_to_class = {v: k for k, v in semantic_mapping.items()}
    max_id = max(id_to_class.keys())
    num_classes = max_id + 1
    cmap = matplotlib.colormaps['tab20']
    colors = [cmap(i % cmap.N) for i in range(num_classes)]  # RGBA tuples
    # For semantic segmentation subplot
    unique_ids = np.unique(semantic_id)
    seg_colors = [colors[idx] for idx in unique_ids]
    listed_cmap = matplotlib.colors.ListedColormap(seg_colors)
    boundaries = np.append(unique_ids, unique_ids[-1]+1)
    norm = matplotlib.colors.BoundaryNorm(boundaries, listed_cmap.N)
    ax5.imshow(semantic_id, cmap=listed_cmap, norm=norm, interpolation='nearest')
    ax5.set_title('Semantic Segmentation', fontsize=14, fontweight='bold')
    ax5.axis('off')

    # --- Semantic mapping table subplot (second row, last column) ---
    table_data = []
    cell_colours = []
    for idx in range(num_classes):
        if idx not in id_to_class:
            continue
        class_name = id_to_class[idx]
        table_data.append([class_name, idx])
        rgb = colors[idx][:3]  # Only RGB, ignore alpha
        cell_colours.append([rgb, [1, 1, 1]])  # Color only class name cell
    n_cols = 2  # Number of columns you want
    n_items = len(table_data)
    n_rows = math.ceil(n_items / n_cols)
    # Pad table_data and cell_colours to fill the grid
    pad_len = n_rows * n_cols - n_items
    table_data_padded = table_data + [["", ""]] * pad_len
    cell_colours_padded = cell_colours + [[[1,1,1], [1,1,1]]] * pad_len
    # Reshape into grid (each row is a list of [class, id, class, id, ...])
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
    # Build column labels (no longer needed)
    # col_labels_multi = []
    # for i in range(n_cols):
    #     col_labels_multi += [f'Class Name {i+1}', f'ID {i+1}']
    ax6 = axs[1, 2]
    ax6.axis('off')
    table = ax6.table(
        cellText=table_grid,
        # colLabels=col_labels_multi,  # Remove column labels
        cellColours=colour_grid,
        loc='center',
        cellLoc='center',
        colLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    # Style grid as before (skip header row styling)
    for (row, col), cell in table.get_celld().items():
        # Highlight UNLABELLED row
        if row > 0 and any('UNLABELLED' == table_grid[row-1][c*2] for c in range(n_cols)):
            cell.set_edgecolor('red')
            cell.set_linewidth(2)
        cell.set_linewidth(0.7)
        cell.set_edgecolor('gray')
    table.auto_set_column_width(list(range(n_cols*2)))

    # axs[0, 2].axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", action="store", type=str, default="")
    parser.add_argument("--index", action="store", type=int, default=0)
    parser.add_argument("--debug", action="store_true", default=False)
    index = parser.parse_args().index
    base = f"{parser.parse_args().base}/cam_0"
    label_path = os.path.join(base, "object_detection", f"{index}.txt")
    rgb_path = os.path.join(base, "uw_rgb", f"{index}.png")
    instance_seg_path = os.path.join(base, "instance_segmentation", f"{index}.png")
    mapping_path = os.path.join(base, "semantic_mapping.json")
    debug = parser.parse_args().debug
    if debug == True:
        debug_path = os.path.join(base, "debug", f"{index}.png")

        plot_all_debug(label_path, rgb_path, debug_path, instance_seg_path, mapping_path)
    elif debug == False:
        plot_all(label_path, rgb_path, instance_seg_path, mapping_path)
