import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import json
import cv2
import os
import matplotlib
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


def load_semantic_mapping(mapping_path):
    with open(mapping_path, 'r') as f:
        label_to_color = json.load(f)
    label_order = list(label_to_color.keys())
    index_to_label = {}
    for idx, label in enumerate(label_order):
        index_to_label[idx] = label
    return index_to_label, label_to_color


def plot_all_debug(label_path, rgb_path, debug_path, instance_seg_path, mapping_path):
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
        alpha = ry - np.atan2(x, z)
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
    index_to_label, label_to_color = load_semantic_mapping(mapping_path)
    unique_ids = np.unique(semantic_id)
    # Build the color list in the order of unique_ids
    colors = [np.array(label_to_color[index_to_label[idx]]) / 255.0 for idx in unique_ids]
    listed_cmap = matplotlib.colors.ListedColormap(colors)
    # Set boundaries so each unique id gets its own color
    boundaries = np.append(unique_ids, unique_ids[-1]+1)
    norm = matplotlib.colors.BoundaryNorm(boundaries, listed_cmap.N)
    ax5.imshow(semantic_id, cmap=listed_cmap, norm=norm, interpolation='nearest')
    ax5.set_title('Semantic Segmentation', fontsize=14, fontweight='bold')
    ax5.axis('off')

    ax6 = axs[1, 2]
    labels = list(label_to_color.keys())
    colors = [np.array(label_to_color[label]) / 255.0 for label in labels]
    n_labels = len(labels)
    n_cols = 3  # Number of columns in the table
    n_rows = int(np.ceil(n_labels / n_cols))

    # Prepare cell text and colors for the table
    cell_text = []
    cell_colors = []
    for row in range(n_rows):
        row_text = []
        row_colors = []
        for col in range(n_cols):
            idx = row * n_cols + col
            if idx < n_labels:
                row_text.append(labels[idx])
                row_colors.append(colors[idx])
            else:
                row_text.append("")  # Empty cell
                row_colors.append([1, 1, 1, 0])  # Transparent/white
        cell_text.append(row_text)
        cell_colors.append(row_colors)

    table = ax6.table(cellText=cell_text,
                      cellColours=cell_colors,
                    #   colLabels=[f'Label {i+1}' for i in range(n_cols)],
                      loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    ax6.set_title('Semantic Mapping', fontsize=14, fontweight='bold')
    ax6.axis('off')

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
        alpha = ry - np.atan2(x, z)
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
    index_to_label, label_to_color = load_semantic_mapping(mapping_path)
    unique_ids = np.unique(semantic_id)
    # Build the color list in the order of unique_ids
    colors = [np.array(label_to_color[index_to_label[idx]]) / 255.0 for idx in unique_ids]
    listed_cmap = matplotlib.colors.ListedColormap(colors)
    # Set boundaries so each unique id gets its own color
    boundaries = np.append(unique_ids, unique_ids[-1]+1)
    norm = matplotlib.colors.BoundaryNorm(boundaries, listed_cmap.N)
    ax5.imshow(semantic_id, cmap=listed_cmap, norm=norm, interpolation='nearest')
    ax5.set_title('Semantic Segmentation', fontsize=14, fontweight='bold')
    ax5.axis('off')

    ax6 = axs[1, 2]
    labels = list(label_to_color.keys())
    colors = [np.array(label_to_color[label]) / 255.0 for label in labels]
    n_labels = len(labels)
    n_cols = 3  # Number of columns in the table
    n_rows = int(np.ceil(n_labels / n_cols))

    # Prepare cell text and colors for the table
    cell_text = []
    cell_colors = []
    for row in range(n_rows):
        row_text = []
        row_colors = []
        for col in range(n_cols):
            idx = row * n_cols + col
            if idx < n_labels:
                row_text.append(labels[idx])
                row_colors.append(colors[idx])
            else:
                row_text.append("")  # Empty cell
                row_colors.append([1, 1, 1, 0])  # Transparent/white
        cell_text.append(row_text)
        cell_colors.append(row_colors)

    table = ax6.table(cellText=cell_text,
                      cellColours=cell_colors,
                      loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    ax6.set_title('Semantic Mapping', fontsize=14, fontweight='bold')
    ax6.axis('off')

    axs[0, 2].axis('off')
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
    mapping_path = os.path.join(base, "semantic_segmentation", "semantic_mapping.json")
    debug = parser.parse_args().debug
    if debug == True:
        debug_path = os.path.join(base, "debug", f"{index}.png")

        plot_all_debug(label_path, rgb_path, debug_path, instance_seg_path, mapping_path)
    elif debug == False:
        plot_all(label_path, rgb_path, instance_seg_path, mapping_path)
