#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from collections import defaultdict
from typing import List, Tuple, Dict

import cv2
import numpy as np

from staff_processor import process_staff_windows, concat_staffs_map_aligned, concat_staff_lines_transparent_aligned


def load_image(path):
    """Loads an image from 'path' and returns it as a BGR image. Or None on error."""
    return cv2.imread(path)


def convert_to_black_and_white(bgr_image, threshold_value=128):
    """
    Converts a BGR color image to grayscale first and then
    then performs a binary conversion (only 0 or 255).
    With THRESH_BINARY_INV, dark pixels become white (255), light pixels black (0).
    """
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    _, bw_img = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
    return bw_img


def dilate_horizontally(bin_image, expansion=2):
    """
    Performs morphological dilation in the X direction only,
    to close small gaps in the lines.
    """
    kernel_width = 1 + 2 * expansion
    kernel = np.ones((1, kernel_width), dtype=np.uint8)
    return cv2.dilate(bin_image, kernel, iterations=1)


def compute_overlap(interval1, interval2):
    """
    interval1 = (x_start1, x_end1)
    interval2 = (x_start2, x_end2)
    Returns the number of overlapping pixels in the X direction (>= 0).
    """
    (s1, e1) = interval1
    (s2, e2) = interval2
    start = max(s1, s2)
    end = min(e1, e2)
    return max(0, end - start + 1)


def can_connect(lineA, lineB, min_overlap_pixels=5):
    """
    lineA, lineB = (y, x_start, x_end)
    Checks whether:
      1) |yA - yB| <= 1
      2) x-ranges overlap at least 'min_overlap_pixels'
    Returns True if lineA can connect directly to lineB.
    """
    (yA, xA_start, xA_end) = lineA
    (yB, xB_start, xB_end) = lineB
    if abs(yA - yB) > 1:
        return False
    overlap = compute_overlap((xA_start, xA_end), (xB_start, xB_end))
    if overlap < min_overlap_pixels:
        return False
    return True


def can_merge_with_group(new_line, group_lines, min_overlap_pixels=5):
    """
    new_line = (y, x_start, x_end)
    group_lines = List of lines [(y1, xs1, xe1), (y2, xs2, xe2), ...]
    Checks whether 'new_line' can connect to at least one line in group_lines.
    """
    for line in group_lines:
        if can_connect(new_line, line, min_overlap_pixels):
            return True
    return False


def find_horizontal_line_segments(dilated_image, min_line_length=100):
    """
    Searches each line for ALL sufficiently long sequences (value=255).
    Returns a list of segments (y, x_start, x_end).
    """
    height, width = dilated_image.shape
    segments = []
    for y in range(height):
        row = dilated_image[y, :]
        x_start = None
        for x in range(width):
            if row[x] == 255:
                if x_start is None:
                    x_start = x
            else:
                if x_start is not None:
                    length = x - x_start
                    if length >= min_line_length:
                        segments.append((y, x_start, x - 1))
                    x_start = None
        if x_start is not None:
            length = width - x_start
            if length >= min_line_length:
                segments.append((y, x_start, width - 1))
    return segments


def merge_lines_connect(segments, min_overlap_pixels=5):
    """
    Builds groups by having each line either
    joining an existing group (if can_merge_with_group==True)
    or forming a new group.
    """
    groups = []
    for line in segments:
        found_group = None
        for g in groups:
            if can_merge_with_group(line, g, min_overlap_pixels):
                g.append(line)
                found_group = g
                break
        if found_group is None:
            groups.append([line])
    return groups


def groups_can_connect(groupA, groupB, min_overlap_pixels=5):
    """
    Checks whether at least one line in groupA
    can connect to at least one line in groupB.
    """
    for lineA in groupA:
        for lineB in groupB:
            if can_connect(lineA, lineB, min_overlap_pixels):
                return True
    return False


def unify_groups_until_stable(groups, min_overlap_pixels=5):
    """
    Repeat the merging of groups until nothing changes.
    If two groups (groupA, groupB) can connect via at least one pair of lines,
    they are combined into a single group.
    """
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(groups):
            j = i + 1
            while j < len(groups):
                if groups_can_connect(groups[i], groups[j], min_overlap_pixels):
                    # Verbinde Gruppe j in Gruppe i
                    groups[i].extend(groups[j])
                    del groups[j]
                    changed = True
                    # Keine Erhöhung von j, da die Gruppe j gelöscht wurde
                else:
                    j += 1
            i += 1
    return groups


def highlight_groups_in_color_debug(bgr_image, groups):
    """
    groups: List of groups (list of lines).
    Each group is assigned a different color (rotation principle).
    Returns debug_info, which provides information about color and segments per group.
    """
    palette = [
        (0, 255, 0),  # Grün
        (255, 0, 0),  # Blau
        (0, 165, 255),  # Orange
        (0, 255, 255),  # Gelb
        (0, 0, 255)  # Rot
    ]
    num_colors = len(palette)

    debug_info = []
    for i, group in enumerate(groups):
        color = palette[i % num_colors]
        group_debug = {
            'group_index': i,
            'color': color,
            'segments': group
        }
        debug_info.append(group_debug)

        for (y, xs, xe) in group:
            bgr_image[y, xs:xe + 1] = color

    return debug_info


def extract_lines_as_transparent(original_bgr, staffs):
    """
    Creates a new image (BGRA) of the same size,
    sets everything to (0,0,0,0) and copies for each line in 'staffs'
    the original pixels with full opacity (A=255).

    Parameters:
    - original_bgr: Original image in BGR format (NumPy array)
    - staffs: List of stafflines, each staffline is a dictionary with:
        {
            'horizontal_groups': [horizontal_group1, horizontal_group2, ...],
            'vertical_groups': [vertical_group1, vertical_group2, ...] # Sorted by x ascending
        }
        Each horizontal group is a list of lines [(y, x_start, x_end), ...].
        Each vertical group is a list of lines [(x, y_start, y_end), ...].

    Return:
    - lines_img: New image (BGRA), only the lines are visible (alpha = 255), rest transparent (alpha = 0)
    """
    height, width, _ = original_bgr.shape
    # Erzeuge leeres Bild (BGRA), alles transparent
    lines_img = np.zeros((height, width, 4), dtype=np.uint8)

    for staff in staffs:
        # Färbe horizontale Gruppen
        for h_group in staff['horizontal_groups']:
            for (y, x_start, x_end) in h_group:
                # Sicherstellen, dass die Indizes innerhalb des Bildes liegen
                x_start_clipped = max(0, min(width - 1, x_start))
                x_end_clipped = max(0, min(width - 1, x_end))
                # Kopiere die Pixel der horizontalen Linie
                lines_img[y, x_start_clipped:x_end_clipped + 1, 0:3] = original_bgr[y, x_start_clipped:x_end_clipped + 1, 0:3]
                # Setze Alpha-Kanal auf 255 (voll sichtbar)
                lines_img[y, x_start_clipped:x_end_clipped + 1, 3] = 255

        # Färbe vertikale Gruppen
        for v_group in staff['vertical_groups']:
            for (x, y_start, y_end) in v_group:
                # Sicherstellen, dass die Indizes innerhalb des Bildes liegen
                y_start_clipped = max(0, min(height - 1, y_start))
                y_end_clipped = max(0, min(height - 1, y_end))
                # Kopiere die Pixel der vertikalen Linie
                lines_img[y_start_clipped:y_end_clipped + 1, x, 0:3] = original_bgr[y_start_clipped:y_end_clipped + 1, x, 0:3]
                # Setze Alpha-Kanal auf 255 (voll sichtbar)
                lines_img[y_start_clipped:y_end_clipped + 1, x, 3] = 255

    return lines_img



def filter_groups_by_x_thickness(groups, max_thickness=4, threshold=0.5):
    """
    Filters out groups where more than 'threshold' proportion of the x's have a thickness > 'max_thickness'.

    groups: List of groups, each group is a list of lines [(y, x_start, x_end), ...]
    max_thickness: Maximum allowed thickness per x
    threshold: Threshold value for the proportion of x's that are too thick (e.g. 0.5 for 50%)

    Return: filtered list of groups
    """
    filtered_groups = []
    for group in groups:
        # Bestimme den x-Bereich der Gruppe
        x_min = min(x_start for (_, x_start, _) in group)
        x_max = max(x_end for (_, _, x_end) in group)

        # Initialisiere ein Array für die x-Zählung
        x_counts = np.zeros(x_max - x_min + 1, dtype=int)

        for (_, x_start, x_end) in group:
            # Erhöhe den Zähler für jeden x in [x_start, x_end]
            x_counts[x_start - x_min: x_end - x_min + 1] += 1

        # Berechne den Anteil der x's mit Dicke > max_thickness
        thick_x = np.sum(x_counts > max_thickness)
        total_x = len(x_counts)
        proportion = thick_x / total_x if total_x > 0 else 0

        if proportion <= threshold:
            filtered_groups.append(group)
        else:
            print(f"Gruppe entfernt: Dicke in {proportion * 100:.1f}% der x's > {max_thickness}px")

    return filtered_groups


def filtered_groups_by_length(groups, min_group_length=250):
    """
    Filters out groups where more than 'threshold' proportion of the x's have a thickness > 'max_thickness'.

    groups: List of groups, each group is a list of lines [(y, x_start, x_end), ...]
    max_thickness: Maximum allowed thickness per x
    threshold: Threshold value for the proportion of x's that are too thick (e.g. 0.5 for 50%)

    Return: filtered list of groups
    """
    filtered_groups = []
    for group in groups:
        # Bestimme den x-Bereich der Gruppe
        x_min = min(x_start for (_, x_start, _) in group)
        x_max = max(x_end for (_, _, x_end) in group)

        if (x_max - x_min) >= min_group_length:
            filtered_groups.append(group)

    return filtered_groups


def dilate_vertically(bin_image, expansion=2):
    """
    Performs morphological dilation in the Y direction only,
    to close small gaps in the vertical lines.
    """
    kernel_height = 1 + 2 * expansion
    kernel = np.ones((kernel_height, 1), dtype=np.uint8)
    return cv2.dilate(bin_image, kernel, iterations=1)


def find_vertical_line_segments(dilated_image, min_line_length=100):
    """
    Searches each column for ALL sufficiently long sequences (value=255).
    Returns a list of segments (x, y_start, y_end).
    """
    height, width = dilated_image.shape
    segments = []
    for x in range(width):
        column = dilated_image[:, x]
        y_start = None
        for y in range(height):
            if column[y] == 255:
                if y_start is None:
                    y_start = y
            else:
                if y_start is not None:
                    length = y - y_start
                    if length >= min_line_length:
                        segments.append((x, y_start, y - 1))
                    y_start = None
        if y_start is not None:
            length = height - y_start
            if length >= min_line_length:
                segments.append((x, y_start, height - 1))
    return segments



def merge_vertical_lines_connect(segments, min_overlap_pixels=5):
    """
    Builds vertical groups by making each vertical line either
    joining an existing group (if can_merge_with_group_vertical==True)
    or forms a new group.
    """
    groups = []
    for line in segments:
        found_group = None
        for g in groups:
            if can_merge_with_group_vertical(line, g, min_overlap_pixels):
                g.append(line)
                found_group = g
                break
        if found_group is None:
            groups.append([line])
    return groups



def can_connect_vertical(lineA, lineB, min_overlap_pixels=5):
    """
    lineA, lineB = (x, y_start, y_end)
    Checks whether:
      1) |xA - xB| <= 1
      2) y-ranges overlap at least 'min_overlap_pixels'
    Returns True if lineA can connect directly to lineB.
    """
    (xA, yA_start, yA_end) = lineA
    (xB, yB_start, yB_end) = lineB
    if abs(xA - xB) > 1:
        return False
    overlap = compute_overlap((yA_start, yA_end), (yB_start, yB_end))
    if overlap < min_overlap_pixels:
        return False
    return True


def can_merge_with_group_vertical(new_line, group_lines, min_overlap_pixels=5):
    """
    new_line = (x, y_start, y_end)
    group_lines = List of vertical lines [(x1, y1_start, y1_end), ...]
    Checks whether 'new_line' can connect to at least one line in group_lines.
    """
    for line in group_lines:
        if can_connect_vertical(new_line, line, min_overlap_pixels):
            return True
    return False



def unify_vertical_groups_until_stable(groups, min_overlap_pixels=5):
    """
    Repeat the merging of vertical groups until nothing changes.
    If two groups (groupA, groupB) can connect via at least one pair of lines,
    they are combined into a single group.
    """
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(groups):
            j = i + 1
            while j < len(groups):
                if groups_can_connect_vertical(groups[i], groups[j], min_overlap_pixels):
                    groups[i].extend(groups[j])
                    del groups[j]
                    changed = True
                else:
                    j += 1
            i += 1
    return groups


def groups_can_connect_vertical(groupA, groupB, min_overlap_pixels=5):
    """
    Checks whether at least one vertical line in groupA
    can connect to at least one vertical line in groupB.
    """
    for lineA in groupA:
        for lineB in groupB:
            if can_connect_vertical(lineA, lineB, min_overlap_pixels):
                return True
    return False



def filter_vertical_groups(groups, max_thickness=4, thickness_threshold=0.5, min_group_length=100):
    """
    Filters vertical groups based on maximum thickness, thickness threshold and minimum length.

    Parameters:
    - groups: List of vertical groups, each group is a list of lines [(x, y_start, y_end), ...]
    - max_thickness: Maximum allowed thickness per y
    - thickness_threshold: Threshold value for the proportion of y's that are too thick (e.g. 0.5 for 50%)
    - min_group_length: Minimum length of the group based on (y_max - y_min)

    Return:
    - filtered list of vertical groups
    """
    filtered_groups = []
    for group in groups:
        # Bestimme den y-Bereich der Gruppe
        y_min = min(y_start for (_, y_start, _) in group)
        y_max = max(y_end for (_, _, y_end) in group)
        group_length = y_max - y_min

        # Initialisiere ein Array für die y-Zählung
        y_counts = np.zeros(y_max - y_min + 1, dtype=int)

        for (_, y_start, y_end) in group:
            # Erhöhe den Zähler für jeden y in [y_start, y_end]
            y_counts[y_start - y_min: y_end - y_min + 1] += 1

        # Berechne den Anteil der y's mit Dicke > max_thickness
        thick_y = np.sum(y_counts > max_thickness)
        total_y = len(y_counts)
        proportion = thick_y / total_y if total_y > 0 else 0

        # Überprüfe die Dicke und die Mindestlänge
        if proportion <= thickness_threshold and group_length >= min_group_length:
            filtered_groups.append(group)
        else:
            if proportion > thickness_threshold:
                print(f"Vertikale Gruppe entfernt: Dicke in {proportion * 100:.1f}% der y's > {max_thickness}px")
            if group_length < min_group_length:
                print(f"Vertikale Gruppe entfernt: Länge {group_length}px < Mindestlänge {min_group_length}px")

    return filtered_groups


def find_staffs(horizontal_groups: list, vertical_groups: list, min_horizontal_lines_per_staff: int = 10):
    """
    Groups horizontal and vertical groups into stafflines based on their intersections.
    Filters out stafflines that do not have enough horizontal line segments.

    Parameters:
    - horizontal_groups: List of horizontal groups, each group is a list of lines [(y, x_start, x_end), ...].
    - vertical_groups: List of vertical groups, each group is a list of lines [(x, y_start, y_end), ...].
    - min_horizontal_lines_per_staff: Minimum number of horizontal line segments,
                                      for a group to be considered a staff.

    Return:
    - List of staff lines, each staff line is a dictionary with:
        {
            'horizontal_groups': [horizontal_group1, horizontal_group2, ...],
            'vertical_groups': [vertical_group1, vertical_group2, ...] # Sorted by x ascending
        }
    """
    parent = {}

    def find(u):
        if u not in parent:
            parent[u] = u
            return u
        if parent[u] != u:
            parent[u] = find(parent[u])
        return parent[u]

    def union(u, v):
        if u not in parent: parent[u] = u
        if v not in parent: parent[v] = v

        u_root = find(u)
        v_root = find(v)
        if u_root != v_root:
            parent[v_root] = u_root

    H = len(horizontal_groups)
    V = len(vertical_groups)

    for i in range(H):
        parent[i] = i
    for i in range(V):
        parent[H + i] = H + i

    for h_idx, h_group in enumerate(horizontal_groups):
        if not h_group: continue

        for v_idx, v_group in enumerate(vertical_groups):
            if not v_group: continue

            intersects = False
            for (hy, hx_start, hx_end) in h_group:
                for (vx, vy_start, vy_end) in v_group:
                    if hx_start <= vx <= hx_end and vy_start <= hy <= vy_end:
                        intersects = True
                        break
                if intersects:
                    break

            if intersects:
                union(h_idx, H + v_idx)

    grouped_components = defaultdict(lambda: {'horizontal_groups': [], 'vertical_groups': [], 'num_h_segments': 0})

    for h_idx, h_group in enumerate(horizontal_groups):
        if not h_group: continue
        root = find(h_idx)
        grouped_components[root]['horizontal_groups'].append(h_group)
        grouped_components[root]['num_h_segments'] += len(h_group)  # Zähle die Anzahl der Segmente in dieser h_group

    for v_idx, v_group in enumerate(vertical_groups):
        if not v_group: continue
        if (H + v_idx) in parent:
            root = find(H + v_idx)
            if root in grouped_components:
                grouped_components[root]['vertical_groups'].append(v_group)

    staff_lines = []
    for root_key, component_data in grouped_components.items():
        total_h_segments_in_component = 0
        for h_g in component_data['horizontal_groups']:
            total_h_segments_in_component += len(h_g)

        if total_h_segments_in_component >= min_horizontal_lines_per_staff:

            valid_vertical_groups = [vg for vg in component_data['vertical_groups'] if vg]  # Entferne leere v_groups

            if valid_vertical_groups:
                try:
                    sorted_vertical = sorted(valid_vertical_groups,
                                             key=lambda vg_list: vg_list[0][0] if vg_list and vg_list[0] else float(
                                                 'inf'))
                except IndexError:
                    print(
                        f"Warnung: Indexfehler beim Sortieren von vertical_groups für root {root_key}. Vertikale Gruppen könnten unsortiert sein.")
                    sorted_vertical = valid_vertical_groups
            else:
                sorted_vertical = []

            staff_line = {
                'horizontal_groups': component_data['horizontal_groups'],
                'vertical_groups': sorted_vertical
            }
            staff_lines.append(staff_line)
    return staff_lines



def group_staff_lines(horizontal_groups, vertical_groups, intersections):
    """
    Groups horizontal and vertical groups into staves based on intersections.

    Parameter:
    - horizontal_groups: List of horizontal groups
    - vertical_groups: List of vertical groups
    - intersections: List of tuples (horizontal_group_index, vertical_group_index)

    Return:
    - List of note rows, each note row is a tuple (horizontal_group, [left_vertical_groups], [right_vertical_groups])
    """
    staff_lines = []
    h_to_v = {}
    for (h_idx, v_idx) in intersections:
        if h_idx not in h_to_v:
            h_to_v[h_idx] = set()
        h_to_v[h_idx].add(v_idx)

    for h_idx, h_group in enumerate(horizontal_groups):
        if h_idx in h_to_v:
            v_indices = list(h_to_v[h_idx])
            left_v = min(v_indices)
            right_v = max(v_indices)
            staff_lines.append((h_group, [left_v], [right_v]))
        else:
            staff_lines.append((h_group, [], []))
    return staff_lines


def export_staff_pixel_images(
    original_img: np.ndarray,
    staff_pixel_map: Dict[int, Dict[str, List[Tuple[int, int]]]],
    output_dir: str = "staff_pixel_exports"
):
    """
    Generates several images per staff to visualize the pixel assignment:
    - Top pixel only (black on white)
    - Bottom pixels only (black on white)
    - All assigned pixels (top + bottom) (black on white)
    - Only the “middle” pixels (pixels in the original image between min/max Y of the
      assigned pixels, but not top/bottom itself) (original color on white)
    - All assigned pixels (black) + middle pixels (original color) on white background

    Saves them as PNG in the specified directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    height, width, channels = original_img.shape
    print(f"Exporting staff pixel images to directory: {output_dir}")

    # Definiere Farben
    color_white = (255, 255, 255)
    color_black = (0, 0, 0)

    def create_blank_image():
        return np.full((height, width, channels), color_white, dtype=np.uint8)

    def draw_binary_pixels(img: np.ndarray, pixels: List[Tuple[int, int]], color: Tuple[int, int, int]):
        count = 0
        for (x, y) in pixels:
            if 0 <= y < height and 0 <= x < width:
                img[y, x] = color
                count += 1
        return count

    def copy_original_pixels(target_img: np.ndarray, source_img: np.ndarray, pixels: List[Tuple[int, int]]):
        count = 0
        for (x, y) in pixels:
             if 0 <= y < height and 0 <= x < width:
                 target_img[y, x] = source_img[y, x]
                 count += 1
        return count

    for idx, entry in staff_pixel_map.items():

        print(f"Processing Staff Index {idx}...")
        top_pixels = entry.get("Top", [])
        bot_pixels = entry.get("Bottom", [])
        all_assigned_pixels = top_pixels + bot_pixels

        if not all_assigned_pixels:
            print(f"  Staff {idx} has no assigned pixels. Skipping image generation.")
            continue

        img_top = create_blank_image()
        draw_binary_pixels(img_top, top_pixels, color_black)
        cv2.imwrite(os.path.join(output_dir, f"staff_{idx}_top.png"), img_top)

        img_bottom = create_blank_image()
        draw_binary_pixels(img_bottom, bot_pixels, color_black)
        cv2.imwrite(os.path.join(output_dir, f"staff_{idx}_bottom.png"), img_bottom)

        img_all_assigned = create_blank_image()
        draw_binary_pixels(img_all_assigned, all_assigned_pixels, color_black)
        cv2.imwrite(os.path.join(output_dir, f"staff_{idx}_all_assigned.png"), img_all_assigned)

        middle_pixels_coords = []
        if top_pixels and bot_pixels:
            all_assigned_coords_set = set(all_assigned_pixels)

            _, ys = zip(*all_assigned_pixels)
            min_y_assigned, max_y_assigned = min(ys), max(ys)

            print(f"  Vertical range for middle pixels (based on assigned): y=[{min_y_assigned}, {max_y_assigned}]")

            for y in range(min_y_assigned, max_y_assigned + 1):
                for x in range(width):
                    coord = (x, y)
                    if coord not in all_assigned_coords_set:
                        middle_pixels_coords.append(coord)

            print(f"  Found {len(middle_pixels_coords)} potential 'middle' pixel coordinates between y={min_y_assigned} and y={max_y_assigned}.")

            img_middle = create_blank_image()
            copied_middle_count = copy_original_pixels(img_middle, original_img, middle_pixels_coords)
            cv2.imwrite(os.path.join(output_dir, f"staff_{idx}_middle_original_color.png"), img_middle)
            print(f"  Saved middle pixels (original color): {copied_middle_count} pixels")

            img_combined = create_blank_image()
            copy_original_pixels(img_combined, original_img, middle_pixels_coords)
            drawn_assigned_count = draw_binary_pixels(img_combined, all_assigned_pixels, color_black)
            cv2.imwrite(os.path.join(output_dir, f"staff_{idx}_all_plus_middle.png"), img_combined)
            print(f"  Saved combined (assigned black + middle original): {drawn_assigned_count} assigned, {copied_middle_count} middle")

        else:
            print(f"  Skipping 'middle' and 'all_plus_middle' image generation for Staff {idx} because Top or Bottom pixels are missing.")

    print("\nFinished exporting staff pixel images.")


def process_image_to_staff_strips(
        input_path: str,
        output_dir_for_page: str,
        params: dict,
        staff_counter_global: int
):
    """
    Verarbeitet ein einzelnes Bild, um Notensystem-Streifen zu extrahieren.
    Speichert seiten-spezifische Debug-Bilder und gibt die konkatenierten Streifen zurück.
    """
    print(f"\nProcessing image: {input_path}")
    os.makedirs(output_dir_for_page, exist_ok=True)

    # Dateinamen für seiten-spezifische Ausgaben
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path_no_merge = os.path.join(output_dir_for_page, f"{base_name}_no_merge.png")
    output_path_merged = os.path.join(output_dir_for_page, f"{base_name}_merged_colorful.png")
    output_path_extracted = os.path.join(output_dir_for_page, f"{base_name}_extracted_lines.png")
    output_path_concat_aligned_page = os.path.join(output_dir_for_page, f"{base_name}_staffs_concat_aligned.png")
    output_path_concat_lines_transparent_page = os.path.join(output_dir_for_page,
                                                             f"{base_name}_staffs_concat_lines_transparent.png")
    output_dir_pixels_page = os.path.join(output_dir_for_page, "staff_pixel_exports")

    page_metadata = {
        "aligned_strip_first_staff_eff_start_x_in_strip": 0,
        "aligned_strip_last_staff_eff_end_x_in_strip": 0,
        "aligned_strip_center_y": 0,  # NEU
        "transparent_strip_first_staff_eff_start_x_in_strip": 0,
        "transparent_strip_last_staff_eff_end_x_in_strip": 0,
        "transparent_strip_center_y": 0,  # NEU
    }



    # 2) Bild laden, binär machen & dilatieren
    original_img = load_image(input_path)
    if original_img is None:
        print(f"Fehler beim Laden von {input_path}.")
        return None, None, 0  # Wichtig: Rückgabe bei Fehler

    bw_img = convert_to_black_and_white(original_img, threshold_value=params['threshold_value'])
    dilated_h = dilate_horizontally(bw_img, expansion=params['horizontal_expansion'])
    dilated_v = dilate_vertically(bw_img, expansion=params['vertical_expansion'])

    # 3) Segmente finden
    horizontal_segments = find_horizontal_line_segments(dilated_h, min_line_length=params['min_line_length_horizontal'])
    print(f"  Gefundene horizontale Segmente: {len(horizontal_segments)}")

    vertical_segments = find_vertical_line_segments(dilated_v, min_line_length=params['min_line_length_vertical'])
    print(f"  Gefundene vertikale Segmente: {len(vertical_segments)}")

    # 4) Ohne Merging (optional speichern)
    if params.get('save_debug_no_merge', False):
        no_merge_img = original_img.copy()
        for (y, x_start, x_end) in horizontal_segments:
            no_merge_img[y, x_start:x_end + 1] = (0, 255, 255)
        for (x, y_start, y_end) in vertical_segments:
            no_merge_img[y_start:y_end + 1, x] = (255, 0, 0)
        cv2.imwrite(output_path_no_merge, no_merge_img)
        print(f"  -> {output_path_no_merge}")

    # 5) Mergen in einem Durchlauf
    horizontal_groups_1 = merge_lines_connect(horizontal_segments, min_overlap_pixels=5)
    vertical_groups_1 = merge_vertical_lines_connect(vertical_segments, min_overlap_pixels=5)

    # 6) Mehrfach-Fusion, bis stabil
    final_horizontal_groups = unify_groups_until_stable(horizontal_groups_1, min_overlap_pixels=5)
    final_vertical_groups = unify_vertical_groups_until_stable(vertical_groups_1, min_overlap_pixels=5)

    print(f"  Gruppen nach unify (horizontal): {len(final_horizontal_groups)}")
    print(f"  Gruppen nach unify (vertikal): {len(final_vertical_groups)}")

    # 7) Filtern der zu dicken und zu kurzen Gruppen
    filtered_horizontal_groups = filter_groups_by_x_thickness(
        final_horizontal_groups,
        max_thickness=params['max_thickness'],
        threshold=params['thickness_threshold']
    )
    filtered_horizontal_groups = filtered_groups_by_length(
        filtered_horizontal_groups,
        min_group_length=params['min_group_length_horizontal']
    )
    print(f"  Gruppen nach Filtern (horizontal): {len(filtered_horizontal_groups)}")

    # final_vertical_groups = filter_vertical_groups( # Diese Funktion gibt es in deinem Code
    #     final_vertical_groups,
    #     max_thickness=params['max_thickness'], # Annahme: gleiche Dicke für vertikal
    #     thickness_threshold=params['thickness_threshold'],
    #     min_group_length=params['min_group_length_vertical']
    # )
    # print(f"  Gruppen nach Filtern (vertikal): {len(final_vertical_groups)}")

    # 8) Staffs finden
    staffs = find_staffs(filtered_horizontal_groups, final_vertical_groups)
    if not staffs:
        print(f"  WARNUNG: Keine Staffs gefunden für {input_path}. Überspringe weitere Verarbeitung dieser Seite.")
        return None, None, page_metadata, 0  # Gebe auch leere Metadaten zurück
    print(f"  Gefundene Staff-Systeme: {len(staffs)}")

    staffs_current_page = len(staffs)

    # 9) Ergebnis einfärben (optional speichern)
    if params.get('save_debug_merged_colorful', False):
        merged_img = original_img.copy()
        for group in filtered_horizontal_groups:
            for (y, x_start, x_end) in group:
                merged_img[y, x_start:x_end + 1] = (0, 255, 255)
        for group in final_vertical_groups:  # Hier vielleicht die gefilterten vertikalen nehmen
            for (x, y_start, y_end) in group:
                merged_img[y_start:y_end + 1, x] = (255, 0, 0)
        cv2.imwrite(output_path_merged, merged_img)
        print(f"  -> {output_path_merged}")

    # 10) Process Staff Windows to get pixel assignments
    staffs_with_map = process_staff_windows(
        original_img,
        staffs,
        threshold_value=params['threshold_value']
    )

    # 11) Export Pixel Images (Debug, optional)
    if params.get('save_debug_pixel_exports', False):
        export_staff_pixel_images(original_img, staffs_with_map, output_dir=output_dir_pixels_page)
        print(f"  Pixel-Export -> {output_dir_pixels_page}")

    concat_aligned, staff_data_for_aligned_metadata, aligned_strip_center_y = concat_staffs_map_aligned(  # Angepasste Rückgabe
        original_img,
        staffs,
        staffs_with_map,
        cut_left_while_copy=params['cut_left_while_copy'],
        staff_index_offset=staff_counter_global,
        pad_top=params['pad_top'],
        pad_bottom=params['pad_bottom'],
        horizontal_gap=params.get('horizontal_gap_aligned', 1),
        envelope_fallback_offset=params.get('envelope_fallback_offset', 10),
    )
    if concat_aligned is not None and concat_aligned.size > 0:
        cv2.imwrite(output_path_concat_aligned_page, concat_aligned)
        print(f"  -> {output_path_concat_aligned_page}")
        if staff_data_for_aligned_metadata:  # Wenn Metadaten zurückgegeben wurden
            page_metadata["aligned_strip_first_staff_eff_start_x_in_strip"] = staff_data_for_aligned_metadata[0][
                'x_placement']
            page_metadata["aligned_strip_last_staff_eff_end_x_in_strip"] = staff_data_for_aligned_metadata[-1][
                                                                               'x_placement'] + \
                                                                           staff_data_for_aligned_metadata[-1]['width']
            page_metadata["aligned_strip_center_y"] = aligned_strip_center_y  # Speichere y-Metadaten
    else:
        print(f"  WARNUNG: concat_aligned für {input_path} ist leer oder None.")
        concat_aligned = None

    # 13) Extrahierte Linien (optional speichern)
    if params.get('save_debug_extracted_lines', False):
        lines_extracted = extract_lines_as_transparent(original_img, staffs)
        cv2.imwrite(output_path_extracted, lines_extracted)
        print(f"  -> {output_path_extracted}")

    concat_lines_transparent_img, staff_data_for_transparent_metadata, transparent_strip_center_y = concat_staff_lines_transparent_aligned(
        # Angepasste Rückgabe
        original_img,
        staffs,
        staffs_with_map,
        cut_left_while_copy=params['cut_left_while_copy'],
        staff_index_offset=staff_counter_global,
        pad_top=params['pad_top'],
        pad_bottom=params['pad_bottom'],
        horizontal_gap=params.get('horizontal_gap_transparent', 1),
        apply_h_lowpass=params.get('apply_h_lowpass', True),
        h_lowpass_kernel_x=params.get('h_lowpass_kernel_x', 20)
    )
    if concat_lines_transparent_img is not None and concat_lines_transparent_img.size > 0:
        cv2.imwrite(output_path_concat_lines_transparent_page, concat_lines_transparent_img)
        print(f"  -> {output_path_concat_lines_transparent_page}")
        if staff_data_for_transparent_metadata:
            page_metadata["transparent_strip_first_staff_eff_start_x_in_strip"] = \
            staff_data_for_transparent_metadata[0]['x_placement']
            page_metadata["transparent_strip_last_staff_eff_end_x_in_strip"] = staff_data_for_transparent_metadata[-1][
                                                                                   'x_placement'] + \
                                                                               staff_data_for_transparent_metadata[-1][
                                                                                   'width']
            page_metadata["transparent_strip_center_y"] = transparent_strip_center_y  # Speichere y-Metadaten

    else:
        print(f"  WARNUNG: concat_lines_transparent_img für {input_path} ist leer oder None.")
        concat_lines_transparent_img = None

    return (concat_aligned,
            concat_lines_transparent_img,
            page_metadata,
            staffs_current_page)


if __name__ == "__main__":
    DEFAULT_PARAMS = {
        'threshold_value': 245,
        'horizontal_expansion': 4,
        'vertical_expansion': 4,
        'min_line_length_horizontal': 250,
        'min_line_length_vertical': 150,
        'max_thickness': 8,
        'thickness_threshold': 0.5,
        'min_group_length_horizontal': 500,
        'min_group_length_vertical': 500,
        'cut_left_while_copy': 118,
        'pad_top': 10,
        'pad_bottom': 10,
        'horizontal_gap_aligned': 1,
        'middle_overlap_aligned': 10,
        'horizontal_gap_transparent': 1,
        'apply_h_lowpass': True,
        'h_lowpass_kernel_x': 20,
        'save_debug_no_merge': True,
        'save_debug_merged_colorful': True,
        'save_debug_extracted_lines': True,
        'save_debug_pixel_exports': True,
    }

    # Erstelle Test-Input/Output Verzeichnisse falls nicht vorhanden
    test_input_dir = "test_input_images"
    test_output_dir = "test_single_output"
    os.makedirs(test_input_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)

    example_image_name = "2.png"
    example_image_path = os.path.join(test_input_dir, example_image_name)

    if not os.path.exists(example_image_path):
        print(f"Bitte lege ein Testbild unter {example_image_path} ab.")
    else:
        # Testaufruf
        output_page_dir = os.path.join(test_output_dir, os.path.splitext(example_image_name)[0])
        aligned_strip, transparent_strip, meta = process_image_to_staff_strips(
            example_image_path,
            output_page_dir,
            DEFAULT_PARAMS
        )

        if aligned_strip is not None:
            print(f"Test: Aligned strip shape: {aligned_strip.shape}")
        if transparent_strip is not None:
            print(f"Test: Transparent strip shape: {transparent_strip.shape}")
        print(f"Einzeltest abgeschlossen. Ergebnisse in {output_page_dir}")
