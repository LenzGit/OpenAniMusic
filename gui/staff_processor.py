import cv2
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict
import networkx as nx
import math

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ConnectedRegion:
    pixels: List[Tuple[int, int]]  # Liste von (x, y) Tupeln
    min_x: int
    max_x: int
    min_y: int
    max_y: int
    min_x_coord: Tuple[int, int]
    max_x_coord: Tuple[int, int]
    min_y_coord: Tuple[int, int]
    max_y_coord: Tuple[int, int]
    is_staff: bool = field(default=False)  # <--- Neues Flag

    def __init__(self, pixels: List[Tuple[int, int]], staff_type: str, is_staff: bool = False):
        self.pixels = pixels
        self.staff_type = staff_type
        self.is_staff = is_staff
        self.min_x = min(p[0] for p in pixels)
        self.max_x = max(p[0] for p in pixels)
        self.min_y = min(p[1] for p in pixels)
        self.max_y = max(p[1] for p in pixels)
        self.min_x_coord = next((p for p in pixels if p[0] == self.min_x), (self.min_x, self.min_y))
        self.max_x_coord = next((p for p in pixels if p[0] == self.max_x), (self.max_x, self.max_y))
        self.min_y_coord = next((p for p in pixels if p[1] == self.min_y), (self.min_x, self.min_y))
        self.max_y_coord = next((p for p in pixels if p[1] == self.max_y), (self.max_x, self.max_y))


def sort_staffs_top_to_bottom(staffs: List[dict]) -> List[dict]:
    """
    Sortiert die Staffs von oben nach unten basierend auf der durchschnittlichen y-Position ihrer horizontalen Linien.

    Parameter:
    - staffs: Liste von Staffs, jede Staff ist ein Dictionary mit 'horizontal_groups' und 'vertical_groups'

    Rückgabe:
    - Sortierte Liste von Staffs
    """

    def average_y(staff):
        y_values = []
        for h_group in staff['horizontal_groups']:
            for (y, _, _) in h_group:
                y_values.append(y)
        return np.mean(y_values) if y_values else 0

    sorted_staffs = sorted(staffs, key=average_y)
    return sorted_staffs


def define_window_between_staffs(upper_staff: dict, lower_staff: dict, image_width: int) -> Tuple[int, int, int, int]:
    """
    Definiert ein rechteckiges Fenster zwischen zwei Staffs basierend auf den extremen Vertikalen der Staffs.

    Parameter:
    - upper_staff: Dictionary mit 'horizontal_groups' und 'vertical_groups'
    - lower_staff: Dictionary mit 'horizontal_groups' und 'vertical_groups'
    - image_width: Breite des Bildes

    Rückgabe:
    - Tuple mit (x_min, y_min, x_max, y_max)
    """
    # Linkeste Vertikale der oberen Staff
    upper_left_v = min(upper_staff['vertical_groups'], key=lambda vg: min(line[0] for line in vg))
    upper_left_x = min(line[0] for line in upper_left_v)
    upper_left_y = max(line[2] for line in upper_left_v)  # Unteres Ende der oberen Staff

    # Rechteste Vertikale der oberen Staff
    upper_right_v = max(upper_staff['vertical_groups'], key=lambda vg: max(line[0] for line in vg))
    upper_right_x = max(line[0] for line in upper_right_v)
    upper_right_y = max(line[2] for line in upper_right_v)  # Unteres Ende der oberen Staff

    # Linkeste Vertikale der unteren Staff
    lower_left_v = min(lower_staff['vertical_groups'], key=lambda vg: min(line[0] for line in vg))
    lower_left_x = min(line[0] for line in lower_left_v)
    lower_left_y = min(line[1] for line in lower_left_v)  # Oberes Ende der unteren Staff

    # Rechteste Vertikale der unteren Staff
    lower_right_v = max(lower_staff['vertical_groups'], key=lambda vg: max(line[0] for line in vg))
    lower_right_x = max(line[0] for line in lower_right_v)
    lower_right_y = min(line[1] for line in lower_right_v)  # Oberes Ende der unteren Staff

    # Definiere top_y als das untere Ende der oberen Staff
    top_y = max(upper_left_y, upper_right_y)

    # Definiere bottom_y als das obere Ende der unteren Staff
    bottom_y = min(lower_left_y, lower_right_y)

    # x-Werte sind das gesamte Bild
    x_min = 0
    x_max = image_width - 1

    return x_min, top_y, x_max, bottom_y


def extract_connected_regions(binary_image: np.ndarray, debug: bool = False) -> List[ConnectedRegion]:
    """
    Findet zusammenhängende schwarze Gebiete in einem binären Bild.

    Parameter:
    - binary_image: Binäres Bild (0 und 255)
    - debug: Wenn True, zeigt das Binärbild an

    Rückgabe:
    - Liste von ConnectedRegion Objekten
    """
    if debug:
        # Zeige das Binärbild an
        cv2.imshow("Binary Image", binary_image)
        cv2.waitKey(0)  # Warte auf einen Tastendruck
        cv2.destroyAllWindows()

    # Verwende OpenCV's Connected Components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

    regions = []
    for label in range(1, num_labels):  # Label 0 ist der Hintergrund
        mask = labels == label
        ys, xs = np.where(mask)
        pixels = list(zip(xs, ys))
        if pixels:
            region = ConnectedRegion(pixels, "None", False)
            regions.append(region)

    return regions


def segment_graph(graph: nx.Graph):
    # 1) Kopie des Graphen anlegen, um ihn zu modifizieren:
    copy_graph = graph.copy()

    # 2) Staff-Knoten ermitteln (top/bottom):
    all_vertices = list(copy_graph.nodes())
    staff_nodes = [
        v for v in all_vertices
        if copy_graph.nodes[v]['region'].is_staff
    ]

    # Liste aktiver Knoten = alle anderen (noch nicht staff)
    active_vertices = [
        v for v in all_vertices
        if v not in staff_nodes
    ]

    # 3) Solange aktive Knoten vorhanden sind
    while active_vertices:
        # a) Erzeuge merge_vertices
        merge_vertices = []
        # b) Seed-Knoten (erster aus active_vertices)
        seed = active_vertices.pop(0)
        merge_vertices.append(seed)

        # c) Schleife, bis staff gefunden in merge_vertices
        while True:
            # i) Prüfen, ob staff-Knoten in merge_vertices
            found_staff_type = "None"
            for mv in merge_vertices:
                st_type = copy_graph.nodes[mv]['region'].staff_type
                if st_type != "None":
                    # staff_type gefunden
                    found_staff_type = st_type
                    break

            if found_staff_type != "None":
                # -> Vererbe staff_type an alle Knoten in merge_vertices
                for mv in merge_vertices:
                    copy_graph.nodes[mv]['region'].staff_type = found_staff_type
                # Merging für diese Gruppe abgeschlossen
                break

            # ii) Finde unter active_vertices den Knoten mit kleinster direkter Distanz
            candidate = None
            min_dist = float('inf')

            for v in copy_graph.nodes():
                if v in merge_vertices:
                    continue  # Überspringen, da wir ihn nicht nochmal nehmen wollen

                # Prüfe jede Kante v->(any in merge_vertices)
                for mv in merge_vertices:
                    if copy_graph.has_edge(v, mv):
                        w = copy_graph[v][mv].get('weight', 1.0)
                        if w < min_dist:
                            min_dist = w
                            candidate = v

            if candidate is None:
                # Keine Verbindung mehr => kein staff => wir brechen ab
                break

            # iii) Entferne candidate aus active_vertices, falls candidate ein Staff ist, mache nichts
            if not copy_graph.nodes[candidate]['region'].is_staff:
                if candidate in active_vertices:
                    active_vertices.remove(candidate)

            # iv) Entferne Kanten candidate <-> merge_vertices
            for mv in merge_vertices:
                if copy_graph.has_edge(candidate, mv):
                    copy_graph.remove_edge(candidate, mv)
                if copy_graph.has_edge(mv, candidate):
                    copy_graph.remove_edge(mv, candidate)

            # v) Prüfe candidate.staff_type
            c_type = copy_graph.nodes[candidate]['region'].staff_type

            if c_type != "None":
                # Vererbe c_type an alle in merge_vertices
                for mv in merge_vertices:
                    copy_graph.nodes[mv]['region'].staff_type = c_type
                # -> Wir sind fertig mit dieser Gruppe
                break
            else:
                # candidate hat keinen staff_type -> mergen
                merge_vertices.append(candidate)

                # ggf. weitere Kanten vom candidate entfernen?
                # z.B. kopieren:
                # for nb in list(copy_graph.adj[candidate]):
                #     copy_graph.remove_edge(candidate, nb)

            # Falls keinerlei Knoten mehr übrig sind, können wir nicht weiter mergen
            if not active_vertices:
                break

    # 4) Gib den veränderten copy_graph zurück
    return copy_graph


def process_staff_windows_old(original_img: np.ndarray, staffs: List[dict], threshold_value: int = 245) -> List[ConnectedRegion]:
    sorted_staffs = sort_staffs_top_to_bottom(staffs)
    print(f"Staffs wurden von oben nach unten sortiert: {len(sorted_staffs)} Staffs gefunden.")

    windows = []
    regions = []
    image_width = original_img.shape[1]

    for i in range(len(sorted_staffs) - 1):
        upper_staff = sorted_staffs[i]
        lower_staff = sorted_staffs[i + 1]
        window_coords = define_window_between_staffs(upper_staff, lower_staff, image_width)
        windows.append(window_coords)

    print(f"Definierte Fenster zwischen Staffs: {len(windows)}")

    for idx, (x_min, y_min, x_max, y_max) in enumerate(windows):
        print(f"Verarbeite Fenster {idx + 1}: ({x_min}, {y_min}, {x_max}, {y_max})")
        window_img = original_img[y_min:y_max, x_min:x_max]
        gray_window = cv2.cvtColor(window_img, cv2.COLOR_BGR2GRAY)
        # Verwende THRESH_BINARY, um Linien schwarz zu lassen und Hintergrund weiß
        _, binary_window = cv2.threshold(gray_window, threshold_value, 255, cv2.THRESH_BINARY)
        # Invertiere das Bild, damit die Linien weiß sind und der Hintergrund schwarz
        binary_window_inv = cv2.bitwise_not(binary_window)

        connected_regions = extract_connected_regions(binary_window_inv, False)

        print(f"  Gefundene zusammenhängende Regionen: {len(connected_regions)}")

        graph, all_regions = build_fully_connected_graph(connected_regions, x_min, x_max, y_min, y_max)

        segmented_graph = segment_graph(graph)

        # 1) Mache eine Kopie des aktuellen Fensters (so wie du oben debug_window_img angelegt hast)
        #    In deinem Fall kannst du es z.B. so nennen:
        debug_window_img = original_img[y_min:y_max, x_min:x_max]
        print(debug_window_img.shape)

        colored_count = 0
        # 2) Iteriere über alle Knoten im Graphen
        for node, data in segmented_graph.nodes(data=True):
            region = data['region']  # Dein ConnectedRegion-Objekt

            # Überspringe Staff-Knoten
            if region.is_staff:
                continue

            stype = region.staff_type

            # Bestimme die Farbe basierend auf staff_type
            if stype == "Bottom":
                color = (0, 255, 0)  # Grün (BGR)
            elif stype == "Top":
                color = (255, 0, 0)  # Blau (BGR)
            else:
                continue  # staff_type=None oder etwas anderes -> nix einfärben

            # Färbe die Pixel des Region-Objekts
            for (x, y) in region.pixels:
                debug_window_img[y, x] = color
                colored_count += 1

        output_path = f"debug_window_colored_{idx + 1}.png"
        cv2.imwrite(output_path, debug_window_img)
        print(f"Speichere gefärbtes Fenster {idx + 1} in {output_path}")

        # Verschiebe die Pixelkoordinaten relativ zum Originalbild
        for region in connected_regions:
            adjusted_pixels = [(x + x_min, y + y_min) for (x, y) in region.pixels]
            adjusted_region = ConnectedRegion(adjusted_pixels, "None", False)
            regions.append(adjusted_region)

    print(f"Gesamt gefundene zusammenhängende Regionen: {len(regions)}")
    return regions

def process_staff_windows(
    original_img: np.ndarray,
    staffs: List[dict],
    threshold_value: int = 245
) -> Dict[int, Dict[str, List[Tuple[int, int]]]]:

    # ---------------------------------------------------------------------
    # 0) Vorbereitungen
    # ---------------------------------------------------------------------
    sorted_staffs = sort_staffs_top_to_bottom(staffs)
    num_staffs    = len(sorted_staffs)
    img_h, img_w  = original_img.shape[:2]

    # Ergebnis‑Container: für jeden Staff eine Top‑ und Bottom‑Liste
    staff_pixel_map: Dict[int, Dict[str, List[Tuple[int, int]]]] = {
        idx: {"Top": [], "Bottom": [], "Middle": []} for idx in range(num_staffs)
    }

    # ---------------------------------------------------------------------
    # 1) Fenster ZWISCHEN benachbarten Staffs
    # ---------------------------------------------------------------------
    for i in range(num_staffs - 1):
        upper_staff = sorted_staffs[i]
        lower_staff = sorted_staffs[i + 1]

        # (x_min, y_min, x_max, y_max)
        x_min, y_min, x_max, y_max = define_window_between_staffs(
            upper_staff, lower_staff, img_w
        )

        if y_max <= y_min:          # Sicherheits‑Check, falls Staffs kollidieren
            continue

        # --- Bildausschnitt -> Binärbild ---------------------------------
        window_img = original_img[y_min:y_max, x_min:x_max]
        gray       = cv2.cvtColor(window_img, cv2.COLOR_BGR2GRAY)
        _, bw      = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        inv        = cv2.bitwise_not(bw)

        # --- Regionen finden, segmentieren --------------------------------
        regions                 = extract_connected_regions(inv, False)
        graph, _                = build_fully_connected_graph(
            regions, x_min, x_max, y_min, y_max
        )
        segmented_graph         = segment_graph(graph)

        # --- Pixel beider Staffs zuordnen ---------------------------------
        # --- Pixel nach staff_type korrekt zuordnen -----------------------
        for _, data in segmented_graph.nodes(data=True):
            reg = data["region"]
            if reg.is_staff:
                continue

            global_pixels = [(px + x_min, py + y_min) for (px, py) in reg.pixels]

            if reg.staff_type == "Top":
                # alles, was dem oberen Linienknoten (Top-Region)
                # zugeordnet wurde, gehört an die Unterkante von Staff i
                staff_pixel_map[i]["Bottom"].extend(global_pixels)

            elif reg.staff_type == "Bottom":
                # alles, was dem unteren Linienknoten gehört,
                # ist die Oberkante von Staff i+1
                staff_pixel_map[i + 1]["Top"].extend(global_pixels)

            # stype == "None" oder unerwartet -> nichts tun

    # ---------------------------------------------------------------------
    # 2) Randfenster OBERHALB des ersten Staff‑Blocks
    # ---------------------------------------------------------------------
    x_min, x_max = 0, img_w - 1
    y_min, y_max = 0, staff_top_y(sorted_staffs[0])
    if y_max > y_min:                         # nur verarbeiten, wenn Platz ist
        border_top = original_img[y_min:y_max, x_min:x_max]
        gray, bw   = cv2.cvtColor(border_top, cv2.COLOR_BGR2GRAY), None
        _, bw      = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        inv        = cv2.bitwise_not(bw)
        regions    = extract_connected_regions(inv, False)

        for reg in regions:
            global_pixels = [
                (px + x_min, py + y_min) for (px, py) in reg.pixels
            ]
            staff_pixel_map[0]["Top"].extend(global_pixels)

    # ---------------------------------------------------------------------
    # 3) Randfenster UNTERHALB des letzten Staff‑Blocks
    # ---------------------------------------------------------------------
    y_min, y_max = staff_bottom_y(sorted_staffs[-1]), img_h - 1
    if y_max > y_min:
        border_bottom = original_img[y_min:y_max, x_min:x_max]
        gray, bw      = cv2.cvtColor(border_bottom, cv2.COLOR_BGR2GRAY), None
        _, bw         = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        inv           = cv2.bitwise_not(bw)
        regions       = extract_connected_regions(inv, False)

        for reg in regions:
            global_pixels = [
                (px + x_min, py + y_min) for (px, py) in reg.pixels
            ]
            staff_pixel_map[num_staffs - 1]["Bottom"].extend(global_pixels)

    print("Adding 'Middle' pixels based on detected staff lines...")
    for staff_idx, staff_details in enumerate(sorted_staffs):
        middle_pixels_for_this_staff = []

        # Horizontale Linien des aktuellen Staffs
        for h_group in staff_details.get('horizontal_groups', []):
            for (y, x_start, x_end) in h_group:
                for x_coord in range(x_start, x_end + 1):
                    if 0 <= y < img_h and 0 <= x_coord < img_w:  # Sicherheitscheck
                        middle_pixels_for_this_staff.append((x_coord, y))

        # Vertikale Linien des aktuellen Staffs
        for v_group in staff_details.get('vertical_groups', []):
            for (x, y_start, y_end) in v_group:
                for y_coord in range(y_start, y_end + 1):
                    if 0 <= y_coord < img_h and 0 <= x < img_w:  # Sicherheitscheck
                        middle_pixels_for_this_staff.append((x, y_coord))

        # Optional: Nur Pixel hinzufügen, die nicht schon Top/Bottom sind, um Redundanz zu minimieren
        # Dies ist aber nicht unbedingt nötig, wenn `pixels_all` in concat_staffs_map_aligned
        # sowieso ein Set wird oder Duplikate nicht stören.
        # Wenn Du es machen willst:
        # current_top_set = set(staff_pixel_map[staff_idx]["Top"])
        # current_bottom_set = set(staff_pixel_map[staff_idx]["Bottom"])
        # unique_middle_pixels = [p for p in list(set(middle_pixels_for_this_staff))
        #                         if p not in current_top_set and p not in current_bottom_set]
        # staff_pixel_map[staff_idx]["Middle"].extend(unique_middle_pixels)

        # Einfachere Variante: Alle Linienpixel als Middle hinzufügen, Duplikate später behandeln
        if middle_pixels_for_this_staff:
            staff_pixel_map[staff_idx]["Middle"].extend(
                list(set(middle_pixels_for_this_staff)))  # set entfernt Duplikate innerhalb von Middle

        # Nochmals Duplikate entfernen, falls Middle-Pixel auch in Top/Bottom gelandet sind
        # (kann passieren, wenn segment_graph z.B. einen Teil einer Linie als Top/Bottom klassifiziert)
        # Top und Bottom haben Vorrang.
        final_middle_set = set(staff_pixel_map[staff_idx]["Middle"])
        final_middle_set.difference_update(set(staff_pixel_map[staff_idx]["Top"]))
        final_middle_set.difference_update(set(staff_pixel_map[staff_idx]["Bottom"]))
        staff_pixel_map[staff_idx]["Middle"] = list(final_middle_set)

        print(f"  Staff {staff_idx}: Added {len(staff_pixel_map[staff_idx]['Middle'])} 'Middle' pixels.")

    # ---------------------------------------------------------------------
    # 5) Fertig
    # ---------------------------------------------------------------------
    return staff_pixel_map

import numpy as np
import cv2
from typing import Dict, List, Tuple

import numpy as np
import cv2
from typing import Dict, List, Tuple

import math
import numpy as np
import cv2
from typing import Dict, List, Tuple

# Assume these helper functions exist and work correctly:
# sort_staffs_top_to_bottom, staff_top_line_y, staff_bottom_line_y, get_staff_rightmost_vertical_x

def extract_lines_as_transparent(original_bgr, staffs):
    """
    Erzeugt ein neues Bild (BGRA) gleicher Größe,
    setzt alles auf (0,0,0,0) und kopiert für jede Linie in 'staffs'
    die Original-Pixel mit voller Deckkraft (A=255).

    Parameter:
    - original_bgr: Originales Bild im BGR-Format (NumPy-Array)
    - staffs: Liste von Stafflines, jede Staffline ist ein Dictionary mit:
        {
            'horizontal_groups': [horizontal_group1, horizontal_group2, ...],
            'vertical_groups': [vertical_group1, vertical_group2, ...]  # Sortiert nach x aufsteigend
        }
        Jede horizontale Gruppe ist eine Liste von Linien [(y, x_start, x_end), ...]
        Jede vertikale Gruppe ist eine Liste von Linien [(x, y_start, y_end), ...]

    Rückgabe:
    - lines_img: Neues Bild (BGRA), nur die Linien sind sichtbar (Alpha = 255), Rest transparent (Alpha = 0)
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


def apply_horizontal_lowpass_to_staff_lines(
    lines_source_bgra: np.ndarray, # Das BGRA-Bild mit allen Linien
    staffs: List[dict],            # Die Staff-Struktur mit horizontal_groups
    kernel_size_x: int = 5         # Größe des horizontalen Glättungskerns (ungerade Zahl)
) -> np.ndarray:
    """
    Wendet einen horizontalen Tiefpassfilter (Box-Filter) selektiv auf die
    horizontalen Liniensegmente eines BGRA-Linienbildes an.
    Vertikale Linien und der transparente Hintergrund bleiben unberührt.

    Parameter:
    - lines_source_bgra: Das von extract_lines_as_transparent erzeugte BGRA-Bild.
    - staffs: Liste der Staff-Strukturen, um die horizontalen Linien zu identifizieren.
    - kernel_size_x: Die Breite des horizontalen Box-Filter-Kernels. Muss ungerade sein.

    Rückgabe:
    - Ein neues BGRA-Bild mit den modifizierten horizontalen Linien.
    """
    if lines_source_bgra.shape[2] != 4:
        print("Fehler: Eingabebild muss BGRA sein.")
        return lines_source_bgra # Oder Fehler auslösen

    if kernel_size_x % 2 == 0:
        print(f"Warnung: kernel_size_x ({kernel_size_x}) sollte ungerade sein. Erhöhe um 1.")
        kernel_size_x += 1
    if kernel_size_x < 1:
        print("Warnung: kernel_size_x ist zu klein, setze auf 1 (kein Effekt).")
        kernel_size_x = 1


    height, width = lines_source_bgra.shape[:2]

    # 1. Maske für horizontale Linien erstellen
    horizontal_lines_mask = np.zeros((height, width), dtype=np.uint8)
    for staff in staffs:
        for h_group in staff.get('horizontal_groups', []):
            for (y, x_start, x_end) in h_group:
                # Sicherstellen, dass die Indizes innerhalb des Bildes liegen
                x_s_clipped = max(0, min(width - 1, x_start))
                x_e_clipped = max(0, min(width - 1, x_end))
                if y >= 0 and y < height and x_e_clipped >= x_s_clipped:
                    horizontal_lines_mask[y, x_s_clipped : x_e_clipped + 1] = 255

    # 2. Horizontalen Tiefpassfilter auf das gesamte Linienbild anwenden
    # Temporär zu BGR konvertieren, da Filter typischerweise auf Farb- oder Graustufenbildern arbeiten
    lines_bgr_temp = lines_source_bgra[:, :, :3]

    # Horizontaler Box-Filter-Kernel
    # kernel = np.ones((1, kernel_size_x), np.float32) / kernel_size_x
    # filtered_bgr = cv2.filter2D(lines_bgr_temp, -1, kernel)

    # Alternative: Horizontaler Gauß-Filter (oft bessere Ergebnisse als Box-Filter)
    # kernel_size_y muss 0 sein, damit der Gauß-Filter nur horizontal wirkt,
    # oder 1, wenn man eine minimale vertikale Glättung nicht vermeiden kann/will.
    # Für rein horizontale Glättung ist es besser, cv2.sepFilter2D zu verwenden oder
    # den Kernel manuell zu definieren, aber für Einfachheit hier cv2.GaussianBlur:
    # Hier setzen wir sigmaY auf einen sehr kleinen Wert, um die vertikale Glättung zu minimieren.
    # Besser ist es, den Filter auf jede Zeile einzeln anzuwenden oder einen 1D-Kernel zu nutzen.
    # Für einen einfachen horizontalen Box-Filter ist cv2.blur praktischer:
    if kernel_size_x > 1:
        filtered_bgr = cv2.blur(lines_bgr_temp, (kernel_size_x, 1))
    else:
        filtered_bgr = lines_bgr_temp.copy() # Kein Filter bei Kernelgröße 1


    # Das gefilterte BGR-Bild wieder mit dem ursprünglichen Alpha-Kanal versehen
    filtered_lines_bgra = cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2BGRA)
    filtered_lines_bgra[:, :, 3] = lines_source_bgra[:, :, 3] # Alpha wiederherstellen

    # 3. Kombiniere die Bilder basierend auf der Maske
    # Erzeuge ein Ergebnisbild, initialisiert mit dem Original
    result_bgra = lines_source_bgra.copy()

    # Wo die Maske weiß ist (horizontale Linien), nimm Pixel vom gefilterten Bild
    # np.where ist effizient dafür:
    # result_bgra = np.where(condition_mask_expanded, image_if_true, image_if_false)
    # Die Maske muss auf 4 Kanäle erweitert werden, um mit BGRA-Bildern zu arbeiten
    mask_expanded_for_bgra = cv2.cvtColor(horizontal_lines_mask, cv2.COLOR_GRAY2BGRA)
    mask_boolean = mask_expanded_for_bgra[:,:,0] == 255 # Boolean Maske für np.where

    # Pixelweise Operation ist langsam, aber für np.where muss die Maske korrekt sein.
    # Einfacher ist, die Pixel direkt zu kopieren:
    indices = np.where(horizontal_lines_mask == 255)
    result_bgra[indices[0], indices[1], :] = filtered_lines_bgra[indices[0], indices[1], :]

    return result_bgra

def concat_staff_lines_transparent_aligned(
    original_img: np.ndarray,
    staffs: List[dict],
    staff_pixel_map: Dict[int, Dict[str, List[Tuple[int, int]]]],
    cut_left_while_copy=None,
    staff_index_offset=None,
    pad_top: int = 5,
    pad_bottom: int = 5,
    horizontal_gap: int = 1,
    apply_h_lowpass: bool = True, # Neuer Parameter zum Steuern
    h_lowpass_kernel_x: int = 5   # Neuer Parameter für Kernelgröße
) -> Tuple[np.ndarray, list, int]:
    """
    Hängt Staffs nebeneinander, wobei nur die erkannten Liniensegmente ...
    Optional wird ein horizontaler Tiefpassfilter auf die horizontalen Linien angewendet.
    """
    img_h, img_w = original_img.shape[:2]

    # 1. Erzeuge das Bild nur mit den Staff-Linien (BGRA, transparent)
    lines_source_bgra = extract_lines_as_transparent(original_img, staffs)
    if lines_source_bgra.shape[2] != 4:
        # ... (Fehlerbehandlung wie zuvor) ...
        return np.zeros((img_h, img_w, 4), dtype=np.uint8)

    # NEU: Wende optional den horizontalen Tiefpassfilter an
    if apply_h_lowpass:
        print(f"Wende horizontalen Tiefpassfilter (Kernel X: {h_lowpass_kernel_x}) auf horizontale Linien an...")
        lines_source_bgra = apply_horizontal_lowpass_to_staff_lines(
            lines_source_bgra,
            staffs,
            kernel_size_x=h_lowpass_kernel_x
        )
        # Optional: Ergebnis zwischenspeichern/anzeigen für Debugging
        # cv2.imwrite("debug_lines_after_h_lowpass.png", lines_source_bgra)

    # 2. Staffs sortieren
    # ... (Rest der Funktion bleibt identisch wie zuvor) ...
    # Der Code ab hier verwendet das (potenziell modifizierte) lines_source_bgra
    # ...
    # staff_data_lines_only = []
    # for idx, staff in enumerate(sorted_staffs):
    #    ...
    #    staff_lines_canvas_bgra[
    #        dest_y0_on_canvas : dest_y0_on_canvas + copy_h,
    #        dest_x0_on_canvas : dest_x0_on_canvas + copy_w
    #    ] = lines_source_bgra[ # <--- Hier wird das modifizierte Bild verwendet
    #        clipped_src_y0 : clipped_src_y1,
    #        clipped_src_x0 : clipped_src_x1
    #    ]
    #    ...

    # Der Rest der Funktion concat_staff_lines_transparent_aligned bleibt gleich.
    # Sie müssen nur die Parameter `apply_h_lowpass` und `h_lowpass_kernel_x`
    # beim Aufruf in `main()` übergeben.

    # ... (wie in der vorherigen Antwort)
    sorted_staffs = sort_staffs_top_to_bottom(staffs)
    if not sorted_staffs:
        print("Warnung: Keine Staffs für Konkatenation übergeben.")
        return np.zeros((50, 50, 4), dtype=np.uint8)

    staff_data_lines_only = []

    for idx, staff in enumerate(sorted_staffs):
        global_idx = staff_index_offset + idx
        cut_val = cut_left_while_copy[global_idx]
        if idx not in staff_pixel_map:
             print(f"Warnung: Kein Pixel-Mapping für Staff {idx}. Überspringe.")
             continue

        pixels_top = staff_pixel_map[idx].get("Top", [])
        pixels_bottom = staff_pixel_map[idx].get("Bottom", [])
        pixels_middle = staff_pixel_map[idx].get("Middle", [])  # NEU
        pixels_all_mapped = pixels_top + pixels_bottom + pixels_middle  # ALLE relevanten Pixel

        staff_actual_top_y = staff_top_line_y(staff)
        staff_actual_bottom_y = staff_bottom_line_y(staff)
        rightmost_vertical_x_orig = get_staff_rightmost_vertical_x(staff)

        valid_lines = (staff_actual_top_y >= 0 and
                       staff_actual_bottom_y > staff_actual_top_y and
                       staff_actual_bottom_y < img_h)

        if not pixels_all_mapped and not valid_lines:
             print(f"Warnung: Weder gemappte Pixel noch gültige Linien für Staff {idx}. Überspringe.")
             continue

        x_min_overall, y_min_overall = 0, 0
        x_max_overall, y_max_overall = img_w - 1, img_h - 1

        if pixels_all_mapped:
            xs_pix, ys_pix = zip(*pixels_all_mapped)
            x_min_pix, y_min_pix = min(xs_pix), min(ys_pix)
            x_max_pix, y_max_pix = max(xs_pix), max(ys_pix)
            x_min_overall = max(0, x_min_pix)
            x_max_overall = min(img_w - 1, max(x_max_pix, rightmost_vertical_x_orig if rightmost_vertical_x_orig >= 0 else 0))
            y_min_overall = max(0, y_min_pix)
            y_max_overall = min(img_h - 1, y_max_pix)
        elif valid_lines:
             h_lines = staff.get("horizontal_groups", [])
             all_h_x_coords = [x_coord for hg in h_lines for (_, x_s, x_e) in hg for x_coord in [x_s, x_e]]
             if all_h_x_coords:
                 x_min_overall = max(0, min(all_h_x_coords))
                 x_max_overall = min(img_w - 1, max(all_h_x_coords))
             x_max_overall = min(img_w - 1, max(x_max_overall, rightmost_vertical_x_orig if rightmost_vertical_x_orig >= 0 else 0))
             y_min_overall = max(0, staff_actual_top_y)
             y_max_overall = min(img_h - 1, staff_actual_bottom_y)

        x_eff_start = x_min_overall
        if global_idx > 0:
            potential_width = x_max_overall - x_min_overall + 1
            if potential_width > cut_val:
                 x_eff_start = x_min_overall + cut_val
            x_eff_start = min(x_eff_start, x_max_overall)

        canvas_y0_orig = max(0, y_min_overall - pad_top)
        canvas_y1_orig = min(img_h - 1, y_max_overall + pad_bottom)
        canvas_h_staff = canvas_y1_orig - canvas_y0_orig + 1

        canvas_x0_orig = x_eff_start
        canvas_x1_orig = x_max_overall
        canvas_w_staff = canvas_x1_orig - canvas_x0_orig + 1

        if canvas_h_staff <= 0 or canvas_w_staff <= 0:
            print(f"Warnung: Ungültige Dimensionen für Staff-{idx}-Linien-Canvas (H={canvas_h_staff}, W={canvas_w_staff}). Überspringe.")
            continue

        staff_lines_canvas_bgra = np.zeros((canvas_h_staff, canvas_w_staff, 4), dtype=np.uint8)

        src_y0 = canvas_y0_orig
        src_x0 = canvas_x0_orig
        clipped_src_y0 = max(0, src_y0)
        clipped_src_x0 = max(0, src_x0)
        clipped_src_y1 = min(lines_source_bgra.shape[0], src_y0 + canvas_h_staff)
        clipped_src_x1 = min(lines_source_bgra.shape[1], src_x0 + canvas_w_staff)
        copy_h = clipped_src_y1 - clipped_src_y0
        copy_w = clipped_src_x1 - clipped_src_x0
        dest_y0_on_canvas = clipped_src_y0 - src_y0
        dest_x0_on_canvas = clipped_src_x0 - src_x0

        if copy_h > 0 and copy_w > 0:
            if (dest_y0_on_canvas + copy_h <= canvas_h_staff and
                dest_x0_on_canvas + copy_w <= canvas_w_staff):
                staff_lines_canvas_bgra[
                    dest_y0_on_canvas : dest_y0_on_canvas + copy_h,
                    dest_x0_on_canvas : dest_x0_on_canvas + copy_w
                ] = lines_source_bgra[
                    clipped_src_y0 : clipped_src_y1,
                    clipped_src_x0 : clipped_src_x1
                ]

        center_in_canvas_y = canvas_h_staff / 2.0
        if valid_lines:
            rel_top_line_y = staff_actual_top_y - canvas_y0_orig
            rel_bottom_line_y = staff_actual_bottom_y - canvas_y0_orig
            if 0 <= rel_top_line_y < canvas_h_staff and \
               0 <= rel_bottom_line_y < canvas_h_staff and \
               rel_bottom_line_y > rel_top_line_y:
               center_in_canvas_y = (rel_top_line_y + rel_bottom_line_y) / 2.0

        staff_data_lines_only.append({
            "crop": staff_lines_canvas_bgra,
            "width": canvas_w_staff,
            "height": canvas_h_staff,
            "center_in_crop_y": center_in_canvas_y,
            "rightmost_vertical_x_orig": rightmost_vertical_x_orig,
            "effective_crop_start_x_orig": canvas_x0_orig,
            "x_placement": 0,
            "canvas_y_start": 0
        })

    if not staff_data_lines_only:
       print("Fehler: Keine gültigen Staff-Daten für transparente Linien-Konkatenation gefunden.")
       return np.zeros((50, 50, 4), dtype=np.uint8), [], 25  # Fallback target_center_y

    max_dist_above_center = 0.0
    max_dist_below_center = 0.0
    for data in staff_data_lines_only:
        max_dist_above_center = max(max_dist_above_center, data["center_in_crop_y"])
        max_dist_below_center = max(max_dist_below_center, data["height"] - data["center_in_crop_y"])

    final_canvas_h = math.ceil(max_dist_above_center + max_dist_below_center) + 2
    target_center_canvas_y = max_dist_above_center + 1

    # Vertikale Platzierung jedes Staff-Crops auf der Streifen-Canvas
    for data_item in staff_data_lines_only:
        data_item["canvas_y_start"] = int(round(target_center_canvas_y - data_item["center_in_crop_y"]))

    x_cursor = 0
    last_effective_right_edge_on_canvas = 0
    for i, data in enumerate(staff_data_lines_only):
        if i == 0:
            data['x_placement'] = 0
        else:
            prev_data = staff_data_lines_only[i-1]
            prev_rightmost_v_orig = prev_data['rightmost_vertical_x_orig']
            if prev_rightmost_v_orig >= 0:
                 prev_rightmost_v_in_prev_crop_x = prev_rightmost_v_orig - prev_data['effective_crop_start_x_orig']
                 if 0 <= prev_rightmost_v_in_prev_crop_x < prev_data['width']:
                     x_cursor = prev_data['x_placement'] + prev_rightmost_v_in_prev_crop_x + horizontal_gap
                 else:
                     x_cursor = prev_data['x_placement'] + prev_data['width'] + horizontal_gap
            else:
                x_cursor = prev_data['x_placement'] + prev_data['width'] + horizontal_gap
            data['x_placement'] = int(round(x_cursor))
        last_effective_right_edge_on_canvas = data['x_placement'] + data['width']

    final_total_w = int(round(last_effective_right_edge_on_canvas))
    final_canvas_bgra = np.zeros((final_canvas_h, final_total_w, 4), dtype=np.uint8)

    for i, data in enumerate(staff_data_lines_only):
        staff_canvas_to_paste_bgra = data["crop"]
        w, h = data["width"], data["height"]
        place_x, place_y = data["x_placement"], data["canvas_y_start"]
        paste_y0_fc, paste_x0_fc = place_y, place_x
        paste_y1_fc, paste_x1_fc = place_y + h, place_x + w
        crop_y0_sc, crop_x0_sc = 0, 0
        if paste_y0_fc < 0:
            crop_y0_sc = -paste_y0_fc
            paste_y0_fc = 0
        if paste_y1_fc > final_canvas_h:
            paste_y1_fc = final_canvas_h
        if paste_x0_fc < 0:
             crop_x0_sc = -paste_x0_fc
             paste_x0_fc = 0
        if paste_x1_fc > final_total_w:
            paste_x1_fc = final_total_w
        actual_paste_h = paste_y1_fc - paste_y0_fc
        actual_paste_w = paste_x1_fc - paste_x0_fc
        actual_crop_h = actual_paste_h
        actual_crop_w = actual_paste_w

        if actual_paste_h > 0 and actual_paste_w > 0 and \
           crop_y0_sc + actual_crop_h <= h and \
           crop_x0_sc + actual_crop_w <= w:
            try:
                 final_canvas_bgra[
                     paste_y0_fc:paste_y1_fc,
                     paste_x0_fc:paste_x1_fc
                 ] = staff_canvas_to_paste_bgra[
                     crop_y0_sc : crop_y0_sc + actual_crop_h,
                     crop_x0_sc : crop_x0_sc + actual_crop_w
                 ]
            except ValueError as e:
                 print(f"  FEHLER beim Pasten des Staff-{i}-Linien-Canvas: {e}")


    return final_canvas_bgra, staff_data_lines_only, int(round(target_center_canvas_y))


def ridge_from_pixels(pixels, keep="top"):
    """
    Reduziert eine Punktmenge auf den 'Kamm' pro X-Spalte.
    keep="top"    -> kleinstes y (oberster Pixel)
    keep="bottom" -> größtes y (unterster Pixel)
    """
    ridge = {}
    for x, y in pixels:
        if keep == "top":
            ridge[x] = min(y, ridge.get(x, y))
        else:                        # bottom
            ridge[x] = max(y, ridge.get(x, y))
    return [(x, y) for x, y in ridge.items()]


def concat_staffs_map_aligned(
        original_img: np.ndarray,
        staffs: List[dict],  # Wird primär für Metadaten (Alignment-Center, rightmost_vertical) benötigt
        staff_pixel_map: Dict[int, Dict[str, List[Tuple[int, int]]]],
        cut_left_while_copy: List[int],
        staff_index_offset: int,
        pad_top: int = 5,
        pad_bottom: int = 5,
        horizontal_gap: int = 1,
        envelope_fallback_offset: int = 10,
        # middle_overlap wird hier nicht direkt für den Kopiervorgang des Staff-Inhalts verwendet,
        # könnte aber noch für staff_actual_top/bottom_y relevant sein, falls diese für was anderes gebraucht werden.
        # Für das reine Kopieren der Pixel aus der Map ist es nicht nötig.
        bg_color=(255, 255, 255)
) -> Tuple[np.ndarray, list, int]:
    img_h, img_w = original_img.shape[:2]
    bg_color_tuple = tuple(bg_color)

    sorted_staffs = sort_staffs_top_to_bottom(staffs)
    if not sorted_staffs:
        return np.full((50, 50, 3), bg_color_tuple, dtype=np.uint8), [], 25

    staff_data = []

    # --- Vorverarbeitung der staff_pixel_map für schnellen Spaltenzugriff auf Y-Extrema ---
    # Erstellt für jeden Staff ein Dictionary: x_coord_orig -> {"min_y_top": y, "max_y_top": y, "min_y_middle":y, ...}
    pixel_extrema_by_x_orig = {}
    for s_idx in range(len(sorted_staffs)):
        if s_idx not in staff_pixel_map: continue
        pixel_extrema_by_x_orig[s_idx] = {}

        categories = ["Top", "Middle", "Bottom"]
        for category in categories:
            for px_orig, py_orig in staff_pixel_map[s_idx].get(category, []):
                if px_orig not in pixel_extrema_by_x_orig[s_idx]:
                    pixel_extrema_by_x_orig[s_idx][px_orig] = {}

                cat_min_key = f"min_y_{category.lower()}"
                cat_max_key = f"max_y_{category.lower()}"

                current_cat_min = pixel_extrema_by_x_orig[s_idx][px_orig].get(cat_min_key, py_orig)
                pixel_extrema_by_x_orig[s_idx][px_orig][cat_min_key] = min(current_cat_min, py_orig)

                current_cat_max = pixel_extrema_by_x_orig[s_idx][px_orig].get(cat_max_key, py_orig)
                pixel_extrema_by_x_orig[s_idx][px_orig][cat_max_key] = max(current_cat_max, py_orig)
    # --- Ende Vorverarbeitung ---

    for idx, staff_structure_info in enumerate(sorted_staffs):
        global_idx = staff_index_offset + idx

        if idx not in staff_pixel_map or not pixel_extrema_by_x_orig.get(idx):
            print(f"Warnung: Kein Pixel-Mapping oder keine X-Extrema für Staff {idx}. Überspringe.")
            continue

        # Hole alle Pixel-Listen für diesen Staff, um Gesamtgrenzen zu bestimmen
        pixels_top = staff_pixel_map[idx].get("Top", [])
        pixels_middle = staff_pixel_map[idx].get("Middle", [])
        pixels_bottom = staff_pixel_map[idx].get("Bottom", [])
        all_pixels_for_this_staff = pixels_top + pixels_middle + pixels_bottom

        if not all_pixels_for_this_staff:
            print(f"Warnung: Staff {idx} (Global {global_idx}) hat keine Pixel in der Map. Überspringe.")
            continue

        # --- Bestimme Gesamtgrenzen des Staffs aus ALLEN seinen Pixeln für den Canvas ---
        # Dies definiert den Bereich des Originalbildes, den der staff_canvas abdecken wird.
        xs_all, ys_all = zip(*all_pixels_for_this_staff)
        x_min_staff_pixels = min(xs_all)
        x_max_staff_pixels = max(xs_all)
        y_min_staff_pixels = min(ys_all)
        y_max_staff_pixels = max(ys_all)

        # Metadaten für Alignment etc.
        staff_actual_top_y_line = staff_top_line_y(staff_structure_info)
        staff_actual_bottom_y_line = staff_bottom_line_y(staff_structure_info)
        rightmost_vertical_x_orig = get_staff_rightmost_vertical_x(staff_structure_info)
        valid_lines_for_center = (staff_actual_top_y_line >= 0 and staff_actual_bottom_y_line > staff_actual_top_y_line)

        # x_max_staff_pixels auch durch rightmost_vertical_x erweitern, falls nötig
        if rightmost_vertical_x_orig >= 0:
            x_max_staff_pixels = max(x_max_staff_pixels, rightmost_vertical_x_orig)

        # --- Apply cut_left_while_copy für den Canvas-Ausschnitt ---
        # canvas_x0_orig_img ist der Start-X-Wert im Originalbild für diesen Staff-Canvas
        canvas_x0_orig_img = x_min_staff_pixels
        if global_idx > 0:
            cut_val = cut_left_while_copy[global_idx]
            potential_width = x_max_staff_pixels - x_min_staff_pixels + 1
            if potential_width > cut_val:
                canvas_x0_orig_img = x_min_staff_pixels + cut_val
            canvas_x0_orig_img = min(canvas_x0_orig_img, x_max_staff_pixels)  # Nicht über das Ende hinaus

        canvas_x1_orig_img = x_max_staff_pixels  # Ende des Canvas-Ausschnitts im Originalbild

        # Y-Grenzen des Canvas-Ausschnitts im Originalbild (mit Padding)
        canvas_y0_orig_img = max(0, y_min_staff_pixels - pad_top)
        canvas_y1_orig_img = min(img_h - 1, y_max_staff_pixels + pad_bottom)

        canvas_w_staff = canvas_x1_orig_img - canvas_x0_orig_img + 1
        canvas_h_staff = canvas_y1_orig_img - canvas_y0_orig_img + 1

        if canvas_h_staff <= 0 or canvas_w_staff <= 0:
            print(f"Warnung: Ungültige Canvas-Dimensionen für Staff {idx}. Überspringe.")
            continue

        staff_canvas = np.full((canvas_h_staff, canvas_w_staff, 3), bg_color_tuple, dtype=np.uint8)

        # --- Fülle den staff_canvas spaltenweise basierend auf dem dynamischen Umschlag ---

        # Bestimme den X-Bereich im Originalbild, der tatsächlich Middle-Pixel hat
        # Dies ist der Bereich, über den wir iterieren MÜSSEN.
        # Wenn keine Middle-Pixel da sind, könnten wir Fallbacks brauchen oder nur Top/Bottom zeichnen.
        x_coords_with_middle = sorted(
            [px for px in pixel_extrema_by_x_orig[idx] if "min_y_middle" in pixel_extrema_by_x_orig[idx][px]])

        if not x_coords_with_middle:
            print(
                f"Hinweis: Staff {idx} hat keine 'Middle'-Pixel mit X-Extrema. Zeichne nur Top/Bottom, falls vorhanden.")
            # In diesem Fall nur Top/Bottom zeichnen (siehe unten) oder den Staff als leer betrachten.
            # Fürs Erste gehen wir davon aus, dass Middle-Pixel (Linien) meist vorhanden sind.
            # Wenn nicht, muss die Logik für `loop_x_start_orig` und `loop_x_end_orig` angepasst werden.
            loop_x_start_orig = canvas_x0_orig_img  # Fallback
            loop_x_end_orig = canvas_x1_orig_img  # Fallback
            if not pixels_top and not pixels_bottom:  # Wenn gar nichts da ist
                print(
                    f"Überspringe Staff {idx}, da keine Middle, Top oder Bottom Pixel für den Umschlag gefunden wurden.")
                continue
            elif pixels_top or pixels_bottom:  # Nur Top/Bottom da, nehme deren X-Grenzen
                temp_xs = []
                if pixels_top: temp_xs.extend([p[0] for p in pixels_top])
                if pixels_bottom: temp_xs.extend([p[0] for p in pixels_bottom])
                if temp_xs:
                    loop_x_start_orig = max(canvas_x0_orig_img, min(temp_xs))
                    loop_x_end_orig = min(canvas_x1_orig_img, max(temp_xs))
                else:  # Sollte nicht passieren, da pixels_all oben geprüft wurde
                    continue
        else:
            loop_x_start_orig = max(canvas_x0_orig_img, min(x_coords_with_middle))
            loop_x_end_orig = min(canvas_x1_orig_img, max(x_coords_with_middle))

        for px_orig_col in range(loop_x_start_orig, loop_x_end_orig + 1):
            # Hole die Y-Extrema für diese Original-X-Spalte
            col_extrema = pixel_extrema_by_x_orig[idx].get(px_orig_col, {})

            # Bestimme die obere Grenze des Umschlags (envelope_y_top)
            envelope_y_top = -1  # Ungültiger Startwert
            if "min_y_top" in col_extrema:  # Gibt es Top-Pixel in dieser Spalte?
                envelope_y_top = col_extrema["min_y_top"]
            elif "min_y_middle" in col_extrema:  # Wenn nicht, nimm den obersten Middle-Pixel
                envelope_y_top = col_extrema["min_y_middle"] - envelope_fallback_offset
            # else: Wenn weder Top noch Middle, bleibt envelope_y_top ungültig. Man könnte hier auch
            #        staff_actual_top_y_line als Fallback nehmen, aber das wäre wieder eine Box-Annahme.
            #        Wenn envelope_y_top ungültig bleibt, wird in dieser Spalte nichts gezeichnet,
            #        es sei denn, es gibt Bottom-Pixel, die dann den Bereich definieren.

            # Bestimme die untere Grenze des Umschlags (envelope_y_bottom)
            envelope_y_bottom = -1  # Ungültiger Startwert
            if "max_y_bottom" in col_extrema:  # Gibt es Bottom-Pixel in dieser Spalte?
                envelope_y_bottom = col_extrema["max_y_bottom"]
            elif "max_y_middle" in col_extrema:  # Wenn nicht, nimm den untersten Middle-Pixel
                envelope_y_bottom = col_extrema["max_y_middle"] + envelope_fallback_offset
            # else: Ähnlich wie oben, Fallback oder nichts zeichnen.

            # Nur wenn gültige Grenzen für den Umschlag gefunden wurden:
            if envelope_y_top != -1 and envelope_y_bottom != -1 and envelope_y_top <= envelope_y_bottom:
                # Konvertiere px_orig_col in canvas_x Koordinate
                cx_canvas = px_orig_col - canvas_x0_orig_img
                if not (0 <= cx_canvas < canvas_w_staff):  # Sollte nicht passieren, da loop_x Grenzen hat
                    continue

                for py_orig_fill in range(envelope_y_top, envelope_y_bottom + 1):
                    # Stelle sicher, dass der Originalpixel im Canvas-Y-Ausschnitt liegt
                    if py_orig_fill >= canvas_y0_orig_img and py_orig_fill <= canvas_y1_orig_img:
                        cy_canvas = py_orig_fill - canvas_y0_orig_img
                        # Stelle sicher, dass die Canvas-Koordinate gültig ist
                        if 0 <= cy_canvas < canvas_h_staff:
                            # Kopiere den Pixel aus dem Originalbild
                            staff_canvas[cy_canvas, cx_canvas] = original_img[py_orig_fill, px_orig_col]

        # Optional: Falls es Top/Bottom Pixel gibt, die außerhalb des Middle-Bereichs liegen
        # (z.B. sehr hohe Fähnchen), diese explizit zeichnen.
        # Dies wäre eine Redundanz, wenn die Envelope-Logik oben sie schon erfasst.
        # Aber es kann als Sicherheitsnetz dienen, falls die Envelope-Grenzen zu eng waren.
        for category_pixels in [pixels_top,
                                pixels_bottom]:  # Auch pixels_middle hier, falls sie nicht Teil des Envelope-Kopierens waren
            for px_orig, py_orig in category_pixels:
                if px_orig >= canvas_x0_orig_img and px_orig <= canvas_x1_orig_img and \
                        py_orig >= canvas_y0_orig_img and py_orig <= canvas_y1_orig_img:
                    cx = px_orig - canvas_x0_orig_img
                    cy = py_orig - canvas_y0_orig_img
                    if 0 <= cy < canvas_h_staff and 0 <= cx < canvas_w_staff:
                        # Überschreibe nur, wenn der Canvas-Pixel noch Hintergrundfarbe hat,
                        # oder entscheide, welche Pixelkategorie Vorrang hat.
                        # Hier überschreiben wir einfach, was bedeutet, dass Top/Bottom
                        # die Envelope-Pixel übermalen könnten, falls es Unterschiede gibt.
                        staff_canvas[cy, cx] = original_img[py_orig, px_orig]

        # --- ALIGNMENT CENTER (wie gehabt, basierend auf Linien) ---
        center_in_canvas_y = canvas_h_staff / 2.0
        if valid_lines_for_center:  # valid_lines_for_center wurde oben definiert
            rel_top_line_y = staff_actual_top_y_line - canvas_y0_orig_img
            rel_bottom_line_y = staff_actual_bottom_y_line - canvas_y0_orig_img
            if 0 <= rel_top_line_y < canvas_h_staff and \
                    0 <= rel_bottom_line_y < canvas_h_staff and \
                    rel_bottom_line_y > rel_top_line_y:
                center_in_canvas_y = (rel_top_line_y + rel_bottom_line_y) / 2.0

        staff_data.append({
            "crop": staff_canvas,
            "width": canvas_w_staff,
            "height": canvas_h_staff,
            "center_in_crop_y": center_in_canvas_y,
            "rightmost_vertical_x_orig": rightmost_vertical_x_orig,
            "effective_crop_start_x_orig": canvas_x0_orig_img,
            "x_placement": 0,
            "canvas_y_start": 0
        })

    # --- Loop 2: Final Concatenation (bleibt wie gehabt) ---
    # ... (Code aus deinem vorherigen Post) ...
    if not staff_data:
        print("Fehler: Keine gültigen Staff-Daten für Konkatenation gefunden (Loop 1).")
        return np.full((50, 50, 3), bg_color_tuple, dtype=np.uint8), [], 25

    max_dist_above_center = 0.0
    max_dist_below_center = 0.0
    for data_item in staff_data:
        max_dist_above_center = max(max_dist_above_center, data_item["center_in_crop_y"])
        max_dist_below_center = max(max_dist_below_center, data_item["height"] - data_item["center_in_crop_y"])

    final_canvas_h = math.ceil(max_dist_above_center + max_dist_below_center) + 2
    if final_canvas_h <= 2: final_canvas_h = 50
    target_center_canvas_y = max_dist_above_center + 1
    if target_center_canvas_y <= 1 and final_canvas_h > 2: target_center_canvas_y = final_canvas_h // 2

    for data_item in staff_data:
        center_in_crop_y = data_item["center_in_crop_y"]
        canvas_y_start_float = target_center_canvas_y - center_in_crop_y
        data_item["canvas_y_start"] = int(round(canvas_y_start_float))

    x_cursor = 0
    last_effective_right_edge_on_canvas = 0
    for i, data_item in enumerate(staff_data):
        if i == 0:
            data_item['x_placement'] = 0
        else:
            prev_data = staff_data[i - 1]
            prev_rightmost_v_orig = prev_data['rightmost_vertical_x_orig']

            if prev_rightmost_v_orig >= 0:
                prev_rightmost_v_in_prev_crop_x = prev_rightmost_v_orig - prev_data['effective_crop_start_x_orig']
                if 0 <= prev_rightmost_v_in_prev_crop_x < prev_data['width']:
                    prev_rightmost_v_canvas_x = prev_data['x_placement'] + prev_rightmost_v_in_prev_crop_x
                    x_cursor = prev_rightmost_v_canvas_x + horizontal_gap
                else:
                    prev_crop_right_edge_canvas_x = prev_data['x_placement'] + prev_data['width']
                    x_cursor = prev_crop_right_edge_canvas_x + horizontal_gap
            else:
                prev_crop_right_edge_canvas_x = prev_data['x_placement'] + prev_data['width']
                x_cursor = prev_crop_right_edge_canvas_x + horizontal_gap
            data_item['x_placement'] = int(round(x_cursor))
        last_effective_right_edge_on_canvas = data_item['x_placement'] + data_item['width']

    final_total_w = int(round(last_effective_right_edge_on_canvas))
    if final_total_w <= 0: final_total_w = 50

    final_canvas = np.full((final_canvas_h, final_total_w, 3), bg_color_tuple, dtype=np.uint8)

    for i, data_item in enumerate(staff_data):
        staff_canvas_to_paste = data_item["crop"]
        w = data_item["width"]
        h = data_item["height"]
        place_x = data_item["x_placement"]
        place_y = data_item["canvas_y_start"]

        paste_y_start = place_y
        paste_y_end = place_y + h
        paste_x_start = place_x
        paste_x_end = place_x + w

        crop_y_start = 0
        crop_y_end = h
        crop_x_start = 0
        crop_x_end = w

        if paste_y_start < 0:
            crop_y_start = -paste_y_start
            paste_y_start = 0
        if paste_y_end > final_canvas_h:
            crop_y_end = h - (paste_y_end - final_canvas_h)
            paste_y_end = final_canvas_h
        if paste_x_start < 0:
            crop_x_start = -paste_x_start
            paste_x_start = 0
        if paste_x_end > final_total_w:
            crop_x_end = w - (paste_x_end - final_total_w)
            paste_x_end = final_total_w

        paste_h_actual = paste_y_end - paste_y_start
        paste_w_actual = paste_x_end - paste_x_start
        crop_h_actual = crop_y_end - crop_y_start
        crop_w_actual = crop_x_end - crop_x_start

        if paste_h_actual > 0 and paste_w_actual > 0 and \
                crop_h_actual == paste_h_actual and crop_w_actual == paste_w_actual and \
                crop_y_start < h and crop_x_start < w:
            try:
                final_canvas[paste_y_start:paste_y_end, paste_x_start:paste_x_end] = \
                    staff_canvas_to_paste[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
            except ValueError as e:
                print(f"  ERROR pasting Staff-{i} canvas (Loop 2): {e}")
        else:
            print(f"  Warnung: Skipping final paste for Staff {i} (Loop 2).")

    return final_canvas, staff_data, int(round(target_center_canvas_y))



def visualize_connected_regions(original_img: np.ndarray, regions: List[ConnectedRegion], output_path: str):
    """
    Visualisiert die zusammenhängenden Regionen, indem Rechtecke um sie gezeichnet werden.

    Parameter:
    - original_img: Originales BGR-Bild als NumPy-Array
    - regions: Liste von ConnectedRegion Objekten
    - output_path: Pfad zum Speichern des visualisierten Bildes
    """
    visual_img = original_img.copy()
    for region in regions:
        # Zeichne ein Rechteck um die Region
        cv2.rectangle(visual_img, (region.min_x, region.min_y), (region.max_x, region.max_y), (0, 0, 255), 1)

    cv2.imwrite(output_path, visual_img)
    print(f"Visualisierung der Regionen wurde in '{output_path}' gespeichert.")


def extract_notes_as_transparent(original_bgr: np.ndarray, staffs: List[dict]) -> np.ndarray:
    """
    Erzeugt ein neues Bild (BGRA) gleicher Größe, setzt alles auf (0,0,0,0) und kopiert für jede Linie in 'staffs'
    die Original-Pixel mit voller Deckkraft (A=255).

    Parameter:
    - original_bgr: Originales Bild im BGR-Format (NumPy-Array)
    - staffs: Liste von Staffs, jede Staff ist ein Dictionary mit 'horizontal_groups' und 'vertical_groups'

    Rückgabe:
    - lines_img: Neues Bild (BGRA), nur die Linien sind sichtbar (Alpha = 255), Rest transparent (Alpha = 0)
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
                lines_img[y, x_start_clipped:x_end_clipped + 1, 0:3] = original_bgr[y,
                                                                       x_start_clipped:x_end_clipped + 1, 0:3]
                # Setze Alpha-Kanal auf 255 (voll sichtbar)
                lines_img[y, x_start_clipped:x_end_clipped + 1, 3] = 255

        # Färbe vertikale Gruppen
        for v_group in staff['vertical_groups']:
            for (x, y_start, y_end) in v_group:
                # Sicherstellen, dass die Indizes innerhalb des Bildes liegen
                y_start_clipped = max(0, min(height - 1, y_start))
                y_end_clipped = max(0, min(height - 1, y_end))
                # Kopiere die Pixel der vertikalen Linie
                lines_img[y_start_clipped:y_end_clipped + 1, x, 0:3] = original_bgr[y_start_clipped:y_end_clipped + 1,
                                                                       x, 0:3]
                # Setze Alpha-Kanal auf 255 (voll sichtbar)
                lines_img[y_start_clipped:y_end_clipped + 1, x, 3] = 255

    return lines_img


def distance_between_regions(regionA: ConnectedRegion, regionB: ConnectedRegion) -> float:
    """
    Berechnet den minimalen Abstand zwischen zwei Regionen:
      - Wenn beide normale Regionen sind, nimm den euklidischen Abstand
        über die 4 Extremkoordinaten.
      - Wenn mindestens eine Region ein Staff ist, verwende nur die y-Koordinate.
    """
    # Fall 1: Beide normal -> Euklidischer Abstand mit 4×4 Extrempunkten
    if not regionA.is_staff and not regionB.is_staff:
        coordsA = [regionA.min_x_coord, regionA.max_x_coord, regionA.min_y_coord, regionA.max_y_coord]
        coordsB = [regionB.min_x_coord, regionB.max_x_coord, regionB.min_y_coord, regionB.max_y_coord]

        min_dist = float('inf')
        for (xA, yA) in coordsA:
            for (xB, yB) in coordsB:
                dist = math.hypot(xA - xB, yA - yB)
                if dist < min_dist:
                    min_dist = dist
        return min_dist

    # Fall 2: Mindestens eine ist Staff
    # Ermittlung: staffRegion & normalRegion
    staffRegion = regionA if regionA.is_staff else regionB
    normalRegion = regionB if regionA.is_staff else regionA

    staff_y = staffRegion.min_y  # Da staffRegion.min_y == staffRegion.max_y

    if normalRegion.max_y < staff_y:
        # normalRegion liegt oben über dem Staff
        return staff_y - normalRegion.max_y
    elif normalRegion.min_y > staff_y:
        # normalRegion liegt unter dem Staff
        return normalRegion.min_y - staff_y
    else:
        # staff_y liegt zwischen [normalRegion.min_y, normalRegion.max_y]
        return 0.0


def build_fully_connected_graph(regions, x_min, x_max, y_top, y_bottom):
    # Erzeuge Staff-Regionen
    top_region = create_top_region(x_min, x_max, 0)  # is_staff=True
    bottom_region = create_bottom_region(x_min, x_max, y_bottom - y_top)  # is_staff=True

    all_regions = regions.copy()
    all_regions.append(top_region)
    all_regions.append(bottom_region)

    G = nx.Graph()
    for i, region in enumerate(all_regions):
        G.add_node(i, region=region)

    n = len(all_regions)
    for i in range(n):
        for j in range(i + 1, n):
            dist = distance_between_regions(all_regions[i], all_regions[j])
            G.add_edge(i, j, weight=dist)

    return G, all_regions


def create_top_region(x_min: int, x_max: int, y_min: int) -> ConnectedRegion:
    """
    Erzeugt eine ConnectedRegion, die den oberen Staff im Fenster repräsentiert.
    """
    # Wir nehmen zwei Pixel am linken und rechten Rand des oberen Staffs
    top_pixels = [(x_min, y_min), (x_max, y_min)]
    return ConnectedRegion(top_pixels, "Top", True)


def create_bottom_region(x_min: int, x_max: int, y_max: int) -> ConnectedRegion:
    """
    Erzeugt eine ConnectedRegion, die den unteren Staff im Fenster repräsentiert.
    """
    # Wir nehmen zwei Pixel am linken und rechten Rand des unteren Staffs
    bottom_pixels = [(x_min, y_max), (x_max, y_max)]
    return ConnectedRegion(bottom_pixels, "Bottom", True)

def staff_top_y(staff: dict) -> int:
    """
    Oberkante eines Staffs = Median aller y_start‑Werte der vertikalen Linien.
    """
    y_starts = [line[1] for vg in staff["vertical_groups"] for line in vg]
    return int(np.median(y_starts)) if y_starts else 0


def staff_bottom_y(staff: dict) -> int:
    """
    Unterkante eines Staffs = Median aller y_end‑Werte der vertikalen Linien.
    """
    y_ends = [line[2] for vg in staff["vertical_groups"] for line in vg]
    return int(np.median(y_ends)) if y_ends else 0

def staff_top_line_y(staff: dict) -> int:
    """Median Y of the top ~5 horizontal lines in the original image."""
    ys = sorted([y for hg in staff.get("horizontal_groups", []) for (y, _, _) in hg])
    top_lines = ys[:5]
    return int(np.median(top_lines)) if top_lines else -1

def staff_bottom_line_y(staff: dict) -> int:
    """Median Y of the bottom ~5 horizontal lines in the original image."""
    ys = sorted([y for hg in staff.get("horizontal_groups", []) for (y, _, _) in hg])
    bottom_lines = ys[-5:]
    return int(np.median(bottom_lines)) if bottom_lines else -1

def get_staff_rightmost_vertical_x(staff: dict) -> int:
    """Findet die maximale X-Koordinate aller vertikalen Liniensegmente."""
    max_x = -1
    # Stelle sicher, dass 'vertical_groups' existiert und eine Liste ist
    vertical_groups = staff.get("vertical_groups", [])
    if not isinstance(vertical_groups, list):
        return -1 # Oder löse einen Fehler aus, wenn die Struktur unerwartet ist

    for v_group in vertical_groups:
        # Stelle sicher, dass v_group eine Liste/Tupel ist
        if not isinstance(v_group, (list, tuple)): continue
        for line_segment in v_group:
             # Stelle sicher, dass line_segment ein Tupel mit mind. 1 Element ist
            if isinstance(line_segment, (list, tuple)) and len(line_segment) > 0:
                x = line_segment[0] # Das erste Element ist die x-Koordinate
                if isinstance(x, (int, float)): # Prüfe, ob es eine Zahl ist
                    max_x = max(max_x, int(x))
    return max_x