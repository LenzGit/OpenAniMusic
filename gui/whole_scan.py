#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import os
from typing import List

import cv2
import numpy as np
import argparse
import re  # Für die natürliche Sortierung

# Importiere die notwendigen Funktionen aus deinen Skripten
# Stelle sicher, dass diese Skripte im selben Verzeichnis sind oder im Python-Pfad
from detect_and_copy_staff_lines import (
    load_image, convert_to_black_and_white, dilate_horizontally,
    compute_overlap, can_connect, can_merge_with_group, find_horizontal_line_segments,
    merge_lines_connect, groups_can_connect, unify_groups_until_stable,
    highlight_groups_in_color_debug, extract_lines_as_transparent,
    filter_groups_by_x_thickness, filtered_groups_by_length, dilate_vertically,
    find_vertical_line_segments, merge_vertical_lines_connect, can_connect_vertical,
    can_merge_with_group_vertical, unify_vertical_groups_until_stable,
    groups_can_connect_vertical, filter_vertical_groups, find_staffs,
    group_staff_lines, export_staff_pixel_images,
    process_image_to_staff_strips  # Die neue Hauptfunktion für ein Bild
)

# Standard Verarbeitungsparameter (wie in der ursprünglichen main-Funktion)
DEFAULT_PROCESSING_PARAMS = {
    'threshold_value': 245,
    'horizontal_expansion': 4,
    'vertical_expansion': 4,
    'min_line_length_horizontal': 250,
    'min_line_length_vertical': 170,
    'max_thickness': 8,
    'thickness_threshold': 0.5,
    'min_group_length_horizontal': 500,
    'min_group_length_vertical': 500,
    'pad_top': 10,
    'pad_bottom': 10,
    'horizontal_gap_aligned': 1,
    'middle_overlap_aligned': 0,
    'horizontal_gap_transparent': 1,
    'apply_h_lowpass': False,
    'h_lowpass_kernel_x': 20,
    # Debug zeugs
    'save_debug_no_merge': False,
    'save_debug_merged_colorful': False,
    'save_debug_extracted_lines': False,
    'save_debug_pixel_exports': True,
    'cut_left_while_copy': None,
    'inter_page_gap': 0,
    'envelope_fallback_offset': 10,
}


def natural_sort_key(s):
    """Schlüssel für natürliche Sortierung (z.B. 'img1', 'img2', 'img10')."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'([0-9]+)', s)]


def concatenate_image_list_horizontally_aligned(image_strips_with_meta: list,
                                                cut_left_while_copy: List[int],  # Neuer Parameter
                                                is_transparent=False,
                                                inter_page_gap: int = 0,
                                                horizontal_expansion: int = 0):
    """
    Metadaten-Format pro Eintrag:
    {
        "strip_image": np.ndarray,
        "first_staff_eff_start_x_in_strip": int,
        "last_staff_eff_end_x_in_strip": int,
        "strip_center_y": int
    }
    """
    if not image_strips_with_meta: return None
    valid_items = [item for item in image_strips_with_meta if
                   item["strip_image"] is not None and item["strip_image"].size > 0]
    if not valid_items: return None

    max_dist_above_center_overall = 0
    max_dist_below_center_overall = 0
    for item in valid_items:
        h_strip = item["strip_image"].shape[0]
        center_y_in_strip = item["strip_center_y"]
        if center_y_in_strip < 0 or center_y_in_strip > h_strip:  # Sicherheitscheck für center_y
            print(
                f"Warnung: Ungültiger strip_center_y ({center_y_in_strip}) für Streifenhöhe {h_strip}. Verwende h_strip/2.")
            center_y_in_strip = h_strip // 2
        max_dist_above_center_overall = max(max_dist_above_center_overall, center_y_in_strip)
        max_dist_below_center_overall = max(max_dist_below_center_overall, h_strip - center_y_in_strip)

    final_canvas_h = math.ceil(max_dist_above_center_overall + max_dist_below_center_overall)
    if final_canvas_h <= 0: final_canvas_h = 50  # Fallback, falls Höhenberechnung fehlschlägt
    target_center_y_on_final_canvas = math.ceil(max_dist_above_center_overall)
    if target_center_y_on_final_canvas <= 0 and final_canvas_h > 0: target_center_y_on_final_canvas = final_canvas_h // 2

    canvas_parts = []
    current_canvas_width_edge = 0  # Rechte Kante des zuletzt platzierten *relevanten* Inhalts

    for i, item in enumerate(valid_items):
        strip_img = item["strip_image"]
        # x-Position des ersten Staffs IM originalen Seitenstreifen
        first_staff_start_x_in_orig_strip = item["first_staff_eff_start_x_in_strip"]
        # x-Position des Endes des letzten Staffs IM originalen Seitenstreifen
        last_staff_end_x_in_orig_strip = item["last_staff_eff_end_x_in_strip"]
        strip_center_y_local = item["strip_center_y"]
        first_global_idx = item["first_global_staff_index"]  # ①

        h_strip_orig, w_strip_orig = strip_img.shape[:2]
        if strip_center_y_local < 0 or strip_center_y_local > h_strip_orig:
            strip_center_y_local = h_strip_orig // 2

        # Kanalkonvertierung etc. (bleibt gleich)
        num_channels_img = strip_img.shape[2] if len(strip_img.shape) == 3 else 1
        fill_value_border = (0, 0, 0, 0) if is_transparent else (255, 255, 255)
        if is_transparent:
            if num_channels_img == 3:
                strip_img = cv2.cvtColor(strip_img, cv2.COLOR_BGR2BGRA)
            elif num_channels_img == 1:
                strip_img = cv2.cvtColor(strip_img, cv2.COLOR_GRAY2BGRA)
        else:
            if num_channels_img == 4:
                strip_img = cv2.cvtColor(strip_img, cv2.COLOR_BGRA2BGR)
            elif num_channels_img == 1:
                strip_img = cv2.cvtColor(strip_img, cv2.COLOR_GRAY2BGR)

        # --- Horizontales Zuschneiden und Platzierung ---
        actual_cut_left = 0  # Wie viel tatsächlich vom Originalstreifen links abgeschnitten wird

        if i == 0:
            # Erste Seite: nichts links abschneiden (außer dem, was schon im Streifen passiert ist)
            img_horizontal_part = strip_img
            x_on_canvas_for_this_part = 0
            # Die rechte Kante des relevanten Inhalts ist last_staff_end_x_in_orig_strip
            current_canvas_width_edge = last_staff_end_x_in_orig_strip
        else:
            try:
                cut_val = cut_left_while_copy[first_global_idx]
            except IndexError:
                raise IndexError(
                    f"cut_left_per_staff hat keinen Eintrag für globalen " +
                    f"Staff#{first_global_idx} (beginnt Seite{i}).")
            # Folgeseiten: cut_left_amount_between_pages vom ersten Staff dieser Seite anwenden
            # Start des Zuschneidens ist der Beginn des ersten Staffs PLUS der zusätzliche Cut
            start_crop_x_in_strip = first_staff_start_x_in_orig_strip

            # Sicherstellen, dass wir nicht über das Ende des relevanten Inhalts oder des Bildes hinaus schneiden
            # Die Breite des relevanten Inhalts im Originalstreifen ist:
            width_of_relevant_content_in_strip = last_staff_end_x_in_orig_strip - first_staff_start_x_in_orig_strip
            if width_of_relevant_content_in_strip <= 0:  # Sollte nicht passieren bei gültigen Streifen
                print(f"Warnung: Ungültige Metadaten für Streifen {i}, relevanter Inhalt hat Breite <=0.")
                img_horizontal_part = strip_img[:, first_staff_start_x_in_orig_strip:]  # Fallback: nur ab erstem Staff
            else:
                # Schneide nicht mehr ab, als der relevante Inhalt breit ist
                start_crop_x_in_strip = min(start_crop_x_in_strip,
                                            first_staff_start_x_in_orig_strip + width_of_relevant_content_in_strip - 1)
                # Schneide nicht mehr ab, als der Streifen breit ist
                start_crop_x_in_strip = min(start_crop_x_in_strip, w_strip_orig - 1)

            if start_crop_x_in_strip < 0: start_crop_x_in_strip = 0  # Darf nicht negativ sein

            img_horizontal_part = strip_img[:, start_crop_x_in_strip:]
            actual_cut_left = start_crop_x_in_strip  # So viel wurde vom *Originalstreifen* links weggeschnitten

            x_on_canvas_for_this_part = current_canvas_width_edge + inter_page_gap - horizontal_expansion

            # Die neue rechte Kante des relevanten Inhalts auf der Canvas:
            # Startposition + (Ende des letzten Staffs im Originalstreifen - tatsächlicher linker Schnitt)
            current_canvas_width_edge = x_on_canvas_for_this_part + (last_staff_end_x_in_orig_strip - actual_cut_left)

        if img_horizontal_part.size == 0:
            print(f"Warnung: Streifen {i} ist nach horizontalem Zuschnitt leer. Überspringe.")
            continue

        # --- Vertikales Padding und Platzierung ---
        h_part_unpadded = img_horizontal_part.shape[0]

        # y_start_on_final_canvas, damit strip_center_y_local auf target_center_y_on_final_canvas landet
        # strip_center_y_local ist relativ zur Oberkante des *Originalstreifens* (strip_img).
        # Wenn wir links geschnitten haben (actual_cut_left), ändert das nichts an der y-Koordinate der Mittellinie.
        y_offset_for_strip_content = target_center_y_on_final_canvas - strip_center_y_local

        part_for_final_canvas = np.full(
            (final_canvas_h, img_horizontal_part.shape[1], img_horizontal_part.shape[2]),
            fill_value_border, dtype=strip_img.dtype)

        src_y0, src_y1 = 0, h_part_unpadded
        dst_y0, dst_y1 = y_offset_for_strip_content, y_offset_for_strip_content + h_part_unpadded

        if dst_y0 < 0: src_y0 = -dst_y0; dst_y0 = 0
        if dst_y1 > final_canvas_h: src_y1 = h_part_unpadded - (dst_y1 - final_canvas_h); dst_y1 = final_canvas_h

        if dst_y1 > dst_y0 and src_y1 > src_y0 and (dst_y1 - dst_y0) == (src_y1 - src_y0):
            part_for_final_canvas[dst_y0:dst_y1, :] = img_horizontal_part[src_y0:src_y1, :]
        else:
            if not (dst_y1 > dst_y0 and src_y1 > src_y0):
                print(
                    f"Warnung: Kein vertikaler Inhalt zum Kopieren für Streifen {i} (dst_y1={dst_y1} <= dst_y0={dst_y0} oder src_y1={src_y1} <= src_y0={src_y0})")
            else:
                print(
                    f"Warnung: Mismatch bei vertikalen Slice-Höhen für Streifen {i}: dst_h={(dst_y1 - dst_y0)}, src_h={(src_y1 - src_y0)}")

        canvas_parts.append({
            "image_part": part_for_final_canvas,
            "x_on_canvas": x_on_canvas_for_this_part
        })

    if not canvas_parts: return None
    # Die Gesamtbreite ist die rechte Kante des letzten platzierten Teils
    # ODER die current_canvas_width_edge, wenn der letzte Staff nicht bis zum Rand ging
    # Wir nehmen die tatsächliche Breite der zusammengesetzten Teile
    final_total_width = 0
    if canvas_parts:
        last_part_info = canvas_parts[-1]
        final_total_width = last_part_info["x_on_canvas"] + last_part_info["image_part"].shape[1]

    if final_total_width <= 0 or final_canvas_h <= 0:
        print("Warnung: Finale Canvas Dimensionen sind ungültig (<=0).")
        return None

    # Erstelle die finale Leinwand
    num_channels_final = 4 if is_transparent else 3
    dtype_final = np.uint8
    fill_color = (0, 0, 0, 0) if is_transparent else (255, 255, 255)
    final_canvas = np.full((final_canvas_h, final_total_width, num_channels_final), fill_color, dtype=dtype_final)

    # Platziere die vorbereiteten Streifen
    for part_info in canvas_parts:
        img_part_to_paste = part_info["image_part"]
        x_start_paste = part_info["x_on_canvas"]

        h_p, w_p = img_part_to_paste.shape[:2]

        y0_fc, y1_fc = 0, h_p  # Da img_part_to_paste bereits die final_canvas_h hat
        x0_fc, x1_fc = x_start_paste, x_start_paste + w_p

        # Sicherstellen, dass wir innerhalb der final_canvas Grenzen bleiben
        if y1_fc > final_canvas_h: y1_fc = final_canvas_h
        if x1_fc > final_total_width: x1_fc = final_total_width

        # Der img_part_to_paste hat bereits die richtige Höhe (final_canvas_h)
        # Wir müssen nur den Breiten-Teil korrekt kopieren.
        paste_width_on_canvas = x1_fc - x0_fc

        if paste_width_on_canvas > 0 and (y1_fc - y0_fc) > 0:
            try:
                final_canvas[y0_fc:y1_fc, x0_fc:x1_fc] = img_part_to_paste[0:(y1_fc - y0_fc), 0:paste_width_on_canvas]
            except ValueError as e:
                print(f"ValueError beim finalen Pasten: {e}")
                print(f"  Target Shape: {final_canvas[y0_fc:y1_fc, x0_fc:x1_fc].shape}")
                print(f"  Source Shape: {img_part_to_paste[0:(y1_fc - y0_fc), 0:paste_width_on_canvas].shape}")
    return final_canvas


def batch_process(input_dir: str, output_dir: str, params: dict):
    """
    Verarbeitet alle Bilder in input_dir und speichert Ergebnisse in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    staff_counter_global = 0
    image_files = []
    for f_name in os.listdir(input_dir):
        if f_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_files.append(f_name)

    image_files.sort(key=natural_sort_key)  # Sortiert "1.png", "2.png", "10.png" korrekt

    if not image_files:
        print(f"Keine Bilddateien im Ordner {input_dir} gefunden.")
        return

    print(f"Gefundene Bilddateien zur Verarbeitung: {len(image_files)}")

    all_aligned_strips_with_meta = []
    all_transparent_strips_with_meta = []

    for img_filename in image_files:
        image_path = os.path.join(input_dir, img_filename)
        page_name = os.path.splitext(img_filename)[0]
        output_dir_for_page = os.path.join(output_dir, f"page_results_{page_name}")

        aligned_strip, transparent_strip, page_meta, staffs_current_page = process_image_to_staff_strips(  # Empfange Metadaten
            image_path,
            output_dir_for_page,
            params,
            staff_counter_global
        )


        if aligned_strip is not None and aligned_strip.size > 0:
            all_aligned_strips_with_meta.append({
                "strip_image": aligned_strip,
                "first_staff_eff_start_x_in_strip": page_meta["aligned_strip_first_staff_eff_start_x_in_strip"],
                "last_staff_eff_end_x_in_strip": page_meta["aligned_strip_last_staff_eff_end_x_in_strip"],
                "strip_center_y": page_meta["aligned_strip_center_y"],  # NEU
                "first_global_staff_index": staff_counter_global
            })
        else:
            print(f"  Kein 'aligned_strip' für {img_filename} erhalten oder er ist leer.")

        if transparent_strip is not None and transparent_strip.size > 0:
            all_transparent_strips_with_meta.append({
                "strip_image": transparent_strip,
                "first_staff_eff_start_x_in_strip": page_meta["transparent_strip_first_staff_eff_start_x_in_strip"],
                "last_staff_eff_end_x_in_strip": page_meta["transparent_strip_last_staff_eff_end_x_in_strip"],
                "strip_center_y": page_meta["transparent_strip_center_y"],  # NEU
                "first_global_staff_index": staff_counter_global
            })
        else:
            print(f"  Kein 'transparent_strip' für {img_filename} erhalten oder er ist leer.")

        staff_counter_global += staffs_current_page

    print("\nBatch-Verarbeitung abgeschlossen. Führe finale Konkatenation durch...")

    if len(params['cut_left_while_copy']) < staff_counter_global-1:
        raise ValueError(
            f"cut_left_per_staff hat {len(params['cut_left_while_copy'])} Einträge, "
            f"benötigt werden aber {staff_counter_global-1}."
        )

    if all_aligned_strips_with_meta:
        print(f"Konkateniere {len(all_aligned_strips_with_meta)} 'aligned' Streifen...")
        grand_aligned_image = concatenate_image_list_horizontally_aligned(
            all_aligned_strips_with_meta,
            cut_left_while_copy=params['cut_left_while_copy'],  # HIER übergeben
            is_transparent=False,
            inter_page_gap=params.get('inter_page_gap', 0),
            horizontal_expansion=params.get('horizontal_expansion', 0),# Ggf. inter_page_gap anpassen
        )
        if grand_aligned_image is not None and grand_aligned_image.size > 0:  # Prüfung hinzugefügt
            output_path_grand_aligned = os.path.join(output_dir, "GRAND_staffs_concat_aligned.png")
            cv2.imwrite(output_path_grand_aligned, grand_aligned_image)
            print(f"  -> Gesamtbild 'aligned' gespeichert: {output_path_grand_aligned}")
        else:
            print("  Fehler: Das finale 'aligned' Bild konnte nicht erstellt werden oder ist leer.")
    else:
        print("Keine 'aligned' Streifen zum Konkatenieren für das Gesamtbild vorhanden.")

    if all_transparent_strips_with_meta:
        print(f"Konkateniere {len(all_transparent_strips_with_meta)} 'transparent' Streifen...")
        grand_transparent_image = concatenate_image_list_horizontally_aligned(
            all_transparent_strips_with_meta,
            cut_left_while_copy=params['cut_left_while_copy'],  # HIER übergeben
            is_transparent=True,
            inter_page_gap=params.get('inter_page_gap', 0),
            horizontal_expansion=params.get('horizontal_expansion', 0),  # Ggf. inter_page_gap anpassen
        )
        if grand_transparent_image is not None and grand_transparent_image.size > 0:  # Prüfung hinzugefügt
            output_path_grand_transparent = os.path.join(output_dir, "GRAND_staffs_concat_lines_transparent.png")
            cv2.imwrite(output_path_grand_transparent, grand_transparent_image)
            print(f"  -> Gesamtbild 'transparent' gespeichert: {output_path_grand_transparent}")
        else:
            print("  Fehler: Das finale 'transparent' Bild konnte nicht erstellt werden oder ist leer.")
    else:
        print("Keine 'transparent' Streifen zum Konkatenieren für das Gesamtbild vorhanden.")

    print("\nFertig!")



# In batch_process_sheets.py

# ... (alle imports und Funktionen darüber bleiben gleich) ...

if __name__ == "__main__":
    input_test_dir = r"..."
    output_test_dir = r"..."

    print(f"--- TESTMODUS ---")
    print(f"Eingabeordner (Test): {os.path.abspath(input_test_dir)}")
    print(f"Ausgabeordner (Test): {os.path.abspath(output_test_dir)}")


    DEFAULT_PROCESSING_PARAMS['cut_left_while_copy']= [150, 150, 150, 150, 150, 150,
                                                       150, 150, 150, 150, 150, 150,
                                                       150, 150, 150, 150, 150, 150,
                                                       150, 150, 150, 150, 150, 150]

    # Stelle sicher, dass der Test-Eingabeordner existiert
    if not os.path.isdir(input_test_dir):
        print(f"FEHLER: Der Test-Eingabeordner '{input_test_dir}' wurde nicht gefunden.")
        print(f"Bitte erstelle ihn und lege Testbilder (z.B. 1.png, 2.png) hinein.")
    else:
        # Verwende die Standardparameter, diese könnten auch aus einer Konfigurationsdatei geladen werden
        processing_params = DEFAULT_PROCESSING_PARAMS.copy()

        # Beispiel: Debug-Ausgaben für einzelne Seiten aktivieren/deaktivieren für den Test
        # Wenn du viele Debug-Bilder für jede Seite sehen willst:
        processing_params['save_debug_no_merge'] = True
        processing_params['save_debug_merged_colorful'] = True

        batch_process(input_test_dir, output_test_dir, processing_params)