#!/usr/bin/env python3
import os
import sys
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import cv2
import matplotlib.pyplot as plt


@dataclass
class AxisLimits:
    x_min: float = 0.0
    x_max: float = 3.6
    y_min: float = 0.0
    y_max: float = 3000.0


@dataclass
class ExtractionResult:
    v25: np.ndarray
    i25: np.ndarray
    v125: np.ndarray
    i125: np.ndarray
    v175: np.ndarray
    i175: np.ndarray


def read_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return img


def find_plot_bbox(gray: np.ndarray) -> Tuple[int, int, int, int]:
    h, w = gray.shape
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180.0, threshold=300, minLineLength=int(min(h, w) * 0.6), maxLineGap=10
    )
    if lines is None:
        # fallback to projection
        th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        col_sum = th.sum(axis=0)
        row_sum = th.sum(axis=1)
        left = int(np.argmax(col_sum[: int(w * 0.2)]))
        right = int(np.argmax(col_sum[int(w * 0.8) :])) + int(w * 0.8)
        top = int(np.argmax(row_sum[: int(h * 0.2)]))
        bottom = int(np.argmax(row_sum[int(h * 0.6) :])) + int(h * 0.6)
        return left + 3, top + 3, right - 3, bottom - 3

    verticals: List[Tuple[int, int, int, int]] = []
    horizontals: List[Tuple[int, int, int, int]] = []
    for l in lines[:, 0, :]:
        x1, y1, x2, y2 = l
        if abs(x1 - x2) <= 5:
            verticals.append((x1, y1, x2, y2))
        if abs(y1 - y2) <= 5:
            horizontals.append((x1, y1, x2, y2))

    if len(verticals) < 2 or len(horizontals) < 2:
        th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        col_sum = th.sum(axis=0)
        row_sum = th.sum(axis=1)
        left = int(np.argmax(col_sum[: int(w * 0.2)]))
        right = int(np.argmax(col_sum[int(w * 0.8) :])) + int(w * 0.8)
        top = int(np.argmax(row_sum[: int(h * 0.2)]))
        bottom = int(np.argmax(row_sum[int(h * 0.6) :])) + int(h * 0.6)
        return left + 3, top + 3, right - 3, bottom - 3

    left = min(v[0] for v in verticals)
    right = max(v[0] for v in verticals)
    top = min(hh[1] for hh in horizontals)
    bottom = max(hh[1] for hh in horizontals)

    inset = 4
    x0 = max(0, left + inset)
    y0 = max(0, top + inset)
    x1 = min(w - 1, right - inset)
    y1 = min(h - 1, bottom - inset)

    if x0 >= x1 or y0 >= y1:
        raise RuntimeError("Detected invalid plot bbox")
    return x0, y0, x1, y1


def remove_grid_and_axes(plot_gray: np.ndarray) -> np.ndarray:
    thr = cv2.threshold(plot_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    horiz = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel_h)
    vert = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel_v)
    grid = cv2.bitwise_or(horiz, vert)

    curves = cv2.bitwise_and(thr, cv2.bitwise_not(grid))
    curves = cv2.medianBlur(curves, 3)
    return curves


def skeletonize_binary(binary_img: np.ndarray) -> np.ndarray:
    # Morphological skeletonization using OpenCV only
    img = (binary_img > 0).astype(np.uint8) * 255
    skel = np.zeros_like(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(img, kernel)
        opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel)
        temp = cv2.subtract(eroded, opened)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break
    return skel


def group_contiguous(indices: np.ndarray, max_gap: int = 2) -> List[np.ndarray]:
    if indices.size == 0:
        return []
    groups: List[List[int]] = [[int(indices[0])]]
    for idx in indices[1:]:
        if int(idx) - groups[-1][-1] <= max_gap:
            groups[-1].append(int(idx))
        else:
            groups.append([int(idx)])
    return [np.array(g, dtype=int) for g in groups]


def assign_tracks(candidates: List[Optional[int]], last_ys: List[Optional[float]]) -> List[Optional[int]]:
    sorted_pairs = sorted([(i, y) for i, y in enumerate(last_ys)], key=lambda t: (float('inf') if t[1] is None else t[1]))
    result: List[Optional[int]] = [None, None, None]
    cand_iter = iter(candidates)
    for track_idx, _ in sorted_pairs:
        try:
            result[track_idx] = next(cand_iter)
        except StopIteration:
            result[track_idx] = None
    return result


def extract_curves(plot_gray: np.ndarray, axis: AxisLimits):
    h, w = plot_gray.shape
    bin_no_grid = remove_grid_and_axes(plot_gray)
    skel = skeletonize_binary(bin_no_grid)

    tracks_x: List[List[int]] = [[], [], []]
    tracks_y: List[List[int]] = [[], [], []]
    last_y: List[Optional[float]] = [None, None, None]

    for x in range(w):
        ys = np.where(skel[:, x] > 0)[0]
        if ys.size == 0:
            continue
        groups = group_contiguous(ys, max_gap=2)
        groups = sorted(groups, key=lambda g: g.size, reverse=True)[:3]
        cand_y = sorted([int(np.mean(g)) for g in groups])

        if all(v is None for v in last_y):
            for t_idx, yv in enumerate(cand_y[:3]):
                tracks_x[t_idx].append(x)
                tracks_y[t_idx].append(yv)
                last_y[t_idx] = float(yv)
            continue

        assignment = assign_tracks(cand_y, last_y)
        for t_idx in range(3):
            cy = assignment[t_idx]
            if cy is not None:
                tracks_x[t_idx].append(x)
                tracks_y[t_idx].append(cy)
                last_y[t_idx] = float(cy)

    def px_to_data(xs: List[int], ys: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        xs_arr = np.array(xs, dtype=float)
        ys_arr = np.array(ys, dtype=float)
        vx = axis.x_min + (xs_arr / (w - 1)) * (axis.x_max - axis.x_min)
        ic = axis.y_max - (ys_arr / (h - 1)) * (axis.y_max - axis.y_min)
        return vx, ic

    data_tracks = [px_to_data(tracks_x[i], tracks_y[i]) for i in range(3)]
    means = [dt[0].mean() if dt[0].size > 0 else np.inf for dt in data_tracks]
    order = np.argsort(means)

    v25, i25 = data_tracks[order[0]]
    v125, i125 = data_tracks[order[1]]
    v175, i175 = data_tracks[order[2]]

    return ExtractionResult(v25=v25, i25=i25, v125=v125, i125=i125, v175=v175, i175=i175)


def visualize(img_rgb: np.ndarray, bbox: Tuple[int, int, int, int], res: ExtractionResult, out_path: str) -> None:
    x0, y0, x1, y1 = bbox
    plt.figure(figsize=(8, 6), dpi=150)
    plot_rgb = img_rgb[y0:y1, x0:x1, :]
    plt.imshow(cv2.cvtColor(plot_rgb, cv2.COLOR_BGR2RGB))

    h, w, _ = plot_rgb.shape
    def data_to_px(v, i, axis: AxisLimits) -> Tuple[np.ndarray, np.ndarray]:
        xs = (v - axis.x_min) / (axis.x_max - axis.x_min) * (w - 1)
        ys = (axis.y_max - i) / (axis.y_max - axis.y_min) * (h - 1)
        return xs, ys

    axis = AxisLimits()
    x25, y25 = data_to_px(res.v25, res.i25, axis)
    x125, y125 = data_to_px(res.v125, res.i125, axis)
    x175, y175 = data_to_px(res.v175, res.i175, axis)

    plt.plot(x25, y25, 'r.', ms=1, label='Tj=25C (extracted)')
    plt.plot(x125, y125, 'g.', ms=1, label='Tj=125C (extracted)')
    plt.plot(x175, y175, 'b.', ms=1, label='Tj=175C (extracted)')
    plt.legend(loc='lower right')
    plt.axis('off')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def save_csv(res: ExtractionResult, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    def write_csv(path: str, v: np.ndarray, i: np.ndarray):
        arr = np.column_stack([v, i])
        header = 'VCE_V,IC_A'
        np.savetxt(path, arr, delimiter=',', header=header, comments='', fmt='%.6f')

    write_csv(os.path.join(out_dir, 'curve_Tj25C.csv'), res.v25, res.i25)
    write_csv(os.path.join(out_dir, 'curve_Tj125C.csv'), res.v125, res.i125)
    write_csv(os.path.join(out_dir, 'curve_Tj175C.csv'), res.v175, res.i175)


def main():
    parser = argparse.ArgumentParser(description="Extract dense data from an IGBT output characteristic plot image.")
    parser.add_argument("--image", type=str, default="/workspace/data/igbt_output_characteristic.png", help="Path to the plot image")
    parser.add_argument("--xlim", type=float, nargs=2, default=[0.0, 3.6], help="X axis limits (VCE min max)")
    parser.add_argument("--ylim", type=float, nargs=2, default=[0.0, 3000.0], help="Y axis limits (IC min max)")
    parser.add_argument("--outdir", type=str, default="/workspace/outputs", help="Directory to save CSV and figures")
    args = parser.parse_args()

    img = read_image(args.image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    bbox = find_plot_bbox(gray)
    x0, y0, x1, y1 = bbox
    plot_gray = gray[y0:y1, x0:x1]

    axis = AxisLimits(x_min=args.xlim[0], x_max=args.xlim[1], y_min=args.ylim[0], y_max=args.ylim[1])
    result = extract_curves(plot_gray, axis)

    save_csv(result, args.outdir)
    visualize(img, bbox, result, os.path.join(args.outdir, "extracted_overlay.png"))

    plt.figure(figsize=(6, 5), dpi=150)
    plt.plot(result.v25, result.i25, '.', ms=1, label='Tj=25C')
    plt.plot(result.v125, result.i125, '.', ms=1, label='Tj=125C')
    plt.plot(result.v175, result.i175, '.', ms=1, label='Tj=175C')
    plt.xlabel('VCE [V]')
    plt.ylabel('IC [A]')
    plt.xlim(axis.x_min, axis.x_max)
    plt.ylim(axis.y_min, axis.y_max)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "extracted_curves.png"))
    plt.close()

    print(f"Saved CSVs and figures to: {args.outdir}")


if __name__ == "__main__":
    main()