from __future__ import annotations

# BlobTracker Live — Streamlit App (One‑Page, ROI, Colors, Grid, Curves)
# ---------------------------------------------------------------------------------
# What this version adds
# • One‑page layout (no scroll): left=Source+Detector+ROI, center=Preview+Overlay
#   controls, right=Export.
# • ROI effects: invert and zoom inside ROI (rect or polygon typed as % coords).
# • Grid overlay: toggle + divisions slider.
# • Curved links: quadratic‑bezier style with curvature control.
# • Color controls and style knobs (point scale, line thickness).
# • Deprecation fix: uses `use_container_width=True`.
# • Sandbox fallback & unit tests preserved.
# ---------------------------------------------------------------------------------
REQUIREMENTS = "streamlit opencv-python numpy"

import os
import sys
import tempfile
from dataclasses import dataclass
from typing import List, Sequence, Tuple

# ---- sandbox detection -----------------------------------------------------------

def _running_in_web_sandbox() -> bool:
    try:
        return sys.platform == "emscripten" or os.environ.get("PYODIDE") is not None
    except Exception:
        return False

# ---- optional imports ------------------------------------------------------------
STREAMLIT_AVAILABLE = False
OPENCV_AVAILABLE = False
NUMPY_AVAILABLE = False
IMPORT_ERROR = None

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except Exception as _e:
    IMPORT_ERROR = _e

try:
    import cv2
    OPENCV_AVAILABLE = True
except Exception as _e:
    IMPORT_ERROR = _e if IMPORT_ERROR is None else IMPORT_ERROR

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except Exception as _e:
    IMPORT_ERROR = _e if IMPORT_ERROR is None else IMPORT_ERROR

# ---- pure-python link logic + tests ---------------------------------------------
@dataclass
class LinkConfig:
    k: int = 3
    max_dist: float = 180

def _sq_dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx, dy = a[0] - b[0], a[1] - b[1]
    return dx * dx + dy * dy

def compute_links(points: Sequence[Tuple[float, float]], cfg: LinkConfig) -> List[Tuple[int, int]]:
    n = len(points)
    if n <= 1:
        return []
    d2 = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            s = _sq_dist(points[i], points[j])
            d2[i][j] = d2[j][i] = s
    neighbors: List[List[int]] = []
    for i in range(n):
        order = list(range(n))
        order.remove(i)
        order.sort(key=lambda j: (d2[i][j], j))
        neighbors.append(order)
    links: set[Tuple[int, int]] = set()
    degree = [0] * n
    maxd2 = cfg.max_dist * cfg.max_dist
    for i in range(n):
        for j in neighbors[i]:
            if degree[i] >= cfg.k:
                break
            if d2[i][j] <= maxd2 and degree[j] < cfg.k:
                e = (i, j) if i < j else (j, i)
                if e not in links:
                    links.add(e)
                    degree[i] += 1
                    degree[j] += 1
    return sorted(links)

# ---- tests ----------------------------------------------------------------------

def _run_tests() -> None:
    import unittest

    class T(unittest.TestCase):
        def test_empty(self):
            self.assertEqual(compute_links([], LinkConfig()), [])
            self.assertEqual(compute_links([(0, 0)], LinkConfig()), [])

        def test_line_chain(self):
            pts = [(x * 10.0, 0.0) for x in range(5)]
            self.assertEqual(len(compute_links(pts, LinkConfig(k=2, max_dist=50))), 4)

        def test_far(self):
            self.assertEqual(
                compute_links([(0, 0), (500, 0), (0, 500)], LinkConfig(k=3, max_dist=100)),
                [],
            )

        def test_square_k1(self):
            pts = [(0, 0), (0, 10), (10, 0), (10, 10)]
            self.assertEqual(len(compute_links(pts, LinkConfig(k=1, max_dist=15))), 2)

        def test_no_oversubscription(self):
            pts = [(x * 5.0, (x % 2) * 5.0) for x in range(12)]
            edges = compute_links(pts, LinkConfig(k=2, max_dist=20))
            deg = [0] * len(pts)
            for i, j in edges:
                deg[i] += 1
                deg[j] += 1
            self.assertTrue(all(d <= 2 for d in deg))

        def test_boundary_ok(self):
            pts = [(0, 0), (30, 0), (60, 0)]
            e = compute_links(pts, LinkConfig(k=1, max_dist=30))
            self.assertTrue(((0, 1) in e) or ((1, 2) in e))

    r = unittest.TextTestRunner(verbosity=1).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(T)
    )
    if not r.wasSuccessful():
        raise SystemExit(1)

# ---- streamlit app ---------------------------------------------------------------
if STREAMLIT_AVAILABLE and OPENCV_AVAILABLE and NUMPY_AVAILABLE:
    # helpers
    def _ui_toggle(label: str, value: bool = False) -> bool:
        try:
            return st.toggle(label, value=value)
        except Exception:
            return st.checkbox(label, value=value)

    def _hex_to_bgr(hx: str) -> Tuple[int, int, int]:
        hx = hx.lstrip("#")
        return int(hx[4:6], 16), int(hx[2:4], 16), int(hx[0:2], 16)

    @dataclass
    class Overlays:
        show_points: bool
        show_labels: bool
        show_lines: bool
        show_boxes: bool
        line_k: int
        max_link: int
        line_thick: int
        point_scale: float
        link_mode: str  # 'nearest' or 'anchor'
        anchor_xy: Tuple[float, float]  # normalized (0..1)
        color_points: Tuple[int, int, int]
        color_lines: Tuple[int, int, int]
        color_boxes: Tuple[int, int, int]
        color_labels: Tuple[int, int, int]
        roi_mode: str  # 'none' | 'rect' | 'poly'
        roi_rect: Tuple[float, float, float, float]  # x,y,w,h in 0..1
        roi_poly: List[Tuple[float, float]]  # list of (x,y) in 0..1
        show_roi: bool
        roi_invert: bool
        roi_zoom: bool
        roi_zoom_factor: float
        show_grid: bool
        grid_divisions: int
        curved_lines: bool
        curvature: float

    def build_detector(
        min_area: float,
        max_area: float,
        min_th: int,
        max_th: int,
        filter_circ: bool,
        min_circ: float,
    ):
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = int(min_th)
        params.maxThreshold = int(max_th)
        params.filterByArea = True
        params.minArea = max(1.0, float(min_area))
        params.maxArea = max(params.minArea + 1.0, float(max_area))
        params.filterByColor = False
        params.filterByConvexity = False
        params.filterByInertia = False
        params.filterByCircularity = bool(filter_circ)
        params.minCircularity = float(min_circ)
        return cv2.SimpleBlobDetector_create(params)

    def preprocess(
        img: np.ndarray, blur: int, use_clahe: bool, invert: bool, overlays: Overlays
    ) -> Tuple[np.ndarray, np.ndarray]:
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if blur > 0:
            k = max(1, int(blur) // 2 * 2 + 1)
            gray = cv2.GaussianBlur(gray, (k, k), 0)
        if use_clahe:
            gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        if invert:
            gray = cv2.bitwise_not(gray)
        # ROI mask
        mask = np.ones((h, w), np.uint8) * 255
        if overlays.roi_mode == "rect":
            x, y, wf, hf = overlays.roi_rect
            x1 = int(x * w)
            y1 = int(y * h)
            x2 = int(min(1.0, x + wf) * w)
            y2 = int(min(1.0, y + hf) * h)
            mask[:, :] = 0
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        elif overlays.roi_mode == "poly" and overlays.roi_poly:
            pts = np.array(
                [[int(px * w), int(py * h)] for px, py in overlays.roi_poly],
                dtype=np.int32,
            )
            mask[:, :] = 0
            cv2.fillPoly(mask, [pts], 255)
        gray_masked = cv2.bitwise_and(gray, mask)
        return gray_masked, mask

    def keypoints_to_xy(keypoints: List[cv2.KeyPoint]) -> np.ndarray:
        if not keypoints:
            return np.zeros((0, 2), dtype=np.float32)
        return np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints], dtype=np.float32)

    def _draw_grid(img: np.ndarray, divisions: int, color=(80, 80, 80)) -> None:
        h, w = img.shape[:2]
        step_x = max(1, w // divisions)
        step_y = max(1, h // divisions)
        for x in range(step_x, w, step_x):
            cv2.line(img, (x, 0), (x, h), color, 1, cv2.LINE_AA)
        for y in range(step_y, h, step_y):
            cv2.line(img, (0, y), (w, y), color, 1, cv2.LINE_AA)

    def _draw_curve(img, p1, p2, color, thickness, curvature) -> None:
        x1, y1 = p1
        x2, y2 = p2
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2
        dx, dy = x2 - x1, y2 - y1
        dist = max(1.0, (dx * dx + dy * dy) ** 0.5)
        nx, ny = -dy / dist, dx / dist
        cx = int(mx + nx * curvature * dist)
        cy = int(my + ny * curvature * dist)
        pts = []
        for t in np.linspace(0, 1, 30):
            xt = int((1 - t) * (1 - t) * x1 + 2 * (1 - t) * t * cx + t * t * x2)
            yt = int((1 - t) * (1 - t) * y1 + 2 * (1 - t) * t * cy + t * t * y2)
            pts.append([xt, yt])
        pts = np.array(pts, dtype=np.int32)
        cv2.polylines(img, [pts], False, color, thickness, cv2.LINE_AA)

    def _apply_roi_visual_effects(
        base: np.ndarray, mask: np.ndarray, overlays: Overlays
    ) -> np.ndarray:
        out = base.copy()
        if overlays.roi_invert:
            inv = cv2.bitwise_not(out)
            out = (
                cv2.bitwise_and(
                    out, cv2.cvtColor(255 - mask, cv2.COLOR_GRAY2BGR)
                )
                + cv2.bitwise_and(inv, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
            )
        if overlays.roi_zoom:
            yx = np.where(mask > 0)
            if yx[0].size > 0:
                y1, y2 = int(np.min(yx[0])), int(np.max(yx[0]))
                x1, x2 = int(np.min(yx[1])), int(np.max(yx[1]))
                roi = out[y1 : y2 + 1, x1 : x2 + 1]
                if roi.size > 0:
                    zoom = max(1.0, overlays.roi_zoom_factor)
                    zr = cv2.resize(
                        roi, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC
                    )
                    zh, zw = zr.shape[:2]
                    th, tw = y2 - y1 + 1, x2 - x1 + 1
                    sy, sx = max(0, (zh - th) // 2), max(0, (zw - tw) // 2)
                    zr = zr[sy : sy + th, sx : sx + tw]
                    out[y1 : y2 + 1, x1 : x2 + 1] = zr
        return out

    def draw_overlays(
        frame: np.ndarray, keypoints: List[cv2.KeyPoint], overlays: Overlays, mask: np.ndarray
    ) -> np.ndarray:
        base = _apply_roi_visual_effects(frame, mask, overlays)
        out = base.copy()
        h, w = out.shape[:2]
        if overlays.show_grid:
            _draw_grid(out, max(2, overlays.grid_divisions))
        if overlays.show_roi and overlays.roi_mode != "none":
            roi_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            out = cv2.addWeighted(out, 1.0, (roi_vis * 0.25).astype(out.dtype), 0.25, 0)
        if overlays.show_points:
            for kp in keypoints:
                r = max(1, int(kp.size * 0.5 * overlays.point_scale))
                cv2.circle(out, (int(kp.pt[0]), int(kp.pt[1])), r, overlays.color_points, 1, cv2.LINE_AA)
        pts = keypoints_to_xy(keypoints)
        if overlays.show_boxes:
            for kp in keypoints:
                s = int(kp.size)
                x, y = int(kp.pt[0]), int(kp.pt[1])
                cv2.rectangle(out, (x - s // 2, y - s // 2), (x + s // 2, y + s // 2), overlays.color_boxes, 1)
        if overlays.show_lines and len(pts) >= 1:
            if overlays.link_mode == "anchor":
                ax = int(overlays.anchor_xy[0] * w)
                ay = int(overlays.anchor_xy[1] * h)
                for p in pts:
                    if (p[0] - ax) ** 2 + (p[1] - ay) ** 2 <= overlays.max_link ** 2:
                        if overlays.curved_lines:
                            _draw_curve(
                                out,
                                (ax, ay),
                                (int(p[0]), int(p[1])),
                                overlays.color_lines,
                                overlays.line_thick,
                                overlays.curvature,
                            )
                        else:
                            cv2.line(
                                out,
                                (ax, ay),
                                (int(p[0]), int(p[1])),
                                overlays.color_lines,
                                overlays.line_thick,
                                cv2.LINE_AA,
                            )
                cv2.circle(out, (ax, ay), 6, overlays.color_lines, -1, cv2.LINE_AA)
            else:
                d2 = np.sum((pts[None, :, :] - pts[:, None, :]) ** 2, axis=2)
                for i in range(len(pts)):
                    idx = np.argsort(d2[i])
                    links = 0
                    for j in idx[1:]:
                        if links >= overlays.line_k:
                            break
                        if d2[i, j] <= overlays.max_link ** 2:
                            p1 = tuple(np.int32(pts[i]))
                            p2 = tuple(np.int32(pts[j]))
                            if overlays.curved_lines:
                                _draw_curve(out, p1, p2, overlays.color_lines, overlays.line_thick, overlays.curvature)
                            else:
                                cv2.line(out, p1, p2, overlays.color_lines, overlays.line_thick, cv2.LINE_AA)
                            links += 1
        if overlays.show_labels:
            for n, kp in enumerate(keypoints):
                x, y = int(kp.pt[0]), int(kp.pt[1])
                cv2.putText(
                    out,
                    f"[{n}] {x},{y}",
                    (x + 6, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    overlays.color_labels,
                    1,
                    cv2.LINE_AA,
                )
        return out

    def parse_poly(text: str) -> List[Tuple[float, float]]:
        pts: List[Tuple[float, float]] = []
        for part in text.split(";"):
            part = part.strip()
            if not part:
                continue
            try:
                x, y = part.split(",")
                fx, fy = float(x) / 100.0, float(y) / 100.0
                if 0 <= fx <= 1 and 0 <= fy <= 1:
                    pts.append((fx, fy))
            except Exception:
                pass
        return pts

    def detect_on_frame(
        frame: np.ndarray,
        detector,
        blur: int,
        use_clahe: bool,
        invert: bool,
        overlays: Overlays,
    ):
        gray, mask = preprocess(frame, blur, use_clahe, invert, overlays)
        kps = detector.detect(gray)
        rendered = draw_overlays(frame, kps, overlays, mask)
        return rendered, kps

    def run_streamlit_app() -> None:
        st.set_page_config(page_title="BlobTracker Live", page_icon="⚪", layout="wide")
        st.markdown(
            """
            <style>
              .block-container{padding-top:.8rem;padding-bottom:.8rem;}
              main .block-container{max-height:92vh; overflow:hidden}
            </style>
            """,
            unsafe_allow_html=True,
        )

        left, center, right = st.columns([1.3, 2.4, 1.5], vertical_alignment="top")

        with left:
            st.markdown("## ⚪ BlobTracker Live")
            media = st.file_uploader(
                "Upload video/image",
                type=["mp4", "mov", "avi", "mkv", "png", "jpg", "jpeg"],
                accept_multiple_files=False,
            )

            st.markdown("#### Preprocess")
            c1, c2, c3 = st.columns(3)
            blur = c1.slider("Blur", 0, 31, 3, 1)
            use_clahe = c2.toggle("CLAHE", True)
            invert = c3.toggle("Invert", False)

            st.markdown("#### Detector")
            d1, d2 = st.columns(2)
            min_area = d1.slider("Min area", 10, 10000, 120, 10)
            max_area = d2.slider("Max area", 500, 50000, 4000, 100)
            th1, th2 = st.columns(2)
            min_th = th1.slider("Min thr", 0, 255, 10)
            max_th = th2.slider("Max thr", 1, 255, 200)
            fc, mc = st.columns(2)
            filter_circularity = fc.toggle("By circularity", False)
            min_circ = mc.slider("Min circ", 0.0, 1.0, 0.7, 0.05)

            st.markdown("#### ROI (Region)")
            roi_mode = st.selectbox("Mode", ["none", "rect", "poly"], index=0)
            show_roi = st.toggle("Show mask", True)
            roi_rect = (0.1, 0.1, 0.8, 0.8)
            roi_poly: List[Tuple[float, float]] = []
            if roi_mode == "rect":
                rx, ry = st.columns(2)
                x = rx.slider("X %", 0, 100, 10) / 100.0
                y = ry.slider("Y %", 0, 100, 10) / 100.0
                rw, rh = st.columns(2)
                wperc = rw.slider("W %", 1, 100, 80) / 100.0
                hperc = rh.slider("H %", 1, 100, 80) / 100.0
                roi_rect = (x, y, wperc, hperc)
            elif roi_mode == "poly":
                st.caption(
                    "Enter points as 'x,y; x,y; ...' in percentages (0-100). Example: 10,10; 90,10; 90,90; 10,90"
                )
                poly_text = st.text_input("Polygon points", value="10,10; 90,10; 90,90; 10,90")
                roi_poly = parse_poly(poly_text)

        with center:
            st.markdown("### Live Preview")
            if media is None:
                st.info("Upload an image or video to begin.")
            else:
                suffix = os.path.splitext(media.name)[1].lower()
                bytes_data = media.read()
                detector = build_detector(
                    min_area, max_area, min_th, max_th, filter_circularity, min_circ
                )

                # overlay controls (top of center for quick access)
                oc1, oc2, oc3 = st.columns(3)
                show_points = oc1.toggle("Points", True)
                show_labels = oc2.toggle("Labels", True)
                show_boxes = oc3.toggle("Boxes", False)
                lc1, lc2, lc3 = st.columns(3)
                show_lines = lc1.toggle("Lines", True)
                line_k = lc2.slider("Links/pt", 1, 6, 3)
                max_link = lc3.slider("Max dist", 10, 600, 180, 5)

                # link mode & anchor
                lm1, lm2, lm3 = st.columns(3)
                link_mode = lm1.selectbox("Link mode", ["nearest", "anchor"], index=0)
                ax = lm2.slider("Anchor X %", 0, 100, 50) / 100.0
                ay = lm3.slider("Anchor Y %", 0, 100, 50) / 100.0

                # styles
                st.divider()
                sc1, sc2, sc3, sc4 = st.columns(4)
                color_points = _hex_to_bgr(sc1.color_picker("Point color", "#00FFC2"))
                color_lines = _hex_to_bgr(sc2.color_picker("Line color", "#FFFFFF"))
                color_boxes = _hex_to_bgr(sc3.color_picker("Box color", "#FFFFFF"))
                color_labels = _hex_to_bgr(sc4.color_picker("Label color", "#FFFFFF"))
                st1, st2, st3 = st.columns(3)
                line_thick = st1.slider("Line thickness", 1, 5, 1)
                point_scale = st2.slider("Point scale", 1, 400, 100) / 100.0
                curved_lines = st3.toggle("Curved links", True)
                curvature = st.slider("Curvature", -1.0, 1.0, 0.25, 0.01)

                # grid / ROI visual FX
                gx, gy, gz = st.columns(3)
                show_grid = gx.toggle("Grid", False)
                grid_divisions = gy.slider("Grid divisions", 2, 32, 12)
                roi_invert = gz.toggle("Invert in ROI", False)
                zx, zy = st.columns(2)
                roi_zoom = zx.toggle("Zoom in ROI", False)
                roi_zoom_factor = zy.slider("Zoom factor", 1.0, 6.0, 2.0, 0.1)

                overlays = Overlays(
                    show_points,
                    show_labels,
                    show_lines,
                    show_boxes,
                    line_k,
                    max_link,
                    line_thick,
                    point_scale,
                    link_mode,
                    (ax, ay),
                    color_points,
                    color_lines,
                    color_boxes,
                    color_labels,
                    roi_mode,
                    roi_rect,
                    roi_poly,
                    show_roi,
                    roi_invert,
                    roi_zoom,
                    roi_zoom_factor,
                    show_grid,
                    grid_divisions,
                    curved_lines,
                    curvature,
                )

                if suffix in [".png", ".jpg", ".jpeg"]:
                    file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)
                    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    rendered, kps = detect_on_frame(
                        frame, detector, blur, use_clahe, invert, overlays
                    )
                    st.image(
                        cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB),
                        caption=f"{media.name} — {len(kps)} blobs",
                        use_container_width=True,
                    )
                    _, png_buf = cv2.imencode(".png", rendered)
                    st.download_button(
                        "Download PNG",
                        data=png_buf.tobytes(),
                        file_name=f"{os.path.splitext(media.name)[0]}_overlay.png",
                        mime="image/png",
                    )
                else:
                    tdir = tempfile.mkdtemp()
                    in_path = os.path.join(tdir, media.name)
                    with open(in_path, "wb") as f:
                        f.write(bytes_data)
                    cap = cv2.VideoCapture(in_path)
                    if not cap.isOpened():
                        st.error("Could not open video. Try converting to MP4.")
                    else:
                        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        fps = cap.get(cv2.CAP_PROP_FPS) or 30
                        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        st.caption(f"{w}×{h} @ {fps:.02f} fps — {total} frames")
                        secs = st.slider("Preview seconds", 1, 12, 4)
                        preview_frames = int(min(total, max(1, secs * fps)))
                        ph = st.empty()
                        prog = st.progress(0)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        for i in range(preview_frames):
                            ret, frame = cap.read()
                            if not ret:
                                break
                            rendered, _ = detect_on_frame(
                                frame, detector, blur, use_clahe, invert, overlays
                            )
                            ph.image(
                                cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB),
                                use_container_width=True,
                            )
                            prog.progress((i + 1) / preview_frames)
                        cap.release()
                        prog.empty()

        with right:
            st.markdown("### Export MP4")
            st.caption("Process current settings to a file")
            # Export controls rely on variables created in center scope when a video is loaded
            if "suffix" not in locals() or suffix in [".png", ".jpg", ".jpeg"]:
                st.info("Upload a **video** to enable export.")
            else:
                out_fps = st.slider("Output FPS", 5, 60, int(round(fps)))
                add_timecode = _ui_toggle("Add timecode HUD", True)
                start_sec = st.number_input("Start (sec)", 0.0, value=0.0, step=0.5)
                end_sec = st.number_input("End (0=full)", 0.0, value=0.0, step=0.5)
                if st.button("Process & Export to MP4", type="primary"):
                    cap2 = cv2.VideoCapture(in_path)
                    if not cap2.isOpened():
                        st.error("Cannot open video for export.")
                    else:
                        total2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
                        fps2 = cap2.get(cv2.CAP_PROP_FPS) or fps
                        w2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        start_f = int(start_sec * fps2)
                        end_f = int(end_sec * fps2) if end_sec > 0 else total2
                        start_f = max(0, min(start_f, total2 - 1))
                        end_f = max(start_f + 1, min(end_f, total2))
                        cap2.set(cv2.CAP_PROP_POS_FRAMES, start_f)
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        out_path = os.path.join(
                            tdir, f"{os.path.splitext(media.name)[0]}_overlay.mp4"
                        )
                        writer = cv2.VideoWriter(out_path, fourcc, out_fps, (w2, h2))
                        bar = st.progress(0)
                        info = st.empty()
                        count = end_f - start_f
                        i = 0
                        while i < count:
                            ret, frame = cap2.read()
                            if not ret:
                                break
                            rendered, _ = detect_on_frame(
                                frame, detector, blur, use_clahe, invert, overlays
                            )
                            if add_timecode:
                                tc = start_f + i
                                tsec = tc / (fps2 or out_fps)
                                mm = int(tsec // 60)
                                ss = int(tsec % 60)
                                ms = int((tsec - int(tsec)) * 1000)
                                cv2.putText(
                                    rendered,
                                    f"{mm:02d}:{ss:02d}.{ms:03d}  F:{tc}/{total2}",
                                    (12, h2 - 12),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (255, 255, 255),
                                    1,
                                    cv2.LINE_AA,
                                )
                            writer.write(rendered)
                            i += 1
                            if i % 5 == 0:
                                bar.progress(min(1.0, i / count))
                                info.caption(f"Rendering {i}/{count} frames…")
                        writer.release()
                        cap2.release()
                        bar.progress(1.0)
                        info.caption("Done!")
                        with open(out_path, "rb") as f:
                            st.download_button(
                                "Download MP4",
                                data=f.read(),
                                file_name=os.path.basename(out_path),
                                mime="video/mp4",
                            )

# ---- entrypoint -----------------------------------------------------------------
if __name__ == "__main__":
    if STREAMLIT_AVAILABLE and OPENCV_AVAILABLE and NUMPY_AVAILABLE and not _running_in_web_sandbox():
        run_streamlit_app()
    else:
        print("\nBlobTracker Live — Sandbox Mode\n" + "-" * 36)
        if IMPORT_ERROR:
            print("Note: missing optional deps:", repr(IMPORT_ERROR))
        print("Running unit tests…")
        _run_tests()
        print("All tests passed. For full app: pip install", REQUIREMENTS)
