"""
SMART DOCUMENT SCANNER  (DIP + AI)
==================================
Single-window dashboard built on your original scanner.

  DIP topics : edge detection (Canny/Sobel), Fourier transform + frequency
               filtering, colour models (HSV/LAB) + colour slicing,
               morphology, thresholding (adaptive/Otsu), CLAHE, median filter
  AI topics  : CSP (backtracking + forward checking)  -> auto corner detection
               A* search                              -> edge tracing
               Expert system (forward chaining)       -> "Scan Advisor"
               K-means clustering (unsupervised)      -> colour palette / ink mode

Run:   python smart_document_scanner.py [image_path]
Keys:  a auto corners | z undo | r reset | e A* edges | m mode | f shadow-fix
       n denoise | h CLAHE | t advisor auto/manual | s save | q/Esc quit
Mouse: click 4 corners (any order) or drag a corner after they are placed.
"""

import heapq
import math
import os
import sys

import cv2
import numpy as np

# ------------------------------------------------------------------ config
INPUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "01.jpg"
SAVE_DIR = "scanned_output"
WIN = "Smart Document Scanner"
MODES = ["color", "gray", "adaptive", "otsu", "ink"]

# thresholds used by the Scan Advisor (tune these for your own photos)
THRESH = dict(uneven=0.20,    # background brightness spread -> shadows
              noise=5.0,      # estimated noise sigma
              contrast=45,    # grey-level std-dev
              dim=110,        # mean brightness
              colour=0.006,   # fraction of strongly coloured pixels
              edges=0.015,    # Canny edge density
              mid=0.40)       # max fraction of mid-tones for a "text page"

# layout (canvas is 1280 x 760)
CW, CH = 1280, 760
RECT_A = (10, 54, 630, 420)      # original + overlay
RECT_B = (650, 54, 620, 420)     # scanned output
RECT_FFT = (10, 484, 300, 266)
RECT_PAL = (320, 484, 300, 266)
RECT_ADV = (630, 484, 640, 266)
A_INNER = (618, 384)             # image area inside panel A

# colours (BGR)
BG, PANEL, BORDER = (30, 27, 25), (44, 41, 38), (80, 74, 68)
TEXT, DIM = (235, 232, 228), (150, 145, 140)
TEAL, AMBER, RED = (190, 200, 60), (60, 190, 255), (90, 90, 235)


# =================================================================== utils
def shrink(img, max_side):
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    s = max_side / max(h, w)
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def fit_size(w, h, bw, bh):
    s = min(bw / w, bh / h)
    return max(1, int(w * s)), max(1, int(h * s)), s


def order_points(pts):
    """Sort 4 points into TL, TR, BR, BL (sum / difference trick)."""
    pts = np.array(pts, dtype="float32")
    out = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1)
    out[0], out[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    out[1], out[3] = pts[np.argmin(d)], pts[np.argmax(d)]
    return out


def warp_document(full, pts_disp, scale):
    """Perspective-correct the document; output size follows real proportions."""
    tl, tr, br, bl = order_points(np.array(pts_disp, np.float32) / scale)
    w = max(int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))), 50)
    h = max(int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))), 50)
    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(np.float32([tl, tr, br, bl]), dst)
    return cv2.warpPerspective(full, M, (w, h))


# ============================================= AI #1: CSP corner detection
VARS = ["TL", "TR", "BR", "BL"]
# binary constraints between corner variables (earlier, later in VARS order)
RULES_CSP = {
    ("TL", "TR"): lambda a, b: b[0] > a[0],
    ("TL", "BR"): lambda a, b: b[0] > a[0] and b[1] > a[1],
    ("TL", "BL"): lambda a, b: b[1] > a[1],
    ("TR", "BR"): lambda a, b: b[1] > a[1],
    ("TR", "BL"): lambda a, b: b[0] < a[0] and b[1] > a[1],
    ("BR", "BL"): lambda a, b: b[0] < a[0],
}


def corner_candidates(disp, limit=14):
    """DIP part: Canny edges + Otsu mask -> contours -> convex hull vertices."""
    h, w = disp.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(disp, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.dilate(cv2.Canny(gray, 50, 150), np.ones((3, 3), np.uint8))
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    pts = []
    for m in (edges, otsu):
        m = m.copy()
        m[:4, :] = m[-4:, :] = 0          # ignore the image frame itself
        m[:, :4] = m[:, -4:] = 0
        cnts, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:3]:
            if cv2.contourArea(c) < 0.10 * w * h:
                continue
            hull = cv2.convexHull(c)
            approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
            for p in approx.reshape(-1, 2).astype(np.float32):
                if all(np.hypot(*(p - q)) > 10 for q in pts):
                    pts.append(p)
    return pts[:limit]


def quad_ok(q, img_area):
    """Global constraints: size, convexity, angles, opposite sides similar."""
    if not (0.15 * img_area <= cv2.contourArea(q) <= 0.92 * img_area):
        return False
    cross = []
    for i in range(4):
        v1, v2 = q[(i + 1) % 4] - q[i], q[(i + 2) % 4] - q[(i + 1) % 4]
        cross.append(v1[0] * v2[1] - v1[1] * v2[0])
    if not (all(c > 0 for c in cross) or all(c < 0 for c in cross)):
        return False
    for i in range(4):
        v1, v2 = q[i - 1] - q[i], q[(i + 1) % 4] - q[i]
        cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        if not 55 <= math.degrees(math.acos(np.clip(cosang, -1, 1))) <= 125:
            return False
    s = [np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)]
    return min(s[0], s[2]) / max(s[0], s[2]) > 0.4 and min(s[1], s[3]) / max(s[1], s[3]) > 0.4


def solve_csp(cands, w, h):
    """Backtracking search + forward checking. Variables: TL,TR,BR,BL.
    Domains: candidate corner points. Returns (best quad or None, stats text)."""
    nodes, sols = 0, []
    min_d = 0.08 * min(w, h)

    def bt(i, assign, domains):
        nonlocal nodes
        if i == 4:
            q = np.array([assign[v] for v in VARS], np.float32)
            if quad_ok(q, w * h):
                sols.append(q)
            return
        var = VARS[i]
        for p in domains[var]:
            nodes += 1
            new, ok = dict(domains), True
            for later in VARS[i + 1:]:            # forward checking
                new[later] = [q for q in domains[later]
                              if RULES_CSP[(var, later)](p, q) and np.hypot(*(p - q)) >= min_d]
                if not new[later]:
                    ok = False
                    break
            if ok:
                assign[var] = p
                bt(i + 1, assign, new)
                del assign[var]

    bt(0, {}, {v: list(cands) for v in VARS})
    stats = f"CSP: {len(cands)} corners, {nodes} nodes, {len(sols)} valid"
    if not sols:
        return None, stats
    return max(sols, key=cv2.contourArea), stats


# ================================================= AI #2: A* edge tracing
CMIN = 0.05
NB = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]


def edge_cost_map(img):
    """Low cost on strong edges (Sobel), high cost on flat areas."""
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    mag = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    mag = np.clip(mag / (np.percentile(mag, 98) + 1e-6), 0, 1)
    return (CMIN + (1 - mag) ** 2).astype(np.float32)


def astar_trace(cost, p0, p1, corridor=16, weight=4.0):
    """Weighted A* between two points inside a corridor around the straight line.
    g = accumulated edge cost, h = weight * CMIN * euclidean distance."""
    h, w = cost.shape
    s = (int(np.clip(p0[0], 0, w - 1)), int(np.clip(p0[1], 0, h - 1)))
    t = (int(np.clip(p1[0], 0, w - 1)), int(np.clip(p1[1], 0, h - 1)))
    corr = np.zeros((h, w), np.uint8)
    cv2.line(corr, s, t, 255, 2 * corridor)

    def hf(n):
        return weight * CMIN * math.hypot(n[0] - t[0], n[1] - t[1])

    g, parent, closed = {s: 0.0}, {}, set()
    heap, expanded = [(hf(s), s)], 0
    while heap:
        _, n = heapq.heappop(heap)
        if n in closed:
            continue
        closed.add(n)
        expanded += 1
        if n == t:
            break
        for dx, dy in NB:
            nx, ny = n[0] + dx, n[1] + dy
            if not (0 <= nx < w and 0 <= ny < h) or not corr[ny, nx] or (nx, ny) in closed:
                continue
            ng = g[n] + math.hypot(dx, dy) * float(cost[ny, nx])
            if ng < g.get((nx, ny), 1e18):
                g[(nx, ny)], parent[(nx, ny)] = ng, n
                heapq.heappush(heap, (ng + hf((nx, ny)), (nx, ny)))
    path = [t]
    while path[-1] != s:
        if path[-1] not in parent:
            return [s, t], expanded            # fallback: straight line
        path.append(parent[path[-1]])
    return path[::-1], expanded


# ========================================== DIP: FFT shadow removal + misc
def fft_flatten(gray, sigma=5.0):
    """Homomorphic-style illumination removal in the frequency domain.
    log(image) -> FFT -> Gaussian low-pass = illumination -> subtract.
    Returns (flattened grey image, coloured spectrum picture)."""
    h, w = gray.shape
    small = shrink(gray, 512)
    L = np.log1p(small.astype(np.float32))
    F = np.fft.fftshift(np.fft.fft2(L))
    rows, cols = L.shape
    u, v = np.arange(cols) - cols // 2, np.arange(rows) - rows // 2
    low = np.exp(-(u[None, :] ** 2 + v[:, None] ** 2) / (2 * sigma ** 2))
    illum = np.real(np.fft.ifft2(np.fft.ifftshift(F * low)))
    illum = cv2.resize(illum, (w, h), interpolation=cv2.INTER_LINEAR)
    flat = np.clip(np.exp(np.log1p(gray.astype(np.float32)) - illum) * 240, 0, 255).astype(np.uint8)

    spec = np.log1p(np.abs(F))
    lo, hi = np.percentile(spec, 5), np.percentile(spec, 99.8)
    spec = (255 * np.clip((spec - lo) / (hi - lo + 1e-6), 0, 1)).astype(np.uint8)
    spec = cv2.applyColorMap(cv2.resize(spec, (256, 256)), cv2.COLORMAP_INFERNO)
    axes = (int(2 * sigma * 256 / cols), int(2 * sigma * 256 / rows))
    cv2.ellipse(spec, (128, 128), axes, 0, 0, 360, (255, 255, 255), 1, cv2.LINE_AA)
    return flat, spec


def flatten_bgr(bgr, flat_gray):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = flat_gray
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def noise_sigma(gray):
    """Immerkaer noise estimate (median version, robust to text edges)."""
    h, w = gray.shape
    ch, cw = min(h, 512), min(w, 512)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    crop = gray[y0:y0 + ch, x0:x0 + cw].astype(np.float32)
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    return float(np.median(np.abs(cv2.filter2D(crop, -1, k))) / (0.6745 * 6))


# ============================================ AI #3: expert system advisor
def analyse(warped):
    """Perception step: measure the scan and turn numbers into symbolic facts."""
    T = THRESH
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    g = shrink(gray, 500)
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    bg = cv2.GaussianBlur(bg, (0, 0), 8)                      # background = paper + light
    norm = np.clip(g.astype(np.float32) / np.maximum(bg, 1) * 235, 0, 255)
    hsv = cv2.cvtColor(shrink(warped, 500), cv2.COLOR_BGR2HSV)
    m = dict(
        brightness=float(g.mean()),
        contrast=float(g.std()),
        unevenness=float((np.percentile(bg, 95) - np.percentile(bg, 5)) / (bg.mean() + 1e-6)),
        noise=noise_sigma(gray),
        edges=float(cv2.Canny(g, 80, 200).mean() / 255),
        colour=float(((hsv[:, :, 1] > 90) & (hsv[:, :, 2] > 70)).mean()),
        midtone=float(((norm > 70) & (norm < 200)).mean()),
    )
    facts = {
        "uneven_light" if m["unevenness"] > T["uneven"] else "even_light",
        "noisy" if m["noise"] > T["noise"] else "clean",
        "low_contrast" if m["contrast"] < T["contrast"] else "good_contrast",
        "dim" if m["brightness"] < T["dim"] else "bright_enough",
        "has_colour" if m["colour"] > T["colour"] else "no_colour",
        "text_heavy" if (m["edges"] > T["edges"] and m["midtone"] < T["mid"]) else "graphic",
    }
    return m, facts


# knowledge base: (rule name, [conditions], conclusion)
KB = [
    ("R1", ["uneven_light"], "shadow_fix"),
    ("R2", ["noisy"], "denoise"),
    ("R3", ["low_contrast"], "clahe"),
    ("R4", ["dim"], "clahe"),
    ("R5", ["text_heavy", "has_colour"], "mode:ink"),
    ("R6", ["text_heavy", "shadow_fix"], "mode:adaptive"),
    ("R7", ["text_heavy", "even_light"], "mode:otsu"),
    ("R8", ["graphic"], "mode:color"),
]


def infer(facts):
    """Forward chaining: keep firing rules until no new fact appears.
    For 'mode:' conclusions the first rule that fires wins (conflict resolution)."""
    wm, fired, log = set(facts), set(), []
    changed = True
    while changed:
        changed = False
        for name, cond, concl in KB:
            if name in fired or not set(cond) <= wm:
                continue
            fired.add(name)
            changed = True
            text = f"{name} {'+'.join(cond)} -> {concl}"
            if concl in wm or (concl.startswith("mode:") and any(f.startswith("mode:") for f in wm)):
                log.append((text + " (skip)", False))
            else:
                wm.add(concl)
                log.append((text, True))
    mode = next((f.split(":")[1] for f in wm if f.startswith("mode:")), "color")
    plan = dict(mode=mode, shadow="shadow_fix" in wm, denoise="denoise" in wm, clahe="clahe" in wm)
    return plan, log


# =================================================== AI #4: K-means palette
def kmeans_palette(bgr):
    """Unsupervised colour analysis (K-means, cv2.kmeans).
    Stage 1: 3 clusters on all pixels -> paper (largest) and ink (darkest).
    Stage 2: 2 clusters on the strongly coloured pixels only -> accent inks
    (stamps, signatures) that would be too small to get their own cluster."""
    cv2.setRNGSeed(0)
    small = shrink(bgr, 200)
    data = small.reshape(-1, 3).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)

    def km(d, k):
        k = min(k, len(d))
        _, lab, cen = cv2.kmeans(d, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        return cen, np.bincount(lab.ravel(), minlength=k)

    cen, cnt = km(data, 3)
    order = np.argsort(-cnt)
    cen, cnt = cen[order], cnt[order]
    total = len(data)
    rest = list(range(1, len(cen)))
    dark = min(rest, key=lambda i: cen[i].mean())
    entries = [(cen[0], cnt[0] / total, "paper"), (cen[dark], cnt[dark] / total, "ink")]

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    coloured = ((hsv[:, :, 1] > 90) & (hsv[:, :, 2] > 70)).ravel()
    if coloured.sum() >= 30:
        cc, cn = km(data[coloured], 2)
        for i in np.argsort(-cn):
            entries.append((cc[i], cn[i] / total, "accent"))
    centers = np.array([e[0] for e in entries[:4]]).clip(0, 255).astype(np.uint8)
    counts = np.array([e[1] for e in entries[:4]])
    roles = [e[2] for e in entries[:4]]
    hsv_c = cv2.cvtColor(centers.reshape(1, -1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    return centers, counts, roles, hsv_c


def ink_mode(bgr, binary, palette):
    """Binarise the text but keep coloured ink (stamps, signatures) in colour.
    K-means finds the accent hues, HSV colour slicing extracts them."""
    _, _, roles, hsv_c = palette
    H, S, V = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV))
    mask = (S > 80) & (V > 60)
    hues = [int(hsv_c[i][0]) for i, r in enumerate(roles) if r == "accent"]
    if hues:
        near = np.zeros(mask.shape, bool)
        for hh in hues:
            d = np.abs(H.astype(int) - hh)
            near |= np.minimum(d, 180 - d) < 15
        mask &= near
    mask = mask.astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))   # remove specks
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    out = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    out[mask > 0] = bgr[mask > 0]
    return out


def make_output(st, plan):
    bgr = st.flat_bgr if plan["shadow"] else st.warped
    gray = st.flat_gray if plan["shadow"] else st.gray
    if plan["denoise"]:
        gray = cv2.medianBlur(gray, 3)
    if plan["clahe"]:
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    mode = plan["mode"]
    if mode == "color":
        return bgr
    if mode == "gray":
        return gray
    if mode == "otsu":
        return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    block = int(max(25, gray.shape[1] // 40)) | 1
    smooth = cv2.GaussianBlur(gray, (3, 3), 0)          # tames speckle before thresholding
    binary = cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, block, 15)
    return binary if mode == "adaptive" else ink_mode(bgr, binary, st.palette)


# ======================================================================= UI
def put(c, text, x, y, scale=0.45, color=TEXT, thick=1):
    cv2.putText(c, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def panel(c, rect, title, accent=TEAL):
    x, y, w, h = rect
    cv2.rectangle(c, (x, y), (x + w, y + h), PANEL, -1)
    cv2.rectangle(c, (x, y), (x + w, y + h), BORDER, 1)
    cv2.rectangle(c, (x, y), (x + 4, y + h), accent, -1)
    put(c, title, x + 14, y + 20, 0.48, TEXT)
    return x + 6, y + 30, w - 12, h - 36


def blit_fit(c, img, box):
    bx, by, bw, bh = box
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    nw, nh, s = fit_size(img.shape[1], img.shape[0], bw, bh)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    x0, y0 = bx + (bw - nw) // 2, by + (bh - nh) // 2
    c[y0:y0 + nh, x0:x0 + nw] = img


def placeholder(c, box, text):
    bx, by, bw, bh = box
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    put(c, text, bx + (bw - tw) // 2, by + bh // 2, 0.5, DIM)


def glow_path(img, path, color):
    pts = np.array(path, np.int32).reshape(-1, 1, 2)
    ov = img.copy()
    cv2.polylines(ov, [pts], False, color, 8, cv2.LINE_AA)
    img[:] = cv2.addWeighted(ov, 0.35, img, 0.65, 0)
    cv2.polylines(img, [pts], False, (255, 255, 255), 2, cv2.LINE_AA)


def draw_overlay(st):
    img = st.disp.copy()
    pts = st.points
    if len(pts) == 4:
        poly = np.array(pts, np.int32)
        ov = img.copy()
        cv2.fillPoly(ov, [poly], TEAL)
        img = cv2.addWeighted(ov, 0.18, img, 0.82, 0)
        cv2.polylines(img, [poly], True, TEAL, 2, cv2.LINE_AA)
    elif len(pts) > 1:
        cv2.polylines(img, [np.array(pts, np.int32)], False, TEAL, 2, cv2.LINE_AA)
    if st.show_astar and st.traces:
        for path in st.traces:
            glow_path(img, path, AMBER)
    labels = VARS if len(pts) == 4 else [str(i + 1) for i in range(len(pts))]
    for p, lab in zip(pts, labels):
        cv2.circle(img, tuple(p), 9, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, tuple(p), 6, AMBER, -1, cv2.LINE_AA)
        put(img, lab, p[0] + 11, p[1] - 9, 0.5, (255, 255, 255), 1)
    return img


def draw_header(c, st):
    cv2.rectangle(c, (0, 0), (CW, 46), (38, 35, 32), -1)
    cv2.rectangle(c, (0, 44), (CW, 46), TEAL, -1)
    put(c, "SMART DOCUMENT SCANNER", 14, 22, 0.7, TEAL, 2)
    put(c, "[a] auto corners  [z] undo  [r] reset  [e] A* edges  [m] mode  [f] shadow-fix  "
           "[n] denoise  [h] CLAHE  [t] advisor  [s] save  [q] quit", 14, 38, 0.38, DIM)
    plan = st.plan()
    put(c, f"ADVISOR: {'AUTO' if st.auto else 'MANUAL'}   MODE: {plan['mode'].upper()}", 900, 20, 0.5, TEXT)
    put(c, st.msg, 900, 38, 0.4, AMBER)


def draw_advisor(c, st):
    bx, by, bw, bh = panel(c, RECT_ADV, "SCAN ADVISOR  (expert system, forward chaining)", AMBER)
    if st.warped is None:
        return placeholder(c, (bx, by, bw, bh), "waiting for 4 corners ...")
    m, lh = st.measures, 17
    x1, x2, y = bx + 10, bx + 285, by + 14
    put(c, "MEASUREMENTS", x1, y, 0.42, DIM)
    rows = [("Brightness", f"{m['brightness']:.0f}"), ("Contrast", f"{m['contrast']:.0f}"),
            ("Light unevenness", f"{m['unevenness']:.2f}"), ("Noise sigma", f"{m['noise']:.1f}"),
            ("Edge density", f"{m['edges']:.3f}"), ("Colour pixels", f"{m['colour'] * 100:.1f}%"),
            ("Mid-tone ratio", f"{m['midtone']:.2f}")]
    for i, (k, v) in enumerate(rows):
        put(c, k, x1, y + (i + 1) * lh, 0.42)
        put(c, v, x1 + 175, y + (i + 1) * lh, 0.42, TEAL)
    y2 = y + (len(rows) + 2) * lh + 4
    put(c, "SEARCH STATS", x1, y2, 0.42, DIM)
    put(c, st.csp_stats, x1, y2 + lh, 0.4)
    put(c, st.astar_stats, x1, y2 + 2 * lh, 0.4)

    put(c, "RULES FIRED", x2, y, 0.42, DIM)
    for i, (text, ok) in enumerate(st.rule_log[:8]):
        put(c, text, x2, y + (i + 1) * lh, 0.4, TEAL if ok else DIM)
    y3 = y + 10 * lh + 4
    plan = st.plan()
    put(c, "PLAN (auto)" if st.auto else "PLAN (manual override)", x2, y3, 0.42, DIM)
    put(c, f"mode = {plan['mode']}", x2, y3 + lh, 0.45, AMBER)
    flags = "  ".join(f"{k}={'on' if plan[k] else 'off'}" for k in ("shadow", "denoise", "clahe"))
    put(c, flags, x2, y3 + 2 * lh, 0.4)


def draw_palette(c, st):
    bx, by, bw, bh = panel(c, RECT_PAL, "K-MEANS PALETTE", TEAL)
    if st.palette is None:
        return placeholder(c, (bx, by, bw, bh), "waiting for 4 corners ...")
    centers, counts, roles, _ = st.palette
    role_col = {"paper": TEXT, "ink": DIM, "accent": AMBER}
    for i in range(len(centers)):
        y = by + 8 + i * 48
        col = tuple(int(v) for v in centers[i])
        cv2.rectangle(c, (bx + 10, y), (bx + 62, y + 38), col, -1)
        cv2.rectangle(c, (bx + 10, y), (bx + 62, y + 38), BORDER, 1)
        put(c, roles[i].upper(), bx + 76, y + 14, 0.5, role_col[roles[i]])
        cv2.rectangle(c, (bx + 76, y + 24), (bx + 266, y + 32), (60, 56, 52), -1)
        cv2.rectangle(c, (bx + 76, y + 24), (bx + 76 + max(2, int(190 * counts[i])), y + 32), col, -1)
        put(c, f"{counts[i] * 100:.1f}%", bx + 200, y + 14, 0.42, TEXT)
    put(c, "accent hues drive colour slicing in INK mode", bx + 10, by + 215, 0.38, DIM)


def draw_fft(c, st):
    box = panel(c, RECT_FFT, "FFT SPECTRUM  (log magnitude)", TEAL)
    if st.spectrum is None:
        return placeholder(c, box, "waiting for 4 corners ...")
    bx, by, bw, bh = box
    blit_fit(c, st.spectrum, (bx, by, bw, bh - 16))
    put(c, "circle = low-pass cutoff (illumination)", bx + 10, by + bh - 4, 0.38, DIM)


def render(st):
    c = np.full((CH, CW, 3), BG, np.uint8)
    draw_header(c, st)
    # panel A - original
    title = "ORIGINAL  -  A* edge trace ON" if st.show_astar else "ORIGINAL  -  corners"
    panel(c, RECT_A, title, TEAL)
    ox, oy = st.origin
    c[oy:oy + st.dh, ox:ox + st.dw] = draw_overlay(st)
    # panel B - result
    plan = st.plan()
    onoff = lambda b: "ON" if b else "off"
    title = (f"SCANNED OUTPUT  -  {plan['mode']}   shadow-fix {onoff(plan['shadow'])}   "
             f"denoise {onoff(plan['denoise'])}   CLAHE {onoff(plan['clahe'])}")
    box = panel(c, RECT_B, title, AMBER)
    if st.result is None:
        placeholder(c, box, "click 4 corners or press [a]")
    else:
        blit_fit(c, st.result, box)
    draw_fft(c, st)
    draw_palette(c, st)
    draw_advisor(c, st)
    return c


# ================================================================== state
class State:
    def __init__(self, full):
        self.full = full
        fh, fw = full.shape[:2]
        self.dw, self.dh, self.scale = fit_size(fw, fh, *A_INNER)
        self.disp = cv2.resize(full, (self.dw, self.dh), interpolation=cv2.INTER_AREA)
        self.origin = (RECT_A[0] + 6 + (A_INNER[0] - self.dw) // 2,
                       RECT_A[1] + 30 + (A_INNER[1] - self.dh) // 2)
        self.points, self.drag = [], None
        self.auto = True
        self.manual = dict(mode="adaptive", shadow=False, denoise=False, clahe=False)
        self.advice = dict(self.manual)
        self.rule_log, self.measures = [], {}
        self.show_astar, self.traces, self.cost = False, None, None
        self.csp_stats, self.astar_stats = "CSP: not run", "A*: off (press e)"
        self.warped = self.result = self.palette = self.spectrum = None
        self.canvas, self.dirty, self.msg = None, True, "ready"

    def plan(self):
        return self.advice if self.auto else self.manual

    def go_manual(self):
        if self.auto:
            self.manual, self.auto = dict(self.advice), False

    def auto_detect(self):
        cands = corner_candidates(self.disp)
        quad, self.csp_stats = solve_csp(cands, self.dw, self.dh)
        if quad is None:
            self.points, self.msg = [], "CSP found no valid quad - click corners"
        else:
            self.points, self.msg = np.round(quad).astype(int).tolist(), "corners found by CSP"
        self.recompute()

    def recompute(self):
        self.traces, self.dirty = None, True
        if len(self.points) != 4:
            self.warped = self.result = self.palette = self.spectrum = None
            return
        self.points = order_points(self.points).round().astype(int).tolist()
        self.warped = warp_document(self.full, self.points, self.scale)
        self.gray = cv2.cvtColor(self.warped, cv2.COLOR_BGR2GRAY)
        self.flat_gray, self.spectrum = fft_flatten(self.gray)
        self.flat_bgr = flatten_bgr(self.warped, self.flat_gray)
        self.palette = kmeans_palette(self.flat_bgr)
        self.measures, facts = analyse(self.warped)
        self.advice, self.rule_log = infer(facts)
        if self.show_astar:
            self.update_traces()
        self.refresh()

    def refresh(self):
        if self.warped is not None:
            self.result = make_output(self, self.plan())
        self.dirty = True

    def update_traces(self):
        if len(self.points) != 4:
            return
        if self.cost is None:
            self.cost = edge_cost_map(self.disp)
        self.traces, total = [], 0
        for i in range(4):
            path, n = astar_trace(self.cost, self.points[i], self.points[(i + 1) % 4])
            self.traces.append(path)
            total += n
        self.astar_stats = f"A*: {total} nodes expanded (4 edges)"

    def save(self):
        if self.result is None:
            self.msg = "nothing to save yet"
            return
        os.makedirs(SAVE_DIR, exist_ok=True)
        n = len([f for f in os.listdir(SAVE_DIR) if f.startswith("scan_")]) + 1
        cv2.imwrite(os.path.join(SAVE_DIR, f"scan_{n}.png"), self.result)
        cv2.imwrite(os.path.join(SAVE_DIR, f"dashboard_{n}.png"), self.canvas)
        self.msg = f"saved scan_{n}.png + dashboard_{n}.png"
        self.dirty = True


def on_mouse(event, x, y, flags, st):
    px, py = x - st.origin[0], y - st.origin[1]
    inside = 0 <= px < st.dw and 0 <= py < st.dh
    if event == cv2.EVENT_LBUTTONDOWN and inside:
        if len(st.points) == 4:                       # grab a corner to drag it
            d = [math.hypot(px - p[0], py - p[1]) for p in st.points]
            if min(d) < 16:
                st.drag = int(np.argmin(d))
        else:
            st.points.append([px, py])
            st.dirty = True
            if len(st.points) == 4:
                st.recompute()
    elif event == cv2.EVENT_MOUSEMOVE and st.drag is not None:
        st.points[st.drag] = [int(np.clip(px, 0, st.dw - 1)), int(np.clip(py, 0, st.dh - 1))]
        st.dirty = True
    elif event == cv2.EVENT_LBUTTONUP and st.drag is not None:
        st.drag = None
        st.recompute()


def load_image():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Could not find input image: {INPUT_PATH}")
    img = cv2.imread(INPUT_PATH)
    if img is None:
        raise FileNotFoundError(f"cv2 failed to read image: {INPUT_PATH}")
    return shrink(img, 2400)


def main():
    st = State(load_image())
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, CW, CH)
    cv2.setMouseCallback(WIN, on_mouse, st)
    st.auto_detect()

    while True:
        if st.dirty:
            st.canvas = render(st)
            st.dirty = False
        cv2.imshow(WIN, st.canvas)
        key = cv2.waitKey(20) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == ord("a"):
            st.auto_detect()
        elif key == ord("z") and st.points:
            st.points.pop()
            st.recompute()
        elif key == ord("r"):
            st.points, st.msg = [], "reset"
            st.recompute()
        elif key == ord("e"):
            st.show_astar = not st.show_astar
            if st.show_astar:
                st.update_traces()
            else:
                st.astar_stats = "A*: off (press e)"
            st.dirty = True
        elif key == ord("m"):
            st.go_manual()
            i = MODES.index(st.manual["mode"])
            st.manual["mode"] = MODES[(i + 1) % len(MODES)]
            st.refresh()
        elif key in (ord("f"), ord("n"), ord("h")):
            st.go_manual()
            flag = {ord("f"): "shadow", ord("n"): "denoise", ord("h"): "clahe"}[key]
            st.manual[flag] = not st.manual[flag]
            st.refresh()
        elif key == ord("t"):
            st.auto = not st.auto
            st.manual = dict(st.advice) if not st.auto else st.manual
            st.refresh()
        elif key == ord("s"):
            st.save()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
