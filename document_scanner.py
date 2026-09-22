# ============================================================
# DOCUMENT SCANNER USING MOUSE POINTS + PERSPECTIVE TRANSFORM
# (Enhanced version)
# ============================================================
# NEW FEATURES ADDED:
#   - File-not-found check
#   - Auto point-reordering (TL/TR/BR/BL, regardless of click order)
#   - Dynamic output size (based on actual document dimensions)
#   - Live preview lines connecting clicked points
#   - Undo last point (press 'z')
#   - Reset all points (press 'r')
#   - Save output image (press 's')
#   - Multiple threshold modes: color / gray / adaptive / otsu (press 'm' to cycle)
#   - CLAHE contrast correction before thresholding
# ============================================================

import cv2
import numpy as np
import os

# ---------------- Config ----------------
INPUT_PATH = "01.jpg"
DISPLAY_SIZE = (1280, 720)   # size of the image shown for point-picking
SAVE_DIR = "scanned_output"

THRESH_MODES = ["color", "gray", "adaptive", "otsu"]
mode_index = 2  # default: "adaptive"

# ---------------- Load Image ----------------
if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(f"Could not find input image: {INPUT_PATH}")

img = cv2.imread(INPUT_PATH)
if img is None:
    raise FileNotFoundError(f"cv2 failed to read image (corrupt or unsupported format): {INPUT_PATH}")

img = cv2.resize(img, DISPLAY_SIZE)
orig = img.copy()

points = []          # clicked points (in display-image coordinates)
warped_result = None  # holds the latest warped/thresholded result for saving


# ---------------- Helper: order points TL, TR, BR, BL ----------------
def order_points(pts):
    """
    Takes 4 unordered (x, y) points and returns them ordered as:
    top-left, top-right, bottom-right, bottom-left.
    Works regardless of the order the user clicked them in.
    """
    pts = np.array(pts, dtype="float32")
    ordered = np.zeros((4, 2), dtype="float32")

    # Top-left has smallest sum (x+y); bottom-right has largest sum
    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]   # TL
    ordered[2] = pts[np.argmax(s)]   # BR

    # Top-right has smallest difference (x-y); bottom-left has largest difference
    diff = np.diff(pts, axis=1)
    ordered[1] = pts[np.argmin(diff)]  # TR
    ordered[3] = pts[np.argmax(diff)]  # BL

    return ordered


# ---------------- Helper: redraw image with current points ----------------
def redraw():
    """Redraws the base image with all clicked points and connecting lines."""
    global img
    img = orig.copy()

    for i, p in enumerate(points):
        cv2.circle(img, tuple(p), 8, (0, 0, 255), -1)
        cv2.putText(img, str(i + 1), (p[0] + 10, p[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Draw connecting lines between points (live preview of the quad)
    if len(points) > 1:
        for i in range(len(points) - 1):
            cv2.line(img, tuple(points[i]), tuple(points[i + 1]), (255, 0, 0), 2)
        if len(points) == 4:
            cv2.line(img, tuple(points[3]), tuple(points[0]), (255, 0, 0), 2)

    cv2.imshow("Select Points", img)


# ---------------- Perspective + Threshold ----------------
def perspective_and_threshold():
    global warped_result

    pts1 = order_points(points)

    # ---- Dynamic output size based on actual document dimensions ----
    (tl, tr, br, bl) = pts1

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = int(max(width_top, width_bottom))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_height = int(max(height_left, height_right))

    # Safety fallback in case points are degenerate (e.g. clicked on same spot)
    max_width = max(max_width, 50)
    max_height = max(max_height, 50)

    pts2 = np.float32([
        [0, 0],
        [max_width, 0],
        [max_width, max_height],
        [0, max_height]
    ])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(orig, matrix, (max_width, max_height))

    apply_current_mode(warped)


def apply_current_mode(warped):
    """Applies the currently selected threshold/display mode and shows it."""
    global warped_result
    mode = THRESH_MODES[mode_index]

    if mode == "color":
        result = warped

    else:
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        # CLAHE improves contrast in shadowed / unevenly lit documents
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        if mode == "gray":
            result = gray

        elif mode == "adaptive":
            result = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                25, 15
            )

        elif mode == "otsu":
            _, result = cv2.threshold(
                gray, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

    warped_result = result
    cv2.imshow("Perspective Transform", warped)
    cv2.imshow(f"Output - mode: {mode}", result)


def save_output():
    if warped_result is None:
        print("Nothing to save yet — select 4 points first.")
        return

    os.makedirs(SAVE_DIR, exist_ok=True)
    existing = len(os.listdir(SAVE_DIR))
    filename = os.path.join(SAVE_DIR, f"scan_{existing + 1}.png")
    cv2.imwrite(filename, warped_result)
    print(f"Saved: {filename}")


def close_result_windows():
    """Closes the transform/output windows so mode switches don't stack duplicates."""
    for name in list(cv2.getWindowProperty.__self__.__dict__.keys()) if False else []:
        pass  # no-op placeholder, kept simple below
    try:
        cv2.destroyWindow("Perspective Transform")
    except cv2.error:
        pass
    for m in THRESH_MODES:
        try:
            cv2.destroyWindow(f"Output - mode: {m}")
        except cv2.error:
            pass


# ---------------- Mouse Callback ----------------
def mouse_click(event, x, y, flags, param):
    global points

    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append([x, y])
        redraw()

        if len(points) == 4:
            perspective_and_threshold()


# ---------------- Main ----------------
cv2.namedWindow("Select Points")
cv2.imshow("Select Points", img)
cv2.setMouseCallback("Select Points", mouse_click)

print("Click 4 corner points in any order (they'll be auto-sorted).")
print("Keys: [z] undo last point   [r] reset   [s] save output   [m] cycle mode   [q]/[ESC] quit")

while True:
    key = cv2.waitKey(20) & 0xFF

    if key == ord('z'):  # undo last point
        if points:
            points.pop()
            redraw()
            close_result_windows()

    elif key == ord('r'):  # reset all points
        points = []
        redraw()
        close_result_windows()

    elif key == ord('s'):  # save current output
        save_output()

    elif key == ord('m'):  # cycle threshold mode
        mode_index = (mode_index + 1) % len(THRESH_MODES)
        if len(points) == 4:
            close_result_windows()
            perspective_and_threshold()
        else:
            print(f"Mode set to '{THRESH_MODES[mode_index]}' (will apply once 4 points are selected).")

    elif key in (ord('q'), 27):  # 'q' or ESC to quit
        break

cv2.destroyAllWindows()
