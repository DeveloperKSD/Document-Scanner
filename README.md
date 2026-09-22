
# Document Scanner (Perspective Transform + Adaptive Threshold)

A small interactive Python tool that takes a photo of a document shot at an
angle and turns it into a flat, clean, "scanned" image — click 4 corners,
and it does the rest.

---

## Tech Stack

| Component | Library / Tool | Purpose |
|---|---|---|
| Language | Python 3.8+ | Core implementation |
| Image I/O & processing | OpenCV (`opencv-python`) | Reading images, GUI windows, mouse events, perspective transform, thresholding |
| Numerical operations | NumPy | Point arrays, geometry (distances, sorting corners), matrix math |
| Contrast enhancement | OpenCV's CLAHE | Local contrast correction to handle shadows/uneven lighting |
| GUI / interaction | OpenCV's `highgui` (`imshow`, `setMouseCallback`, `waitKey`) | Displaying windows, capturing mouse clicks, keyboard shortcuts |
| File output | OpenCV (`imwrite`) / `os` | Saving results to disk |

No external ML models, frameworks, or cloud services — everything runs
locally with classical computer vision techniques.

---

## Basic Flow

```
 Load photo (skewed document)
          │
          ▼
 Display in a window
          │
          ▼
 User clicks 4 corners ──► Points auto-sorted into TL, TR, BR, BL
          │
          ▼
 Compute perspective transform matrix
 (maps the skewed quadrilateral → a straight rectangle)
          │
          ▼
 Warp the image using that matrix
 (dynamic output size based on real document proportions)
          │
          ▼
 Convert to grayscale → CLAHE contrast boost
          │
          ▼
 Apply threshold (adaptive / Otsu / plain gray / color)
          │
          ▼
 Display result ──► Save to disk (optional) ──► Undo/Reset/Cycle modes as needed
```

**Interaction controls:**
- Click — select a corner point (up to 4)
- `z` — undo last point
- `r` — reset all points
- `m` — cycle through output modes (color / gray / adaptive / Otsu)
- `s` — save current output to `scanned_output/`
- `q` / `Esc` — quit

---

## What This Project Covers (Key Learnings)

**Computer vision fundamentals**
- How a **perspective transform** maps one quadrilateral onto another using
  a 3×3 transformation matrix (`cv2.getPerspectiveTransform` +
  `cv2.warpPerspective`).
- Why 4 point correspondences are exactly enough to solve for a
  perspective (homography) transform.
- How to derive **destination rectangle size dynamically** from real-world
  point distances (`np.linalg.norm`) instead of hardcoding it — preserving
  the true aspect ratio of the document.

**Practical geometry / robustness**
- **Corner ordering** — sorting 4 unordered points into TL/TR/BR/BL using
  the sum (`x+y`) and difference (`x−y`) trick, so the user doesn't have to
  click in a specific sequence.
- Defensive coding: guarding against missing files (`cv2.imread` returning
  `None`), degenerate point selections, and re-running transforms cleanly.

**Image enhancement techniques**
- Difference between **global thresholding**, **adaptive thresholding**,
  and **Otsu's method**, and when each is appropriate (adaptive handles
  uneven lighting/shadows better than a single global cutoff).
- What **CLAHE** (Contrast Limited Adaptive Histogram Equalization) does
  and why it helps before thresholding.
- Color space basics — why grayscale conversion is a required step before
  most thresholding operations.

**Interactive application design**
- Building a **stateful GUI loop** with OpenCV's `highgui` — mouse
  callbacks, keyboard shortcuts, and redrawing state without a full
  framework like Tkinter/Qt.
- Managing **mutable global state** (`points`, current mode) safely inside
  callback-driven code.
- Designing small **UX affordances** (undo, reset, live preview lines,
  mode cycling) that make a CLI/GUI-hybrid tool actually usable instead of
  "click 4 times and hope."

**Software engineering habits**
- Separating **pure computation** (perspective math, point ordering) from
  **I/O/display logic** (drawing, saving, window management) into distinct
  functions.
- Writing code that fails **loudly and early** (file-not-found checks)
  rather than crashing deep inside a library call.

---

## Possible Next Steps

- Auto-detect document edges with `cv2.Canny` + contour detection, removing
  the need to click manually.
- Batch/multi-page mode with export to a single PDF.
- OCR integration (`pytesseract`) to extract text from the scanned output.
- Webcam capture instead of a static image file.
