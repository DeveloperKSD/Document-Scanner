# DOCUMENT SCANNER


An interactive tool that turns a skewed photo of a document into a clean, flat
scan. It finds the page corners on its own, decides how to clean the image with
a small expert system, and shows every step on one dashboard.

Built by extending a basic "click 4 corners + threshold" scanner with features
from our **Digital Image Processing** and **Artificial Intelligence** syllabi.

---

## Tech Stack

| Component | Tool | Purpose |
|---|---|---|
| Language | Python 3.8+ | Core implementation |
| Image processing and GUI | OpenCV (`opencv-python`) | Edges, contours, warping, thresholding, K-means, dashboard window, mouse and keyboard events |
| Numerical work | NumPy (incl. `np.fft`) | Geometry, Fourier transform, array maths |
| Search | `heapq`, `math` (standard library) | Priority queue for A* |

No ML frameworks, models or cloud services. Everything is classical CV and
classical AI, and runs locally.

---

## Basic Flow

```
 Load photo
     |
     v
 Canny edges + Otsu mask -> contours -> candidate corners
     |
     v
 CSP solver (backtracking + forward checking) -> best valid quadrilateral
     |          (or click 4 corners / drag a corner manually)
     v
 Perspective warp (output size from real edge lengths)
     |
     v
 FFT shadow removal -> K-means palette -> measure image (light, noise, colour, edges)
     |
     v
 Scan Advisor (forward-chaining rules) picks: mode + shadow fix + denoise + CLAHE
     |
     v
 Output: color / gray / adaptive / Otsu / ink   ->   optional A* edge trace   ->   save
```

---

## Features and Why We Used Them

| Feature | Topic covered | Why it is here |
|---|---|---|
| Auto corner detection | DIP: edge detection, contours. AI: CSP | Removes manual clicking. Corners are variables, candidate points are domains, and angle, convexity and size rules are constraints. |
| A* edge tracing | DIP: gradient operators. AI: informed search | Traces the page border along the strongest Sobel gradient. Cost is low on edges and high on flat areas, so A* follows the edge. |
| Scan Advisor | AI: knowledge representation, forward chaining, expert system | Measures the image, turns numbers into facts, and fires rules to choose the pipeline. Shows which rules fired, so the decision is explainable. |
| FFT shadow removal | DIP: Fourier transform, frequency-domain filtering | Illumination is low frequency and text is high frequency. A low-pass filter on log(image) estimates the lighting so it can be removed. |
| K-means ink mode | DIP: colour models, colour slicing, morphology. AI: unsupervised learning | Clusters colours to find paper, ink and accent hues, then keeps stamps and signatures in colour while the text is binarized. |
| Thresholding, CLAHE, median filter | DIP: enhancement | Adaptive/Otsu binarization, local contrast boost, and noise removal, applied when the advisor asks for them. |

---

## Controls

| Input | Action |
|---|---|
| Click x4 | Place corners (any order); drag a corner to adjust |
| `a` | Auto-detect corners (CSP) |
| `z` / `r` | Undo point / reset |
| `e` | Toggle A* edge trace |
| `m` | Cycle mode: color, gray, adaptive, otsu, ink |
| `f` / `n` / `h` | Toggle shadow fix / denoise / CLAHE |
| `t` | Advisor auto or manual |
| `s` | Save scan and dashboard to `scanned_output/` |
| `q` / `Esc` | Quit |

Run: `python smart_document_scanner.py your_image.jpg`

---

## What We Learnt

**Digital Image Processing**
- A perspective transform maps a quadrilateral to a rectangle, and 4 point pairs are enough to solve it.
- Canny and Sobel edges plus contour approximation can locate a document without any learning.
- Working in the log domain turns lighting (multiplicative) into something a frequency-domain filter can separate from content.
- Otsu suits evenly lit pages, while adaptive thresholding copes with shadows.
- Colour slicing in HSV lets us keep coloured ink while binarizing the rest.

**Artificial Intelligence**
- A CSP formulation with forward checking prunes the search heavily. It explored only a few dozen nodes on our test image.
- A* only works well with a sensible cost function. Here, the edge map is the cost.
- Forward chaining lets rules build on earlier conclusions, and conflict resolution is needed when two rules suggest different modes.
- K-means struggles with small colour regions, so we ran a second pass on coloured pixels only to catch stamps and signatures.

**Engineering**
- Separating computation (CSP, A*, FFT, inference) from drawing made each part easy to test on its own.
- Showing the internals (spectrum, palette, fired rules, search statistics) makes the system easier to debug and to explain.

---

## Limitations and Next Steps

- Advisor thresholds (`THRESH` in the code) are hand-tuned and may need adjusting for different cameras and lighting.
- Auto-detection assumes the page is clearly distinct from its background.
- Ideas: OCR with `pytesseract`, webcam mode with block-matching corner tracking, a decision-tree document classifier, multi-page export to PDF.
