# Smart Document Scanner (DIP + AI)

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

## How to Run

### 1. Prerequisites

- Python 3.8 or newer
- An input document image (`.jpg`, `.jpeg` or `.png`). A webcam is **not** required.
- A normal desktop Python environment. The tool uses OpenCV's GUI windows, so it will not open in headless environments such as Google Colab.

### 2. Clone / Download the Project

```bash
git clone <repository-url>
cd <repository-folder>
```

Or download the project ZIP and extract it.

### 3. Install Dependencies

```bash
pip install opencv-python numpy
```

To use a virtual environment, create and activate it first:

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Then run the `pip install` command above.

### 4. Add an Input Image

Place your document image in the project folder. The default filename is `01.jpg`:

```text
project/
├── smart_document_scanner.py
├── 01.jpg
└── README.md
```

You can also use any filename by passing it as a command-line argument.

### 5. Run the Scanner

Using the default `01.jpg`:

```bash
python smart_document_scanner.py
```

Or with your own image:

```bash
python smart_document_scanner.py path/to/document.jpg
```

### 6. What Happens After Running

A single-window dashboard opens and the program tries to detect the document
corners automatically (see **Basic Flow** above). If detection fails, click the
four corners yourself.

### 7. Using the Scanner

| Key | Action |
|---|---|
| `A` | Automatically detect corners |
| `Z` | Undo last corner |
| `R` | Reset corners |
| `E` | Toggle A* edge tracing |
| `M` | Cycle processing modes (color, gray, adaptive, otsu, ink) |
| `F` | Toggle FFT shadow correction |
| `N` | Toggle denoising |
| `H` | Toggle CLAHE enhancement |
| `T` | Toggle automatic / manual Scan Advisor |
| `S` | Save the scanned document |
| `Q` / `Esc` | Quit |

### 8. Manual Corner Selection

If automatic detection does not find a valid document:

1. Click the four corners of the document, in any order.
2. Once four points are selected, perspective correction runs automatically.
3. Drag any corner to fine-tune its position.

### 9. Saving the Result

Press `S`. The program creates a `scanned_output` folder:

```text
scanned_output/
├── scan_1.png        # final processed document
└── dashboard_1.png   # full dashboard screenshot
```

Each further save creates the next numbered pair.

### 10. Troubleshooting

**`ModuleNotFoundError: No module named 'cv2'`** or **`'numpy'`**

```bash
pip install opencv-python numpy
```

**`Could not find input image: 01.jpg`**

Place an image named `01.jpg` in the project folder, or pass the path:

```bash
python smart_document_scanner.py my_document.jpg
```

**The OpenCV window does not appear**

Run the script in a normal desktop Python environment, not a headless one such as Google Colab.

### Quick Start

```bash
pip install opencv-python numpy
python smart_document_scanner.py
```

Put `01.jpg` beside the script first, and press `S` to save the result.

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
