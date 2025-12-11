* *cv2.cvtColor(images\[i], cv2.COLOR\_RGB2GRAY) -* Grayscale images have only one channel, representing brightness. Instead of each pixel having red, green, and blue values, each pixel has a single intensity value, typically ranging from 0 (black) to 255 (white).

* *cv2.equalizeHist(gray) -* takes a grayscale image (gray) and redistributes its pixel intensity values to enhance the contrast across the image.

* *cv2.cvtColor(images\[i], cv2.COLOR\_RGB2LAB) -* **LAB** is a perceptually uniform color space designed to approximate human vision more closely than RGB.

  It has three channels:

  * **L** – Lightness (brightness), from 0 (black) to 100 (white)
  * **A** – Green to Red component
  * **B** – Blue to Yellow component

* cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  * **Divides the image into small tiles** (defined by `tileGridSize`)
  * **Applies histogram equalization** in each tile separately
  * Then **blends** the results to avoid artificial borders
  * **Limits contrast amplification** (via `clipLimit`) to prevent noise from getting boosted

* So by converting to LAB, we can:

  1. Isolate the **lightness** component (`L`)
  2. Apply **CLAHE only to L** (to enhance contrast)
  3. Keep color information (A & B) untouched
  4. Recombine L, A, B and convert back to RGB

  This preserves the **original colors** while **boosting contrast**, without introducing weird color artifacts.

<br />

* cv2.GaussianBlur(noisy, (15, 15), 0) - Gaussian blur uses a **Gaussian function** (bell-shaped curve) to **weight surrounding pixels** when computing the value of each output pixel. This creates a natural, smooth blur — much like what you see when you slightly defocus a camera.
* cv2.medianBlur(noisy, 15) - **Median blur** is a noise-reduction filter that replaces each pixel’s value with the **median** value of the neighboring pixels, defined by the kernel size.

  Unlike Gaussian blur (which uses a weighted average), median blur **preserves edges better** and is especially good at removing **salt-and-pepper noise** (random white and black dots). ( removes **outlier pixel values**, which is why it's effective for **salt-and-pepper noise**.
* cv2.threshold(gray\_image, 127, 255, cv2.THRESH\_BINARY) - **Thresholding** is a method of **image segmentation**. It converts a grayscale image into a **binary image** (black and white) by assigning:

  * Pixels that were bright (≥127) become **white (255)**
  * Pixels that were dark (<127) become **black (0)**

  <br />

- cv2.threshold(gray\_image, 0, 255, cv2.THRESH\_BINARY + cv2.THRESH\_OTSU) -&#x20;

  1. **Analyzes the image histogram** to find the best threshold that separates the pixels into two distinct groups (foreground vs. background).
  2. Applies **binary thresholding** using that optimal value.
  3. Returns:
     * The **threshold value found by Otsu**
     * The resulting **binary image**
  4. You **must** combine it with a basic thresholding type like `cv2.THRESH_BINARY (` What to do with pixels`)`, because `cv2.THRESH_OTSU` **only tells OpenCV how to calculate the threshold**, not how to apply it.
  5. Other valid combos:&#x20;

     * `cv2.THRESH_TOZERO + cv2.THRESH_OTSU`
     * `cv2.THRESH_TRUNC + cv2.THRESH_OTSU`

  <br />
- cv2.Canny(gray\_image, 50, 150) - Canny edge detection involves several steps:

  1. **Noise reduction**
     * Applies a Gaussian blur (internally) to reduce noise.
  2. **Gradient calculation**
     * Finds the intensity gradient (change in brightness) in the image using Sobel operators.
  3. **Non-maximum suppression**
     * Thins out edges to retain only the **strongest edge pixels** in a given direction.
  4. **Hysteresis thresholding**
     * Uses `threshold1` and `threshold2` to decide which edges to keep:
       * Gradients > 150 → **strong edge** → kept
       * Gradients < 50 → **not an edge** → discarded
       * Between 50–150 → kept **only if connected to a strong edge**

  <br />

  | Parameter | Value           | Meaning                                             |
  | :-------- | :-------------- | :-------------------------------------------------- |
  | `50`      | Lower threshold | For **hysteresis**: potential edge if gradient > 50 |
  | `150`     | Upper threshold | Strong edge if gradient > 150                       |

  \
  Copied from: ChatGPT - <[https://chatgpt.com/](https://chatgpt.com/ "ChatGPT")>

