==================================================================================================================
Question 4)

# Canny Edge Detection Implementation

This project implements a Canny Edge Detection algorithm using Python and OpenCV. The algorithm includes the following key steps:
1. **Gaussian Blur**: Smoothens the image to reduce noise.
2. **Gradient Calculation**: Calculates the magnitude and direction of gradients using the Sobel operator.
3. **Non-Maximum Suppression (NMS)**: Thins edges by suppressing non-maximum pixels.
4. **Double Thresholding**: Categorizes pixels as strong, weak, or irrelevant edges.
5. **Hysteresis**: Connects weak edges to strong edges if they are part of the same edge.

## Requirements
- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Matplotlib

## Code

### Implementation

```python
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

class CannyEdgeDetector:
    def __init__(self, img):
        self.img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        self.gaussian_blur = None
        self.gradient_magnitude = None
        self.suppressed = None
        self.thresholded = None

    def apply_gaussian_blur(self, kernel_size=5):
        self.gaussian_blur = cv.GaussianBlur(self.img, (kernel_size, kernel_size), 1.4)
        return self.gaussian_blur

    def compute_gradient(self):
        gx = cv.Sobel(self.gaussian_blur, cv.CV_64F, 1, 0, ksize=3)
        gy = cv.Sobel(self.gaussian_blur, cv.CV_64F, 0, 1, ksize=3)
        self.magnitude = np.sqrt(gx**2 + gy**2)
        self.direction = np.arctan2(gy, gx)
        return self.magnitude, self.direction

    def non_maximum_suppression(self):
        rows, cols = self.magnitude.shape
        self.suppressed = np.zeros((rows, cols), dtype=np.int32)
        angle = self.direction * 180 / np.pi
        angle[angle < 0] += 180

        for i in range(1, rows-1):
            for j in range(1, cols-1):
                q, r = 255, 255
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = self.magnitude[i, j+1]
                    r = self.magnitude[i, j-1]
                elif 22.5 <= angle[i, j] < 67.5:
                    q = self.magnitude[i+1, j-1]
                    r = self.magnitude[i-1, j+1]
                elif 67.5 <= angle[i, j] < 112.5:
                    q = self.magnitude[i+1, j]
                    r = self.magnitude[i-1, j]
                elif 112.5 <= angle[i, j] < 157.5:
                    q = self.magnitude[i-1, j-1]
                    r = self.magnitude[i+1, j+1]

                if (self.magnitude[i, j] >= q) and (self.magnitude[i, j] >= r):
                    self.suppressed[i, j] = self.magnitude[i, j]
                else:
                    self.suppressed[i, j] = 0

        return self.suppressed

    def threshold(self, lowThresholdRatio=0.05, highThresholdRatio=0.09):
        highThreshold = self.suppressed.max() * highThresholdRatio
        lowThreshold = highThreshold * lowThresholdRatio

        weak, strong = 25, 255
        strong_i, strong_j = np.where(self.suppressed >= highThreshold)
        weak_i, weak_j = np.where((self.suppressed <= highThreshold) & (self.suppressed >= lowThreshold))

        self.thresholded = np.zeros_like(self.suppressed, dtype=np.int32)
        self.thresholded[strong_i, strong_j] = strong
        self.thresholded[weak_i, weak_j] = weak

        return self.thresholded

    def hysteresis(self, weak=25, strong=255):
        rows, cols = self.thresholded.shape
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                if self.thresholded[i, j] == weak:
                    if (
                        (self.thresholded[i+1, j-1] == strong) or (self.thresholded[i+1, j] == strong) or
                        (self.thresholded[i+1, j+1] == strong) or (self.thresholded[i, j-1] == strong) or
                        (self.thresholded[i, j+1] == strong) or (self.thresholded[i-1, j-1] == strong) or
                        (self.thresholded[i-1, j] == strong) or (self.thresholded[i-1, j+1] == strong)
                    ):
                        self.thresholded[i, j] = strong
                    else:
                        self.thresholded[i, j] = 0
        return self.thresholded



Example Usage

# Load image
img = cv.imread('assets/q4/image.jpg')

# Initialize the detector
detector = CannyEdgeDetector(img)

# Apply Gaussian Blur
gs = detector.apply_gaussian_blur()

# Compute Gradient
gm, gd = detector.compute_gradient()

# Non-Maximum Suppression
nms = detector.non_maximum_suppression()

# Thresholding
th = detector.threshold()

# Hysteresis
hy = detector.hysteresis()

# Plot results
fig, ax = plt.subplots(2, 3, figsize=(15, 5))
ax[0,0].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
ax[0,0].set_title('Original Image')
ax[0,0].axis("off")
ax[0,1].imshow(gs, cmap='gray')
ax[0,1].set_title('Gaussian Blur')
ax[0,1].axis("off")
ax[0,2].imshow(gm, cmap='gray')
ax[0,2].set_title('Gradient Magnitude')
ax[0,2].axis("off")
ax[1,0].imshow(nms, cmap='gray')
ax[1,0].set_title('Non-Maximum Suppression')
ax[1,0].axis("off")
ax[1,1].imshow(th, cmap='gray')
ax[1,1].set_title('Threshold')
ax[1,1].axis("off")
ax[1,2].imshow(hy, cmap='gray')
ax[1,2].set_title('Hysteresis')
ax[1,2].axis("off")
plt.show()

