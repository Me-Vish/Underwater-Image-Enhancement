import cv2
import matplotlib.pyplot as plt

# Load image
img = cv2.imread("enhanced.jpg")

if img is None:
    print("⚠️ Error: Image not found. Make sure 'sample.jpg' is in the project folder.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Histogram Equalization
hist_eq = cv2.equalizeHist(gray)

# Save output
cv2.imwrite("hist_eq.jpg", hist_eq)

# Show Original vs Enhanced
plt.subplot(1, 2, 1)
plt.imshow(gray, cmap="gray")
plt.title("Original (Gray)")

plt.subplot(1, 2, 2)
plt.imshow(hist_eq, cmap="gray")
plt.title("Histogram Equalized")

plt.show()

print("✅ Histogram Equalization complete! Saved as hist_eq.jpg")
