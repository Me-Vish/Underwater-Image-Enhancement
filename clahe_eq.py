import cv2
import matplotlib.pyplot as plt

# Load image
img = cv2.imread("assets/underwater.jpeg.jpeg")

if img is None:
    print("⚠️ Error: Image not found. Make sure 'sample.jpg' is in the project folder.")
    exit()

# Convert to LAB color space
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)

# Apply CLAHE
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
cl = clahe.apply(l)

# Merge channels back
limg = cv2.merge((cl, a, b))
clahe_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

# Save output
cv2.imwrite("clahe.jpg", clahe_img)

# Show Original vs Enhanced
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(clahe_img, cv2.COLOR_BGR2RGB))
plt.title("CLAHE Enhanced")

plt.show()

print("✅ CLAHE Enhancement complete! Saved as clahe.jpg")
