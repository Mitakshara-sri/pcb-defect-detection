from pathlib import Path
import cv2
import matplotlib.pyplot as plt
# Define absolute path relative to project root
BASE_DIR = Path(__file__).resolve().parent.parent
IMAGE_PATH = BASE_DIR / "data" / "sample.jpg"
# Read image
img = cv2.imread(str(IMAGE_PATH))
if img is None:
    raise FileNotFoundError(f"Image not found at {IMAGE_PATH}. Please make sure 'sample.jpg' is in the 'data' folder.")
# Convert BGR (OpenCV default) to RGB (Matplotlib default)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print("Image loaded successfully!")
print("Image shape:", img.shape)
# Display image
plt.figure(figsize=(10, 7))
plt.imshow(img_rgb)
plt.title("Sample PCB Image")
plt.axis("off")
plt.show()
