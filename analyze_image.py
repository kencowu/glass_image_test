import cv2
import numpy as np

def analyze_image(image_path):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return
    
    # Print basic image information
    print(f"Image shape: {img.shape}")
    print(f"Image type: {img.dtype}")
    print(f"Image min value: {np.min(img)}")
    print(f"Image max value: {np.max(img)}")
    print(f"Image mean value: {np.mean(img)}")
    
    # Convert to grayscale for additional analysis
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"\nGrayscale image mean: {np.mean(gray)}")
    print(f"Grayscale image std: {np.std(gray)}")
    
    # Check for any non-zero pixels
    non_zero = np.count_nonzero(gray)
    total_pixels = gray.size
    print(f"\nNon-zero pixels: {non_zero}")
    print(f"Total pixels: {total_pixels}")
    print(f"Percentage of non-zero pixels: {(non_zero/total_pixels)*100:.2f}%")

if __name__ == "__main__":
    image_path = "/Users/kencowu/Desktop/Cursor/lab_image_test/source_image_jpg/the-smartphone-hit-the-floor-it-fell-into-a-crack-free-photo.jpeg"
    analyze_image(image_path) 