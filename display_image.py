import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_image(image_path):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return
    
    # Convert BGR to RGB for matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create a figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Display original image
    ax1.imshow(img_rgb)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Display grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ax2.imshow(gray, cmap='gray')
    ax2.set_title('Grayscale Image')
    ax2.axis('off')
    
    # Add image information
    plt.figtext(0.5, 0.01, f'Image Shape: {img.shape}\nMean Value: {np.mean(img):.2f}\nStd Dev: {np.std(img):.2f}', 
                ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('image_analysis.png')
    plt.close()

if __name__ == "__main__":
    image_path = "/Users/kencowu/Desktop/Cursor/lab_image_test/source_image_jpg/the-smartphone-hit-the-floor-it-fell-into-a-crack-free-photo.jpeg"
    display_image(image_path) 