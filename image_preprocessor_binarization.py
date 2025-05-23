import cv2
import numpy as np
import os
from typing import List, Tuple
import imutils

class ImagePreprocessor:
    def __init__(self, target_size: Tuple[int, int] = (640, 640)):
        """
        Initialize the image preprocessor.
        
        Args:
            target_size: Tuple of (width, height) for the output images
        """
        self.target_size = target_size
        self.output_dir = "preprocessed_images"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output directory created/verified: {self.output_dir}")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess a single image using adaptive thresholding for binarization.
        
        Args:
            image: Input BGR image
            
        Returns:
            Preprocessed binary image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,  # Block size
            2    # C constant
        )
        
        # Apply morphological operations to clean up the binary image
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Calculate aspect ratio
        h, w = binary.shape[:2]
        aspect = w / float(h)
        
        # Calculate new dimensions while maintaining aspect ratio
        if aspect > 1:  # Wider than tall
            new_w = self.target_size[0]
            new_h = int(new_w / aspect)
        else:  # Taller than wide
            new_h = self.target_size[1]
            new_w = int(new_h * aspect)
            
        # Resize image while maintaining aspect ratio
        resized = cv2.resize(binary, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create a square canvas with padding
        square = np.zeros((self.target_size[1], self.target_size[0]), dtype=np.uint8)
        
        # Calculate padding
        x_offset = (self.target_size[0] - new_w) // 2
        y_offset = (self.target_size[1] - new_h) // 2
        
        # Place the resized image in the center of the square canvas
        square[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return square

    def process_directory(self, source_dir: str = "source_image") -> List[str]:
        """
        Process all images in the source directory.
        
        Args:
            source_dir: Directory containing source images
            
        Returns:
            List of paths to the processed images
        """
        if not os.path.exists(source_dir):
            raise ValueError(f"Source directory {source_dir} does not exist")
            
        processed_paths = []
        
        # Get all image files with more comprehensive extensions
        image_files = [f for f in os.listdir(source_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.avif', '.tiff', '.tif'))]
        
        print(f"\nFound {len(image_files)} images to process:")
        for img in image_files:
            print(f"- {img}")
        
        for image_file in image_files:
            try:
                # Read image
                image_path = os.path.join(source_dir, image_file)
                image = cv2.imread(image_path)
                
                if image is None:
                    print(f"\nError: Could not read image {image_file}")
                    continue
                    
                # Preprocess image
                processed = self.preprocess_image(image)
                
                # Determine output filename and extension
                base_name, ext = os.path.splitext(image_file)
                # Convert all output to jpg for consistency
                output_filename = f"preprocessed_{base_name}.jpg"
                output_path = os.path.join(self.output_dir, output_filename)
                
                # Save the processed image
                cv2.imwrite(output_path, processed)
                processed_paths.append(output_path)
                
                print(f"✓ Successfully processed: {image_file} -> {output_filename}")
                
            except Exception as e:
                print(f"\nError processing {image_file}: {str(e)}")
                continue
            
        return processed_paths

def main():
    # Initialize preprocessor
    preprocessor = ImagePreprocessor()
    
    # Process all images in source_image directory
    try:
        print("\nStarting image preprocessing...")
        processed_paths = preprocessor.process_directory()
        print(f"\nProcessing complete!")
        print(f"Successfully processed {len(processed_paths)} images:")
        for path in processed_paths:
            print(f"- {path}")
    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 