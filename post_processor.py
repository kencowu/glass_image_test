import cv2
import numpy as np
import os
from typing import List, Tuple
import imutils

class ImagePostProcessor:
    def __init__(self):
        """
        Initialize the image post-processor.
        """
        self.input_dir = "processed_phones"
        self.output_dir = "post_processed_image"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output directory created/verified: {self.output_dir}")

    def _restore_colors(self, image: np.ndarray) -> np.ndarray:
        """
        Restore original colors by removing the adaptive thresholding effect.
        
        Args:
            image: Input BGR image
            
        Returns:
            Image with restored colors
        """
        # Convert to LAB color space for better color processing
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply bilateral filter to reduce noise while preserving edges
        l = cv2.bilateralFilter(l, 9, 75, 75)
        
        # Merge channels back
        lab = cv2.merge((l, a, b))
        
        # Convert back to BGR
        restored = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Apply slight contrast enhancement
        restored = cv2.convertScaleAbs(restored, alpha=1.1, beta=5)
        
        return restored

    def process_image(self, image_path: str) -> str:
        """
        Process a single image.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Path to the processed image
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
        
        # Restore colors
        restored = self._restore_colors(image)
        
        # Save processed image
        output_path = os.path.join(self.output_dir, f"restored_{os.path.basename(image_path)}")
        cv2.imwrite(output_path, restored)
        
        return output_path

    def process_directory(self) -> List[str]:
        """
        Process all images in the input directory.
        
        Returns:
            List of paths to the processed images
        """
        if not os.path.exists(self.input_dir):
            raise ValueError(f"Input directory {self.input_dir} does not exist")
            
        processed_paths = []
        
        # Get all image files
        image_files = [f for f in os.listdir(self.input_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
        
        print(f"\nFound {len(image_files)} images to process:")
        for img in image_files:
            print(f"- {img}")
        
        for image_file in image_files:
            try:
                # Process image
                image_path = os.path.join(self.input_dir, image_file)
                output_path = self.process_image(image_path)
                processed_paths.append(output_path)
                print(f"✓ Successfully processed: {image_file} -> {os.path.basename(output_path)}")
                
            except Exception as e:
                print(f"\nError processing {image_file}: {str(e)}")
                continue
            
        return processed_paths

def main():
    # Initialize post-processor
    post_processor = ImagePostProcessor()
    
    # Process all images in processed_phones directory
    try:
        print("\nStarting image post-processing...")
        processed_paths = post_processor.process_directory()
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