import cv2
import numpy as np
import os
from typing import List, Tuple
import imutils

class PhoneDetector:
    def __init__(self):
        """
        Initialize the phone detector.
        """
        self.output_dir = "detection_output"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output directory created/verified: {self.output_dir}")

    def detect_and_isolate_phone(self, image: np.ndarray) -> np.ndarray:
        """
        Detect phone in the binary image and isolate it by removing background.
        
        Args:
            image: Input binary image
            
        Returns:
            Image with phone isolated and background removed
        """
        # Find contours in the binary image
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return image
        
        # Find the largest contour (assuming it's the phone)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create a mask for the phone
        mask = np.zeros_like(image)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)
        
        # Apply the mask to isolate the phone
        result = cv2.bitwise_and(image, mask)
        
        # Optional: Clean up small artifacts
        kernel = np.ones((3,3), np.uint8)
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
        
        return result

    def process_directory(self, source_dir: str = "preprocessed_images") -> List[str]:
        """
        Process all images in the source directory.
        
        Args:
            source_dir: Directory containing preprocessed images
            
        Returns:
            List of paths to the processed images
        """
        if not os.path.exists(source_dir):
            raise ValueError(f"Source directory {source_dir} does not exist")
            
        processed_paths = []
        
        # Get all image files
        image_files = [f for f in os.listdir(source_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        print(f"\nFound {len(image_files)} images to process:")
        for img in image_files:
            print(f"- {img}")
        
        for image_file in image_files:
            try:
                # Read image
                image_path = os.path.join(source_dir, image_file)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
                
                if image is None:
                    print(f"\nError: Could not read image {image_file}")
                    continue
                
                # Detect and isolate phone
                processed = self.detect_and_isolate_phone(image)
                
                # Determine output filename
                base_name, ext = os.path.splitext(image_file)
                output_filename = f"detected_{base_name}.jpg"
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
    # Initialize detector
    detector = PhoneDetector()
    
    # Process all images in preprocessed_images directory
    try:
        print("\nStarting phone detection and isolation...")
        processed_paths = detector.process_directory()
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