from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Tuple
import os

class YOLOPhoneDetector:
    def __init__(self, target_size: Tuple[int, int] = (300, 600)):
        """
        Initialize the YOLO phone detector.
        
        Args:
            target_size: Tuple of (width, height) for the output images
        """
        print("Initializing YOLO model...")
        self.model = YOLO('yolov8n-seg.pt')  # Load the YOLOv8 segmentation model
        print("YOLO model loaded successfully")
        self.target_size = target_size
        self.output_dir = "yolo_processed_phones"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output directory created/verified: {self.output_dir}")
        
    def detect_phones(self, image_path: str) -> List[np.ndarray]:
        """
        Detect and extract individual phone images using YOLOv8.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            List of cropped and processed phone images
        """
        print(f"Reading image from: {image_path}")
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
        print(f"Image read successfully. Shape: {image.shape}")
            
        # Run YOLOv8 inference
        print("Running YOLO inference...")
        results = self.model(image)
        print("YOLO inference completed")
        
        # Process detections
        phone_images = []
        print(f"Number of detections: {len(results[0].boxes.data)}")
        for i, detection in enumerate(results[0].boxes.data):
            # Get box coordinates
            x1, y1, x2, y2, conf, cls = detection
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            print(f"Detection {i+1}: Class {cls}, Confidence {conf:.2f}")
            
            # Check if detected object is a cell phone (class 67 in COCO dataset)
            if cls == 67:  # Cell phone class
                print(f"Processing phone detection {i+1}")
                # Add padding
                padding = 10
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(image.shape[1], x2 + padding)
                y2 = min(image.shape[0], y2 + padding)
                
                # Get the mask for this detection
                print("Getting mask for detection...")
                mask = results[0].masks.data[i].cpu().numpy()
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
                
                # Create a binary mask
                binary_mask = (mask > 0.5).astype(np.uint8) * 255
                
                # Apply the mask to the image
                masked_image = cv2.bitwise_and(image, image, mask=binary_mask)
                
                # Crop phone image
                phone_img = masked_image[y1:y2, x1:x2]
                
                # Process and resize
                processed_phone = self._process_phone_image(phone_img)
                phone_images.append(processed_phone)
                
                # Save the processed phone image
                output_path = os.path.join(self.output_dir, f"phone_{i+1}.jpg")
                cv2.imwrite(output_path, processed_phone)
                print(f"Saved phone {i+1} to: {output_path}")
                print(f"Confidence: {conf:.2f}")
        
        return phone_images
    
    def _process_phone_image(self, phone_img: np.ndarray) -> np.ndarray:
        """
        Process a single phone image to standardize its size.
        
        Args:
            phone_img: Cropped phone image
            
        Returns:
            Processed phone image
        """
        # Get current dimensions
        h, w = phone_img.shape[:2]
        aspect = w / float(h)
        
        # Calculate new dimensions
        new_h = self.target_size[1]
        new_w = int(new_h * aspect)
        
        if new_w > self.target_size[0]:
            new_w = self.target_size[0]
            new_h = int(new_w / aspect)
        
        # Resize the image
        resized = cv2.resize(phone_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create a transparent background
        final = np.zeros((self.target_size[1], self.target_size[0], 4), dtype=np.uint8)
        
        # Convert resized image to RGBA
        if resized.shape[2] == 3:
            resized_rgba = cv2.cvtColor(resized, cv2.COLOR_BGR2BGRA)
        else:
            resized_rgba = resized
            
        # Set alpha channel based on non-zero pixels
        alpha = np.any(resized_rgba[:, :, :3] > 0, axis=2).astype(np.uint8) * 255
        resized_rgba[:, :, 3] = alpha
        
        # Center the image
        x_offset = (self.target_size[0] - new_w) // 2
        y_offset = (self.target_size[1] - new_h) // 2
        
        # Paste the resized image
        final[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_rgba
        
        return final

def detect_phones_yolo(image_path: str) -> List[str]:
    """
    Process an image containing multiple phones using YOLOv8.
    
    Args:
        image_path: Path to the input image
        
    Returns:
        List of paths to the saved phone images
    """
    print(f"Starting phone detection for image: {image_path}")
    detector = YOLOPhoneDetector()
    phone_images = detector.detect_phones(image_path)
    
    return [os.path.join(detector.output_dir, f"phone_{i+1}.jpg") 
            for i in range(len(phone_images))]

if __name__ == "__main__":
    # Example usage
    image_path = "modern-touch-screen-smartphone-broken-600nw-2524629559.webp"
    try:
        print(f"Processing image: {image_path}")
        saved_paths = detect_phones_yolo(image_path)
        print(f"\nProcessed {len(saved_paths)} phones:")
        for path in saved_paths:
            print(f"- {path}")
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc() 