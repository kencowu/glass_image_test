from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Tuple
import os
import sys

class YOLOPhoneDetector:
    def __init__(self, target_size: Tuple[int, int] = (300, 600), conf_threshold: float = 0.1):
        """
        Initialize the YOLO phone detector.
        
        Args:
            target_size: Tuple of (width, height) for the output images
            conf_threshold: Confidence threshold for detections
        """
        print("Initializing YOLO model...")
        # Load the YOLOv8m model (medium size, better accuracy)
        self.model = YOLO('yolov8m.pt')
        # Set model parameters
        self.model.conf = conf_threshold  # Lower confidence threshold
        self.model.iou = 0.45  # IOU threshold for NMS
        self.model.agnostic = True  # Class-agnostic NMS
        self.model.multi_label = True  # Allow multiple labels per box
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
            
        # Run YOLOv8 inference with specific parameters
        print("Running YOLO inference...")
        results = self.model(image, verbose=True)
        print("YOLO inference completed")
        
        # Create a copy of the image for visualization
        vis_image = image.copy()
        
        # Process detections
        phone_images = []
        print(f"Number of detections: {len(results[0].boxes.data)}")
        
        # Print all detections for debugging
        print("\nAll detections:")
        for i, detection in enumerate(results[0].boxes.data):
            x1, y1, x2, y2, conf, cls = detection
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            print(f"Detection {i+1}: Class {cls}, Confidence {conf:.2f}")
            
            # Draw all detections on visualization image
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_image, f"Class {cls:.0f} ({conf:.2f})", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Check if detected object is a cell phone (class 67 in COCO dataset)
            if cls == 67:  # Cell phone class
                print(f"Processing phone detection {i+1}")
                # Add padding
                padding = 20  # Increased padding
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(image.shape[1], x2 + padding)
                y2 = min(image.shape[0], y2 + padding)
                
                # Crop phone image using bounding box only
                phone_img = image[y1:y2, x1:x2]
                
                # Process and resize
                processed_phone = self._process_phone_image(phone_img)
                phone_images.append(processed_phone)
                
                # Save the processed phone image
                output_path = os.path.join(self.output_dir, f"phone_{i+1}.jpg")
                cv2.imwrite(output_path, processed_phone)
                print(f"Saved phone {i+1} to: {output_path}")
                print(f"Confidence: {conf:.2f}")
        
        # Save visualization image
        vis_path = os.path.join(self.output_dir, "detection_visualization.jpg")
        cv2.imwrite(vis_path, vis_image)
        print(f"\nSaved detection visualization to: {vis_path}")
        
        # Print COCO class names for reference
        print("\nCOCO class names for reference:")
        coco_names = self.model.names
        for class_id, class_name in coco_names.items():
            print(f"Class {class_id}: {class_name}")
        
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
    detector = YOLOPhoneDetector(conf_threshold=0.1)  # Lower confidence threshold
    phone_images = detector.detect_phones(image_path)
    
    return [os.path.join(detector.output_dir, f"phone_{i+1}.jpg") 
            for i in range(len(phone_images))]

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
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