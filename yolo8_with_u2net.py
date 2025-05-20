import torch
import cv2
import numpy as np
from typing import List, Tuple
import os
import sys
from PIL import Image
import imutils
from ultralytics import YOLO
import torch.nn.functional as F
import torchvision.transforms as transforms

# U2Net model architecture
class REBNCONV(torch.nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()
        self.conv_s1 = torch.nn.Conv2d(in_ch, out_ch, 3, padding=1*dirate, dilation=1*dirate)
        self.bn_s1 = torch.nn.BatchNorm2d(out_ch)
        self.relu_s1 = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))
        return xout

class RSU7(torch.nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7, self).__init__()
        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)
        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool5 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv6d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)
        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)
        hx6 = self.rebnconv6(hx)
        hx7 = self.rebnconv7(hx6)
        hx6d = self.rebnconv6d(torch.cat((hx7,hx6),1))
        hx6dup = F.interpolate(hx6d, size=hx5.shape[2:], mode='bilinear', align_corners=True)
        hx5d = self.rebnconv5d(torch.cat((hx6dup,hx5),1))
        hx5dup = F.interpolate(hx5d, size=hx4.shape[2:], mode='bilinear', align_corners=True)
        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = F.interpolate(hx4d, size=hx3.shape[2:], mode='bilinear', align_corners=True)
        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = F.interpolate(hx3d, size=hx2.shape[2:], mode='bilinear', align_corners=True)
        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:], mode='bilinear', align_corners=True)
        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))
        return hx1d + hxin

class U2NET(torch.nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(U2NET, self).__init__()
        self.stage1 = RSU7(in_ch,32,64)
        self.pool12 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage2 = RSU7(64,32,128)
        self.pool23 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage3 = RSU7(128,64,256)
        self.pool34 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage4 = RSU7(256,128,512)
        self.pool45 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage5 = RSU7(512,256,512)
        self.pool56 = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.stage6 = RSU7(512,256,512)
        self.stage6d = RSU7(1024,256,512)
        self.stage5d = RSU7(1024,128,256)
        self.stage4d = RSU7(512,64,128)
        self.stage3d = RSU7(256,32,64)
        self.stage2d = RSU7(128,16,64)
        self.stage1d = RSU7(128,16,64)
        self.side1 = torch.nn.Conv2d(64,out_ch,3,padding=1)
        self.side2 = torch.nn.Conv2d(64,out_ch,3,padding=1)
        self.side3 = torch.nn.Conv2d(128,out_ch,3,padding=1)
        self.side4 = torch.nn.Conv2d(256,out_ch,3,padding=1)
        self.side5 = torch.nn.Conv2d(512,out_ch,3,padding=1)
        self.side6 = torch.nn.Conv2d(512,out_ch,3,padding=1)
        self.outconv = torch.nn.Conv2d(6*out_ch,out_ch,1)

    def forward(self,x):
        hx = x
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)
        hx6 = self.stage6(hx)
        hx6d = self.stage6d(torch.cat((hx6,hx5),1))
        hx6dup = F.interpolate(hx6d, size=hx4.shape[2:], mode='bilinear', align_corners=True)
        hx5d = self.stage5d(torch.cat((hx6dup,hx4),1))
        hx5dup = F.interpolate(hx5d, size=hx3.shape[2:], mode='bilinear', align_corners=True)
        hx4d = self.stage4d(torch.cat((hx5dup,hx3),1))
        hx4dup = F.interpolate(hx4d, size=hx2.shape[2:], mode='bilinear', align_corners=True)
        hx3d = self.stage3d(torch.cat((hx4dup,hx2),1))
        hx3dup = F.interpolate(hx3d, size=hx1.shape[2:], mode='bilinear', align_corners=True)
        hx2d = self.stage2d(torch.cat((hx3dup,hx1),1))
        hx2dup = F.interpolate(hx2d, size=hx1.shape[2:], mode='bilinear', align_corners=True)
        hx1d = self.stage1d(torch.cat((hx2dup,hx1),1))
        d1 = self.side1(hx1d)
        d2 = self.side2(hx2d)
        d2 = F.interpolate(d2, size=d1.shape[2:], mode='bilinear', align_corners=True)
        d3 = self.side3(hx3d)
        d3 = F.interpolate(d3, size=d1.shape[2:], mode='bilinear', align_corners=True)
        d4 = self.side4(hx4d)
        d4 = F.interpolate(d4, size=d1.shape[2:], mode='bilinear', align_corners=True)
        d5 = self.side5(hx5d)
        d5 = F.interpolate(d5, size=d1.shape[2:], mode='bilinear', align_corners=True)
        d6 = self.side6(hx6d)
        d6 = F.interpolate(d6, size=d1.shape[2:], mode='bilinear', align_corners=True)
        d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))
        return torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(d4), torch.sigmoid(d5), torch.sigmoid(d6)

class YOLOPhoneDetector:
    def __init__(self, target_size: Tuple[int, int] = (300, 600), conf_threshold: float = 0.1):
        """
        Initialize the YOLO phone detector.
        
        Args:
            target_size: Tuple of (width, height) for the output images
            conf_threshold: Confidence threshold for detections
        """
        print("Initializing YOLO model...")
        
        # Load the YOLOv8l model (large size, highest accuracy)
        self.model = YOLO('yolov8l.pt')
        
        # Set model parameters
        self.model.conf = conf_threshold  # Confidence threshold
        self.model.iou = 0.45  # NMS IoU threshold
        print("YOLO model loaded successfully")
        
        # Initialize U2Net model
        print("Initializing U2Net model...")
        self.u2net = U2NET()
        self.u2net.load_state_dict(torch.load('u2net.pth', map_location='cpu'))
        self.u2net.eval()
        print("U2Net model loaded successfully")
        
        self.target_size = target_size
        self.output_dir = "processed_phones"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output directory created/verified: {self.output_dir}")
        
        # Initialize image transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((320, 320)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _create_phone_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Create a mask for the phone using U2Net.
        
        Args:
            image: Input BGR image
            
        Returns:
            Binary mask where phone is white (255) and background is black (0)
        """
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(rgb_image)
            
            # Transform image
            img_tensor = self.transform(pil_image).unsqueeze(0)
            
            # Get prediction
            with torch.no_grad():
                d0, _, _, _, _, _, _ = self.u2net(img_tensor)
                pred = d0[:, 0, :, :]
                pred = pred.squeeze()
                pred_np = pred.cpu().data.numpy()
            
            # Resize prediction to original image size
            pred_np = cv2.resize(pred_np, (image.shape[1], image.shape[0]))
            
            # Convert to binary mask
            mask = (pred_np * 255).astype(np.uint8)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            
            return mask
            
        except Exception as e:
            print(f"Warning: Background removal failed: {e}")
            return np.zeros(image.shape[:2], dtype=np.uint8)

    def _calculate_phone_angle(self, binary_image: np.ndarray) -> float:
        """
        Calculate the angle of the phone using PCA on the binary image points.
        
        Args:
            binary_image: Binary image of the phone
            
        Returns:
            Angle in degrees
        """
        # Find points that belong to the phone (white pixels)
        y_coords, x_coords = np.nonzero(binary_image)
        coords = np.column_stack((x_coords, y_coords))
        
        # Calculate PCA
        mean = np.mean(coords, axis=0)
        coords_centered = coords - mean
        cov = np.cov(coords_centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Get the angle from the principal component
        principal_vector = eigenvectors[:, -1]  # Vector with largest eigenvalue
        angle = np.degrees(np.arctan2(principal_vector[1], principal_vector[0]))
        
        # Normalize angle to be between 0 and 180 degrees
        angle = angle % 180
        
        # Additional check to ensure phone is not upside down
        # Calculate the center of mass of the phone
        center_y = np.mean(y_coords)
        height = binary_image.shape[0]
        
        # If the center of mass is in the bottom half, flip the phone
        if center_y > height / 2:
            angle = (angle + 180) % 180
            
        return angle

    def _process_phone_image(self, phone_img: np.ndarray) -> np.ndarray:
        """
        Process a single phone image to standardize its size and orientation.
        
        Args:
            phone_img: Cropped phone image
            
        Returns:
            Processed phone image
        """
        # Get the initial mask using U2Net
        mask = self._create_phone_mask(phone_img)
        
        # Calculate the angle
        current_angle = self._calculate_phone_angle(mask)
        print(f"Detected phone angle: {current_angle:.2f} degrees")
        
        # Rotate if needed
        if abs(current_angle - 90) > 0.5:
            rotation_angle = 90 - current_angle
            print(f"Rotating image by {rotation_angle:.2f} degrees to make it vertical")
            rotated = imutils.rotate_bound(phone_img, rotation_angle)
            # Update mask after rotation
            mask = self._create_phone_mask(rotated)
            print("Image has been rotated to vertical orientation")
        else:
            print("Image is already vertical (within 0.5 degrees), no rotation needed")
            rotated = phone_img
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get the bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add padding
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(rotated.shape[1] - x, w + 2 * padding)
            h = min(rotated.shape[0] - y, h + 2 * padding)
            
            # Crop the image
            cropped = rotated[y:y+h, x:x+w]
            # Update mask for cropped region
            mask = mask[y:y+h, x:x+w]
        else:
            cropped = rotated
            mask = self._create_phone_mask(cropped)
        
        # Resize to target size while maintaining aspect ratio
        h, w = cropped.shape[:2]
        aspect = w / float(h)
        
        # Calculate new dimensions
        new_h = self.target_size[1]
        new_w = int(new_h * aspect)
        
        if new_w > self.target_size[0]:
            new_w = self.target_size[0]
            new_h = int(new_w / aspect)
        
        # Resize the image and mask
        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
        resized_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # Create a transparent background
        final = np.zeros((self.target_size[1], self.target_size[0], 4), dtype=np.uint8)
        
        # Convert resized image to RGBA
        if resized.shape[2] == 3:
            resized_rgba = cv2.cvtColor(resized, cv2.COLOR_BGR2BGRA)
        else:
            resized_rgba = resized
        
        # Apply the mask to create transparency
        resized_rgba[:, :, 3] = resized_mask
        
        # Calculate position to paste the resized image
        x_offset = (self.target_size[0] - new_w) // 2
        y_offset = (self.target_size[1] - new_h) // 2
        
        # Paste the resized image onto the transparent background
        final[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_rgba
        
        return final

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
        
        # Create a copy of the image for visualization
        vis_image = image.copy()
        
        # Process detections
        phone_images = []
        print(f"Number of detections: {len(results[0].boxes)}")
        
        # Get base name for unique output naming
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Print all detections for debugging
        print("\nAll detections:")
        for i, detection in enumerate(results[0].boxes):
            x1, y1, x2, y2 = map(int, detection.xyxy[0])
            conf = float(detection.conf[0])
            cls = int(detection.cls[0])
            print(f"Detection {i+1}: Class {cls}, Confidence {conf:.2f}")
            
            # Draw all detections on visualization image
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_image, f"Phone ({conf:.2f})", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Process all detections as phones (since model is trained for phone detection)
            print(f"Processing phone detection {i+1}")
            # Add padding
            padding = 20  # Increased padding
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image.shape[1], x2 + padding)
            y2 = min(image.shape[0], y2 + padding)
            
            # Crop phone image using bounding box
            phone_img = image[y1:y2, x1:x2]
            
            # Process and resize with background removal
            processed_phone = self._process_phone_image(phone_img)
            phone_images.append(processed_phone)
            
            # Save the processed phone image with a unique name
            output_path = os.path.join(self.output_dir, f"{base_name}_phone_{i+1}.png")
            cv2.imwrite(output_path, processed_phone)
            print(f"Saved phone {i+1} to: {output_path}")
            print(f"Confidence: {conf:.2f}")
        
        # Save visualization image with unique name
        vis_path = os.path.join(self.output_dir, f"{base_name}_detection_visualization.jpg")
        cv2.imwrite(vis_path, vis_image)
        print(f"\nSaved detection visualization to: {vis_path}")
        
        return phone_images

def detect_phones_yolo(image_path: str) -> List[str]:
    """
    Process an image containing multiple phones using YOLOv5.
    
    Args:
        image_path: Path to the input image
        
    Returns:
        List of paths to the saved phone images
    """
    print(f"Starting phone detection for image: {image_path}")
    detector = YOLOPhoneDetector(conf_threshold=0.1)  # Lower confidence threshold
    phone_images = detector.detect_phones(image_path)
    
    return [os.path.join(detector.output_dir, f"phone_{i+1}.png") 
            for i in range(len(phone_images))]

def main():
    # Set input directory
    input_dir = "preprocessed_images"
    
    # Get all image files from input directory
    image_files = [f for f in os.listdir(input_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
    
    print(f"\nFound {len(image_files)} images to process:")
    for img in image_files:
        print(f"- {img}")
    
    # Process each image
    total_processed = 0
    for image_file in image_files:
        try:
            image_path = os.path.join(input_dir, image_file)
            print(f"\nProcessing {image_file}...")
            saved_paths = detect_phones_yolo(image_path)
            total_processed += len(saved_paths)
            print(f"Successfully processed {len(saved_paths)} phones from {image_file}")
            for path in saved_paths:
                print(f"  - Saved to: {path}")
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
            continue
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed {total_processed} phones from {len(image_files)} images")

if __name__ == "__main__":
    main() 