# Phone Image Processing

This project contains scripts for processing phone images using various computer vision techniques.

## Features

- Phone detection using YOLOv8
- Background removal
- Image preprocessing and enhancement

## Requirements

- Python 3.8+
- OpenCV
- Ultralytics YOLO
- NumPy

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a conda environment:
```bash
conda create -n lab_image_test python=3.8
conda activate lab_image_test
```

3. Install dependencies:
```bash
pip install ultralytics opencv-python numpy
```

## Usage

### Phone Detection

To detect phones in an image using YOLOv8:

```bash
python yolo_phone_detector.py path/to/your/image.jpg
```

The script will:
1. Load the YOLOv8 model
2. Process the input image
3. Detect phones in the image
4. Save detected phones to the `yolo_processed_phones` directory

## Project Structure

- `yolo_phone_detector.py`: Main script for phone detection using YOLOv8
- `yolo_processed_phones/`: Directory for processed phone images
- `source_image/`: Directory for source images

## License

[Your chosen license] 