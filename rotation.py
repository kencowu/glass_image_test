import os
from PIL import Image
import cv2
import numpy as np

def intersection(line1, line2):
    rho1, theta1 = line1
    rho2, theta2 = line2
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    if np.linalg.det(A) == 0:
        return None
    x0, y0 = np.linalg.solve(A, b)
    return [float(x0), float(y0)]

def find_four_edges_and_corners(image, debug_path=None):
    """
    Thresholding + Hough Line approach: Detects four straight lines (top, bottom, left, right)
    using Hough Line Transform on a thresholded image. Draws the four lines and their intersections
    (corners) in the debug image. Returns the four corners if found.
    """
    import math

    def intersection(line1, line2):
        rho1, theta1 = line1
        rho2, theta2 = line2
        A = np.array([
            [np.cos(theta1), np.sin(theta1)],
            [np.cos(theta2), np.sin(theta2)]
        ])
        b = np.array([[rho1], [rho2]])
        if np.linalg.det(A) == 0:
            return None
        x0, y0 = np.linalg.solve(A, b)
        return [float(x0), float(y0)]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Morphological close to connect edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Hough Line Transform
    lines = cv2.HoughLines(closed, 1, np.pi / 180, threshold=120)
    debug_img = image.copy()
    h_img, w_img = image.shape[:2]
    y_center = h_img // 2
    x_center = w_img // 2

    # Draw all detected lines in green
    if lines is not None:
        for l in lines:
            rho, theta = l[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv2.line(debug_img, pt1, pt2, (0, 255, 0), 1)

    # Separate lines into near-vertical and near-horizontal
    verticals = []
    horizontals = []
    if lines is not None:
        for l in lines:
            rho, theta = l[0]
            angle = np.degrees(theta)
            if (80 < angle < 100) or (260 < angle < 280):  # near-vertical
                if abs(np.cos(theta)) > 1e-6:
                    x_int = (rho - y_center * np.sin(theta)) / np.cos(theta)
                    verticals.append((rho, theta, x_int))
            elif (angle < 10 or angle > 170):  # near-horizontal
                if abs(np.sin(theta)) > 1e-6:
                    y_int = (rho - x_center * np.cos(theta)) / np.sin(theta)
                    horizontals.append((rho, theta, y_int))

    # Select leftmost and rightmost verticals by x-intercept
    selected_verticals = []
    if len(verticals) >= 2:
        verticals_sorted = sorted(verticals, key=lambda x: x[2])
        selected_verticals = [verticals_sorted[0][:2], verticals_sorted[-1][:2]]
    elif len(verticals) == 1:
        selected_verticals = [verticals[0][:2], verticals[0][:2]]

    # Select topmost and bottommost horizontals by y-intercept
    selected_horizontals = []
    if len(horizontals) >= 2:
        horizontals_sorted = sorted(horizontals, key=lambda x: x[2])
        selected_horizontals = [horizontals_sorted[0][:2], horizontals_sorted[-1][:2]]
    elif len(horizontals) == 1:
        selected_horizontals = [horizontals[0][:2], horizontals[0][:2]]

    # Draw the four selected edges in yellow and thicker
    for l in selected_verticals + selected_horizontals:
        rho, theta = l
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        cv2.line(debug_img, pt1, pt2, (0, 255, 255), 4)

    # Find corners by intersection
    corners = []
    if len(selected_verticals) == 2 and len(selected_horizontals) == 2:
        for h in selected_horizontals:
            for v in selected_verticals:
                pt = intersection(h, v)
                if pt is not None:
                    corners.append(pt)
        if len(corners) == 4:
            corners = np.array(corners)
            s = corners.sum(axis=1)
            diff = np.diff(corners, axis=1)
            ordered = np.zeros((4, 2), dtype="float32")
            ordered[0] = corners[np.argmin(s)]  # top-left
            ordered[2] = corners[np.argmax(s)]  # bottom-right
            ordered[1] = corners[np.argmin(diff)]  # top-right
            ordered[3] = corners[np.argmax(diff)]  # bottom-left
            # Draw corners in red
            for idx, pt in enumerate(ordered):
                cv2.circle(debug_img, (int(pt[0]), int(pt[1])), 10, (0, 0, 255), -1)
                cv2.putText(debug_img, str(idx), (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            corners = ordered
        else:
            corners = None
    else:
        corners = None

    # Always save debug images
    if debug_path is not None:
        cv2.imwrite(debug_path, debug_img)

    if corners is not None and len(corners) == 4:
        return corners, (selected_verticals, selected_horizontals)
    else:
        return None, (selected_verticals, selected_horizontals)

def four_point_transform(image, pts):
    rect = pts
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    if maxWidth < 10 or maxHeight < 10:
        print("Invalid output size, skipping transform.")
        return image

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def rotate_to_vertical(image_path, save_path=None, overwrite=True):
    """
    Detects the four main edges of the phone, computes their intersections (corners),
    applies a perspective transform to make the phone vertical, and saves the result.
    Always saves a debug image with detected edges and corners.
    """
    try:
        img_cv = cv2.imread(image_path)
        if img_cv is None:
            print(f"Error: Cannot read {image_path}")
            return False

        debug_path = None
        if save_path is not None:
            base, ext = os.path.splitext(save_path)
            debug_path = f"{base}_edges_debug{ext}"

        corners, _ = find_four_edges_and_corners(img_cv, debug_path=debug_path)
        if corners is not None and len(corners) == 4:
            warped = four_point_transform(img_cv, corners)
            rotated_pil = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        else:
            print(f"Could not find four edges/corners for {image_path}, saving original.")
            rotated_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

        if save_path is not None:
            rotated_pil.save(save_path)
        elif overwrite:
            rotated_pil.save(image_path)
        else:
            print("No valid save path specified.")
            return False
        return True
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False

def process_all_images_in_folder(input_folder, output_folder):
    """
    Processes all images in the given folder, rectifies them using the four main edges,
    and saves results to output_folder. Also saves debug images.
    """
    supported_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(supported_ext):
            image_path = os.path.join(input_folder, filename)
            save_path = os.path.join(output_folder, filename)
            print(f"Processing {image_path} -> {save_path} ...")
            success = rotate_to_vertical(image_path, save_path=save_path, overwrite=False)
            if success:
                print(f"Rectified {filename} and saved to output_rotated (with debug edges).")
            else:
                print(f"Failed to process {filename}.")

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    input_folder = os.path.join(base_dir, "processed_phones")
    output_folder = os.path.join(base_dir, "output_rotated")
    process_all_images_in_folder(input_folder, output_folder)