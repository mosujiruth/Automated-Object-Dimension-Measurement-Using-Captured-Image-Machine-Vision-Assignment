import cv2
import matplotlib.pyplot as plt

from skimage.measure import label, regionprops


MIN_AREA = 1000


def detect_objects_regionprops(binary_image, min_area=MIN_AREA):

    # Convert OpenCV binary image (0/255) into True/False
    binary_bool = binary_image > 0

    # Connected-component labelling
    label_image = label(
        binary_bool,
        connectivity=1,
        background=0
    )

    objects = []

    for region in regionprops(label_image):

        # Ignore small noise
        if region.area < min_area:
            continue

        minr, minc, maxr, maxc = region.bbox

        width = maxc - minc
        height = maxr - minr

        objects.append({
            "label": region.label,
            "area": region.area,
            "x": minc,
            "y": minr,
            "width_pixels": width,
            "height_pixels": height,
            "major_axis": region.axis_major_length,
            "minor_axis": region.axis_minor_length,
            "centroid": region.centroid
        })

    # Sort from left to right
    objects.sort(key=lambda obj: obj["x"])

    return objects


def draw_regionprops_boxes(image, objects):

    output = image.copy()

    for number, obj in enumerate(objects, start=1):

        x = obj["x"]
        y = obj["y"]
        w = obj["width_pixels"]
        h = obj["height_pixels"]

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

        cv2.putText(
            output,
            f"RP Object {number}",
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

    return output