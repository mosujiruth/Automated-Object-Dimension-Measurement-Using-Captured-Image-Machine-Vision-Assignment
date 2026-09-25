import cv2
import numpy as np

from image_processing import (
    resize_image,
    convert_to_grayscale,
    apply_gaussian_blur,
    apply_threshold,
    apply_morphology
)

from praveen_object_detection import (
    detect_objects,
    draw_bounding_boxes
)

from praveen_regionprops_detection import (
    detect_objects_regionprops,
    draw_regionprops_boxes
)


# -------------------------------------------------
# OPEN WEBCAM
# -------------------------------------------------

cap = cv2.VideoCapture(0)


# -------------------------------------------------
# MAIN LOOP
# -------------------------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        print("Cannot read webcam")
        break


    # -------------------------------------------------
    # PERSON 1: PREPROCESSING
    # -------------------------------------------------

    image = resize_image(frame)

    gray = convert_to_grayscale(image)

    blurred = apply_gaussian_blur(gray)

    binary = apply_threshold(blurred)

    cleaned = apply_morphology(binary)


    # -------------------------------------------------
    # METHOD 1: CONTOUR DETECTION
    # -------------------------------------------------

    contour_objects = detect_objects(
        cleaned,
        min_area=1000
    )

    contour_result = draw_bounding_boxes(
        image,
        contour_objects
    )


    # -------------------------------------------------
    # METHOD 2: REGIONPROPS DETECTION
    # -------------------------------------------------

    region_objects = detect_objects_regionprops(
        cleaned,
        min_area=1000
    )

    region_result = draw_regionprops_boxes(
        image,
        region_objects
    )


    # -------------------------------------------------
    # DISPLAY METHOD TITLES
    # -------------------------------------------------

    cv2.putText(
        contour_result,
        "Contour Method",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        region_result,
        "Regionprops Method",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )


    # -------------------------------------------------
    # SIDE-BY-SIDE COMPARISON
    # -------------------------------------------------

    comparison = np.hstack(
        (
            contour_result,
            region_result
        )
    )


    # -------------------------------------------------
    # DISPLAY WINDOWS
    # -------------------------------------------------

    cv2.imshow(
        "Contour vs Regionprops Comparison",
        comparison
    )

    cv2.imshow(
        "Cleaned Binary",
        cleaned
    )


    # -------------------------------------------------
    # KEY CONTROLS
    # -------------------------------------------------

    key = cv2.waitKey(1) & 0xFF


    # -------------------------------------------------
    # PRESS S TO PRINT ONE COMPARISON
    # -------------------------------------------------

    if key == ord("s"):

        if (
            len(contour_objects) > 0
            and len(region_objects) > 0
        ):

            contour_obj = contour_objects[0]
            region_obj = region_objects[0]

            contour_width = contour_obj[
                "width_pixels"
            ]

            contour_height = contour_obj[
                "height_pixels"
            ]

            region_width = region_obj[
                "width_pixels"
            ]

            region_height = region_obj[
                "height_pixels"
            ]

            width_difference = abs(
                contour_width
                - region_width
            )

            height_difference = abs(
                contour_height
                - region_height
            )


            print("\n-------------------------------")
            print("DETECTION METHOD COMPARISON")
            print("-------------------------------")

            print(
                f"Contour Method: "
                f"x={contour_obj['x']}, "
                f"y={contour_obj['y']}, "
                f"W={contour_width}px, "
                f"H={contour_height}px"
            )

            print(
                f"Regionprops Method: "
                f"x={region_obj['x']}, "
                f"y={region_obj['y']}, "
                f"W={region_width}px, "
                f"H={region_height}px"
            )

            print(
                f"Width Difference = "
                f"{width_difference}px"
            )

            print(
                f"Height Difference = "
                f"{height_difference}px"
            )

            print("-------------------------------")

        else:

            print(
                "No object detected by both methods."
            )


    # -------------------------------------------------
    # PRESS Q TO QUIT
    # -------------------------------------------------

    if key == ord("q"):
        break


# -------------------------------------------------
# CLEAN EXIT
# -------------------------------------------------

cap.release()

cv2.destroyAllWindows()