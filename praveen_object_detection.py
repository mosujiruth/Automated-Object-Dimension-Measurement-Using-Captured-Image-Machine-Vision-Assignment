import cv2

MIN_AREA = 1000


def detect_objects(binary_image, min_area=MIN_AREA):

    contours, _ = cv2.findContours(
        binary_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    objects = []

    image_height, image_width = binary_image.shape[:2]

    image_area = image_width * image_height

    for contour in contours:

        area = cv2.contourArea(contour)

        # Remove tiny noise
        if area < min_area:
            continue

        # Remove extremely large regions/background
        if area > image_area * 0.50:
            continue

        x, y, w, h = cv2.boundingRect(contour)

    

        objects.append({
            "contour": contour,
            "area": area,
            "x": x,
            "y": y,
            "width_pixels": w,
            "height_pixels": h
        })

    objects.sort(key=lambda obj: obj["x"])

    return objects


def draw_bounding_boxes(image, objects):

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
            (0, 255, 0),
            2
        )

        cv2.putText(
            output,
            f"Object {number}",
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

      # print(
#     f"Object {number}: "
#     f"x={x}, y={y}, "
#     f"width={w}px, height={h}px, "
#     f"area={obj['area']:.2f}px"
# )

    return output