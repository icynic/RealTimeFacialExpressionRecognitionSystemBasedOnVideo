import cv2 as cv
import mediapipe as mp
import numpy as np
import time
from icecream import ic
from classifier import ExpressionClassifier


mp_model_path = "models/face_landmarker.task"
classifier_model_path = "models/expression_classifier.tflite"
categories_path = "blendshapes/categories.csv"
max_faces = 100





# Global variable to store the latest detected bounding boxes
latest_bounding_boxes = []






def landmarks_to_rectangle(result, output_image):
    global latest_bounding_boxes
    current_boxes = []

    # Get image dimensions for denormalization
    img_h = output_image.height
    img_w = output_image.width

    if result.face_landmarks:
        for landmarks_list in result.face_landmarks:
            if not landmarks_list:  # Check if the list of landmarks is empty
                continue

            # Initialize min/max coordinates
            min_x = landmarks_list[0].x
            max_x = landmarks_list[0].x
            min_y = landmarks_list[0].y
            max_y = landmarks_list[0].y

            for landmark in landmarks_list:
                min_x = min(min_x, landmark.x)
                max_x = max(max_x, landmark.x)
                min_y = min(min_y, landmark.y)
                max_y = max(max_y, landmark.y)

            # Denormalize and create bounding box (x, y, width, height)
            # Ensure coordinates are within image bounds after denormalization if needed,
            # though min/max logic on normalized coords should handle this.
            origin_x = int(min_x * img_w)
            origin_y = int(min_y * img_h)
            width = int((max_x - min_x) * img_w)
            height = int((max_y - min_y) * img_h)
            current_boxes.append((origin_x, origin_y, width, height))

    latest_bounding_boxes = current_boxes


def result_callback(
    result: mp.tasks.vision.FaceLandmarkerResult,  # type:ignore
    output_image: mp.Image,
    timestamp_ms: int,
):




    
    landmarks_to_rectangle(result, output_image)
    
    
    
    
    # ic(result)
    for face in result.face_blendshapes:
        # print(timestamp_ms)
        feature_vector = np.array(
            [blendshape_category.score for blendshape_category in face]
        )

        prediction = classifier.predict(feature_vector)
        print(prediction)


options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=mp_model_path),
    running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
    result_callback=result_callback,
    output_face_blendshapes=True,
    num_faces=max_faces,
)

classifier = ExpressionClassifier(classifier_model_path, categories_path)


with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
    cam = cv.VideoCapture(0)
    start_time = time.time()

    while cam.isOpened():
        success, frame = cam.read()
        if not success:
            print("Camera frame unavailable")
            continue

        current_timestamp_ms = int((time.time() - start_time) * 1000)

        image_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        landmarker.detect_async(image_mp, current_timestamp_ms)





        # Draw rectangles on the frame for each detected face
        # Use a copy of the list to avoid issues if it's updated by the callback during iteration
        bboxes_to_draw = list(latest_bounding_boxes)
        for x, y, w, h in bboxes_to_draw:
            # Ensure coordinates are integers for cv.rectangle
            start_point = (int(x), int(y))
            end_point = (int(x + w), int(y + h))
            cv.rectangle(frame, start_point, end_point, (0, 255, 0), 2)  # Green rectangle, thickness 2





        cv.imshow("camera", frame)
        if cv.waitKey(1) & 0xFF == ord("q"):  # Add a way to exit the loop
            break
