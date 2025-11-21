import cv2 as cv
import mediapipe as mp
import numpy as np
from pathlib import Path
import csv

# Mediapipe model
mp_model_path = "models/face_landmarker.task"
# dataset_path = "datasets/CK+"
dataset_path = r"C:\Users\effax\Downloads\fer2013+"

blendshapes_save_path = "blendshapes/blendshapes.csv"
categories_save_path = "blendshapes/categories.csv"
# Maximum number of faces in a single image of the dataset
max_faces = 1


def detect_image_blendshapes(
    image_path: str | Path,
) -> mp.tasks.vision.FaceLandmarkerResult:  # type:ignore
    image_np = cv.imread(image_path)
    assert image_np is not None, f"Cannot read image: {image_path}"

    # Check and handle grayscale image
    if len(image_np.shape) == 2 or image_np.shape[2] == 1:
        image_np = cv.cvtColor(image_np, cv.COLOR_GRAY2RGB)

    image_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)

    face_landmarker_result = landmarker.detect(image_mp)

    result = []

    for face in face_landmarker_result.face_blendshapes:
        # Extract blendshape scores
        feature_vector = np.array(
            [blendshape_category.score for blendshape_category in face]
        )
        result.append(feature_vector)

    return result


dataset = Path(dataset_path)
# Ensure output directory exists
Path(blendshapes_save_path).parent.mkdir(parents=True, exist_ok=True)
Path(categories_save_path).parent.mkdir(parents=True, exist_ok=True)

# Write the category names to categories.csv
with open(categories_save_path, "w", newline="") as categories_csvfile:
    categories_csv_writer = csv.writer(categories_csvfile)
    categories = [item.name for item in sorted(dataset.iterdir()) if item.is_dir()]
    categories_csv_writer.writerow(categories)


# Open the CSV file before the loops start
with open(blendshapes_save_path, "w", newline="") as blendshapes_csvfile:
    blendshapes_csv_writer = csv.writer(blendshapes_csvfile)

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=mp_model_path),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        output_face_blendshapes=True,
        num_faces=max_faces,
    )

    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        # Iterate through the dataset
        for index, category in enumerate(sorted(dataset.iterdir())):
            if not category.is_dir():  # Skip files not in a category
                continue
            print("Collecting", index, category.name)
            for image_path in category.iterdir():
                result = detect_image_blendshapes(image_path)

                for face in result:
                    row = [index] + face.tolist()
                    blendshapes_csv_writer.writerow(row)

                # quit()

print("Done")





import matplotlib.pyplot as plt
from pathlib import Path

dataset_path = "datasets/CK+"
plt.rcParams['font.family'] = 'SimHei'

dataset = Path(dataset_path)

categories = []
image_counts = []

for category_dir in sorted(dataset.iterdir()):
    if category_dir.is_dir():
        categories.append(category_dir.name)
        count = 0
        for _ in category_dir.iterdir():
            count += 1
        image_counts.append(count)


plt.figure(figsize=(12, 6))
plt.bar(categories, image_counts, color='green')

plt.xlabel("类别")
plt.ylabel("图片数量")
plt.title("不同类别下的图片数量")
plt.tight_layout() 

plt.show()

