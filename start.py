import cv2

from beauty_prediction_model import BeautyPredictionModel

model = BeautyPredictionModel("model.h5")
image_path = "test_image.jpg"

img = cv2.imread(image_path)
score = model.make_prediction(img)

print(f"score: {score:.2f}")
