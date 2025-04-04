import os
import shutil
import cv2
import numpy as np

original_dataset = "dataset"
adv_dataset = "adversarial_dataset"

def generate_adversarial_image(image):
    noise = np.random.normal(0, 10, image.shape).astype(np.uint8) 
    adv_image = cv2.add(image, noise)
    return adv_image

for split in ["train", "test"]:
    for category in ["REAL", "FAKE"]:
        os.makedirs(os.path.join(adv_dataset, split, category), exist_ok=True)

for split in ["train", "test"]:
    for category in ["REAL", "FAKE"]:
        input_folder = os.path.join(original_dataset, split, category)
        output_folder = os.path.join(adv_dataset, split, category)
        
        for filename in os.listdir(input_folder):
            if filename.endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(input_folder, filename)
                image = cv2.imread(image_path)

                if image is not None:
                    adv_image = generate_adversarial_image(image)
                    adv_filename = f"{os.path.splitext(filename)[0]}_adv{os.path.splitext(filename)[1]}"
                    adv_image_path = os.path.join(output_folder, adv_filename)
                    cv2.imwrite(adv_image_path, adv_image)
