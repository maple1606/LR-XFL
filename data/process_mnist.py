import os
import numpy as np
import pandas as pd
from PIL import Image

root = "MNIST_EVEN_ODD"
even_dir = os.path.join(root, "Even")
odd_dir = os.path.join(root, "Odd")
destination = "../experiments/data/MNIST_C_to_Y"
output_file = os.path.join(destination, "mnist.csv")

os.makedirs(destination, exist_ok=True)

def process_images_from_folder(folder_path, label):
    images_data = []
    for img_file in os.listdir(folder_path):
        if img_file.endswith('.jpg'):  
            img_path = os.path.join(folder_path, img_file)
            img = Image.open(img_path).convert('L') 
            img_array = np.array(img).flatten()  
            img_array = np.append(img_array, label)  
            images_data.append(img_array)
    return images_data

even_images = process_images_from_folder(even_dir, label=0)
odd_images = process_images_from_folder(odd_dir, label=1)

print(f"Number of even images: {len(even_images)}")
print(f"Number of odd images: {len(odd_images)}")

all_images = np.vstack([even_images, odd_images])

columns = [f'pixel_{i}' for i in range(28*28)] + ['label'] 
mnist_df = pd.DataFrame(all_images, columns=columns)

mnist_df.to_csv(output_file, index=False, header=False)

print(f"MNIST Even/Odd dataset saved to {output_file}")
