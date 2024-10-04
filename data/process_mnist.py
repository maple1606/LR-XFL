import json
import torch
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
import os

def load_sample_indices(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data['samples']

def filter_mnist_by_samples(dataset, sample_indices):
    filtered_images = []
    filtered_labels = []
    
    for i in sample_indices:
        image, label = dataset[i]  
        flattened_image = image.view(-1).numpy()  
        filtered_images.append(flattened_image)
        filtered_labels.append(label)

    data_with_labels = np.column_stack((filtered_labels, filtered_images))  
    return pd.DataFrame(data_with_labels)

def download_and_label_mnist(json_file, root="MNIST_EVEN_ODD", save_dir="../experiments/data/MNIST_C_to_Y"):
    transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])

    dataset = datasets.ImageFolder(root=root, transform=transform)
    
    sample_indices = load_sample_indices(json_file)

    filtered_df = filter_mnist_by_samples(dataset, sample_indices)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filtered_df.to_csv(f"{save_dir}/mnist_train.csv", index=False, header=False)
    print(f"Filtered training data saved as {save_dir}/mnist_train.csv")

if __name__ == "__main__":
    download_and_label_mnist("train_samples_mnist.json")
