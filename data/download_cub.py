#!/usr/bin/env python
import os
import json
import numpy as np
import pandas as pd


def download_cub(root="CUB_200_2011", force=True):

    if os.path.isdir(root) and not force:
        print("Dataset already downloaded")
        return

    """   
    os.system("git clone https://github.com/chentinghao/download_google_drive.git")
    from download_google_drive.download_gdrive import download_file_from_google_drive
    download_file_from_google_drive("1hbzc_P1FuxMkcabkgn9ZKinBwW683j45", "CUB_200_2011.tgz")
    print("Dataset downloaded")
    """

    os.system("tar -zxvf CUB_200_2011.tgz")
    print("\nDataset extracted")

    origin_dir = os.curdir
    os.chdir(root)

    os.system("mv ../attributes.txt .")
    os.rename("attributes.txt", "attributes_names.txt")

    with open("attributes_names.txt", "r") as f:
        attribute_names = []
        lines = f.readlines()
        for line in lines:
            attr_name = line.split(" ")[1]
            attribute_names.append(attr_name)

    attribute_per_class = pd.read_csv(os.path.join("attributes", "class_attribute_labels_continuous.txt"), 
                                  sep=" ", 
                                  header=None).to_numpy()

    print("Attribute per class loaded")

    classes = pd.read_csv("image_class_labels.txt", sep=" ", header=None).to_numpy()[:, 1]
    print("Image_classes loaded")

    tot_annotations, count = 0, 0
    attributes = np.zeros(shape=(11788, 312))
    # denoised_attributes = np.zeros(shape=(11788, 312))
    with open(os.path.join("attributes", "image_attribute_labels.txt"), "r") as f:
        lines = f.readlines()
        for line in lines:
            fields = line.split(" ")
            if fields[2] == "1":
                img_idx = int(fields[0]) - 1
                attr_idx = int(fields[1]) - 1
                attributes[img_idx][attr_idx] = 1

    np.save("original_attributes.npy", attributes)
    print("Original attributes saved")

    very_denoised_attributes = np.zeros(shape=(11788, 312))
    attribute_sparsity = np.zeros(attributes.shape[1])
    for c in np.unique(classes):
        imgs = classes == c
        class_attributes = attribute_per_class[c - 1, :] > 50
        very_denoised_attributes[imgs, :] = class_attributes
        attribute_sparsity += class_attributes
    attributes_to_filter = attribute_sparsity < 10
    very_denoised_attributes = very_denoised_attributes[:, ~attributes_to_filter]
    attributes_filtered = np.sum(attributes_to_filter)
    print("Number of attributes remained", attributes_filtered)
    np.save("attributes.npy", very_denoised_attributes)
    print("Denoised attributes Saved")

    with open("attributes_names.txt", "w") as f:
        json.dump(np.asarray(attribute_names)[~attributes_to_filter].tolist(), f)
    print("Attribute names saved")

    for item in os.listdir():
        if item not in ["images", "images.txt", "attributes_names.txt", "attribute_groups.json",
                        "original_attributes.npy",  "attributes.npy"]:
            os.system(f"rm -r {item}")
    os.system("rm -rf ../download_google_drive")
    print("Temporary file cleaned")

    os.system("mv images/* .")
    with open("images.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            fields = line.split(" ")
            img_idx = f"{int(fields[0]):05d}"
            img_name = fields[1][:-1]
            img_name = os.path.join(os.path.dirname(img_name), os.path.basename(img_name))
            new_name = os.path.join(os.path.dirname(img_name), img_idx) + os.path.splitext(img_name)[1]
            os.rename(img_name, new_name)
    os.system("rm -r images")
    os.remove("images.txt")
    print("Images sorted and renamed")
    print("Dataset configured correctly")

    os.chdir(origin_dir)


if __name__ == "__main__":
    download_cub()
