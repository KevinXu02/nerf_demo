import numpy as np


def data_loader():
    data = np.load(f"lego_200x200.npz")
    # Training images: [100, 200, 200, 3]
    images_train = data["images_train"] / 255.0
    # Cameras for the training images
    # (camera-to-world transformation matrix): [100, 4, 4]
    c2ws_train = data["c2ws_train"]
    # Validation images:
    images_val = data["images_val"] / 255.0
    # Cameras for the validation images: [10, 4, 4]
    # (camera-to-world transformation matrix): [10, 200, 200, 3]
    c2ws_val = data["c2ws_val"]
    # Test cameras for novel-view video rendering:
    # (camera-to-world transformation matrix): [60, 4, 4]
    c2ws_test = data["c2ws_test"]
    # Camera focal length
    focal = data["focal"]  # float
    return images_train, c2ws_train, images_val, c2ws_val, c2ws_test, focal
