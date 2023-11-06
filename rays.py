import torch
from torch.utils.data import Dataset
import numpy as np

# from utils import pixel_to_ray
from utils import get_intrinsic_matrix
from utils import pixel_to_camera
from utils import transform
from utils import sample_along_rays

# from utils import pixel_to_ray


class RaysData(Dataset):
    def __init__(self, images_train, K, c2ws_train, focal):
        self.images_train = images_train  # N*W*H*3
        self.K = K
        self.c2ws_train = c2ws_train  # N*4*4
        self.batch_size = images_train.shape[0]
        self.W = images_train.shape[1]
        self.H = images_train.shape[2]
        self.focal = focal

    def sample_rays(self, N):
        # sample rays from image
        # N the number of rays
        # pixels is the color of the pixels
        # return rays_o: Nx3 matrix
        # return rays_d: Nx3 matrix
        # return pixels: Nx3 matrix
        n_samples = N // self.batch_size
        rays_o = np.zeros((N, 3))
        rays_d = np.zeros((N, 3))
        pixels = np.zeros((N, 3))
        for i in range(self.batch_size):
            c2w = self.c2ws_train[i]
            image = self.images_train[i]
            # sample n_samples points in each image without perturb
            xy = np.random.uniform(0, 1, size=(n_samples, 2))
            xy[:, 0] = xy[:, 0] * self.W
            xy[:, 1] = xy[:, 1] * self.H
            xy = xy.astype(np.int32)
            for j in range(n_samples):
                dirs = np.array(
                    [
                        (xy[j][0] - self.W / 2),
                        (xy[j][1] - self.H / 2),
                        self.focal,
                    ]
                )
                ray_o = c2w[:3, 3]
                ray_d = c2w[:3, :3] @ dirs
                ray_d = ray_d / np.linalg.norm(ray_d)
                rays_o[i * n_samples + j] = ray_o
                rays_d[i * n_samples + j] = ray_d
                pixels[i * n_samples + j] = image[xy[j][0], xy[j][1]]
        return rays_o, rays_d, pixels
