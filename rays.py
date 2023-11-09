import torch
from torch.utils.data import Dataset
import numpy as np

# from utils import pixel_to_ray
from utils import get_intrinsic_matrix
from utils import pixel_to_camera
from utils import transform
from utils import sample_along_rays
from utils import sample_along_rays_np, get_rays_full_image, get_rays_full_image_torch

# from utils import pixel_to_ray


class RaysData(Dataset):
    def __init__(self, images_train, K, c2ws_train, focal):
        self.images_train = images_train  # N*H*W*3
        self.K = K
        self.c2ws_train = c2ws_train  # N*4*4
        self.num_img = images_train.shape[0]
        self.H = images_train.shape[1]
        self.W = images_train.shape[2]
        self.focal = focal
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def sample_rays(self, N):
        # sample rays from image
        # N the number of rays
        # pixels is the color of the pixels, Nx3
        # rays_o is the origin of the rays Nx3
        # rays_d is the direction of the rays Nx3
        n_samples_per_img = N // self.num_img
        rays_o = []
        rays_d = []
        pixels = []
        c2ws = torch.from_numpy(self.c2ws_train).float()
        images = torch.from_numpy(self.images_train)
        for c2w, image in zip(c2ws, images):
            rays_o_c, rays_d_c = get_rays_full_image_torch(
                self.H, self.W, self.focal, c2w
            )
            idx = np.random.choice(self.H * self.W, n_samples_per_img, replace=False)
            rays_o.append(rays_o_c[idx])
            rays_d.append(rays_d_c[idx])
            pixels.append(image.view(-1, 3)[idx])
        rays_o = torch.cat(rays_o, dim=0).to(self.device)
        rays_d = torch.cat(rays_d, dim=0).to(self.device)
        pixels = torch.cat(pixels, dim=0).to(self.device)

        return (rays_o, rays_d, pixels)
