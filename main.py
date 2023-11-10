import viser, time  # pip install viser
import numpy as np
from data import data_loader
from rays import RaysData
from model import NeRF, ReplicateNeRFModel
import torch
import torch.nn as nn
import tqdm
from tqdm import tqdm
from PIL import Image
from utils import *
from train_utils import *


if __name__ == "__main__":
    # --- You Need to Implement These ------
    images_train, c2ws_train, images_val, c2ws_val, c2ws_test, focal = data_loader()
    # K = get_intrinsic_matrix(focal, images_train.shape[1], images_train.shape[2])
    # dataset = RaysData(images_train, K, c2ws_train, focal)
    # H, W = images_train.shape[1:3]
    # rays_o, rays_d, target = dataset.sample_rays(500)
    # t_vals = sample_t_vals(near=0, far=6.0, n_samples=64, perturb=False)
    # points, step_size = sample_along_rays(rays_o, rays_d, t_vals)
    # # transform all the data to np
    # rays_o = rays_o.cpu().numpy()
    # rays_d = rays_d.cpu().numpy()
    # points = points.cpu().numpy()
    # target = target.cpu().numpy()
    # print("rays_o.shape", rays_o.shape)
    # print("rays_d.shape", rays_d.shape)
    # print("points.shape", points.shape)
    # print("target.shape", target.shape)
    # # match the shape of target with points
    # target = target[:, None, :].repeat(points.shape[1], axis=1)
    # # --------------------------------------
    # viser_visualize(
    #     images_train[:10],
    #     c2ws_train[:10],
    #     rays_o[:50],
    #     rays_d[:50],
    #     points[:50],
    #     target[:50],
    #     200,
    #     200,
    #     get_intrinsic_matrix(focal, 200, 200),
    # )
    # model = train(
    #     n_samples=64,
    #     rays_num=5000,
    #     start_iter=3000,
    #     num_iterations=10000,
    #     log_freq=500,
    #     save_dir="./saved_models/",
    #     img_dir="./img/",
    #     trian_existing_model=True,
    #     learning_rate=5e-5,
    # )
    model = train_coarse_to_fine()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeRF().to(device)
    model.load_state_dict(torch.load("./saved_models/model_iter_4000.pt"))

    # img = render_img(model, c2ws_train[0], 200, 200, K)
    # render_gif(model, c2ws_test, 200, 200, K)
    # img.save("./img_val_rendered.png")
    # K = get_intrinsic_matrix(focal, 10, 10)
    # rays_o, rays_d = get_rays_full_image(10, 10, K[0, 0], c2ws_val[1])
    # print("rays_o.shape", rays_o.shape)
    # print("rays_d.shape", rays_d.shape)
    # # Sample points along each ray
    # points, step_size = sample_along_rays_np(
    #     rays_o, rays_d, perturb=False, n_samples=10
    # )
    # viser_visualize(
    #     images_val[1].reshape(1, 200, 200, 3),
    #     c2ws_val[1].reshape(1, 4, 4),
    #     rays_o,
    #     rays_d,
    #     points,
    #     10,
    #     10,
    #     get_intrinsic_matrix(focal, 100, 100),
    # )

    img = images_train[0]
    img = img * 255
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    img.save("./img_val_0.png")
