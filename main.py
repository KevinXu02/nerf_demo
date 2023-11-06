import viser, time  # pip install viser
import numpy as np
from data import data_loader
from utils import sample_along_rays
from utils import get_intrinsic_matrix
from rays import RaysData
from model import NeRF, ReplicateNeRFModel
from utils import volrend
from utils import mse2psnr, viser_visualize, pos_encoding, render_img
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm
from tqdm import tqdm


def render_rays(points, rays_d, step_size):
    model = NeRF()
    density, color = model(points, rays_d)
    colours = volrend(density, color, step_size)
    return colours


def train(
    num_iters=1000,
    rays_num=1000,
    n_samples=64,
    save_dir="./saved_models/",
    log_freq=200,
    img_dir=None,
    exp_name=None,
    exp_dir=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images_train, c2ws_train, _, _, _, focal = data_loader()
    K = get_intrinsic_matrix(focal, images_train.shape[1], images_train.shape[2])
    dataset = RaysData(images_train, K, c2ws_train, focal)

    model = NeRF().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
    )
    loss_fn = nn.MSELoss()

    for epoch in tqdm(range(num_iters)):
        optimizer.zero_grad()
        rays_o, rays_d, pixels = dataset.sample_rays(rays_num)

        points, step_size = sample_along_rays(
            rays_o, rays_d, perturb=True, n_samples=n_samples
        )
        # print(points.shape, rays_d.shape, step_size.shape)
        points_train = torch.tensor(points).to(device)
        rays_d_train = torch.tensor(
            np.tile(rays_d[:, np.newaxis, :], (1, n_samples, 1))
        ).to(device)
        points_train = pos_encoding(points_train, 10)
        rays_d_train = pos_encoding(rays_d_train, 4)
        alpha, rgb = model(torch.cat([points_train, rays_d_train], dim=-1))
        colours = volrend(alpha, rgb, torch.tensor(step_size).to(device))
        # print("colours", pixels)
        loss = loss_fn(colours, torch.tensor(pixels).to(device))
        psnr = mse2psnr(loss)
        loss.backward()
        optimizer.step()
        tqdm.write(f"Epoch: {epoch+1}, Loss: {loss.item()}, PSNR: {psnr.item()}")

        if (epoch + 1) % log_freq == 0 and save_dir is not None:
            torch.save(model.state_dict(), f"{save_dir}/model1_epoch_{epoch+1}.pt")
            tqdm.write(f"Model saved at {save_dir}/model1_epoch_{epoch+1}.pt")

    return model


if __name__ == "__main__":
    # --- You Need to Implement These ------
    images_train, c2ws_train, images_val, c2ws_val, c2ws_test, focal = data_loader()
    K = get_intrinsic_matrix(focal, images_train.shape[1], images_train.shape[2])
    dataset = RaysData(images_train, K, c2ws_train, focal)
    rays_o, rays_d, pixels = dataset.sample_rays(100)
    points, step_size = sample_along_rays(rays_o, rays_d, perturb=False)
    H, W = images_train.shape[1:3]
    # ---------------------------------------
    viser_visualize(images_train, c2ws_train, rays_o, rays_d, points, H, W, K)
    # model = train(num_iters=5000, n_samples=64, rays_num=3000)
    # load model form saved_models
    # model = NeRF()
    # model.load_state_dict(torch.load("./saved_models/model1_epoch_600.pt"))
    # render_img(model, c2ws_test[0], W, H, K)
