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


def train(
    num_iterations=5000,
    rays_num=5000,
    n_samples=64,
    save_dir="./saved_models/",
    log_freq=200,
    img_dir="./img/",
    trian_existing_model=True,
    start_iter=5000,
    learning_rate=1e-4,
    # model_path="./saved_models/model_iter_2000.pt",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images_train, c2ws_train, images_val, c2ws_val, c2ws_test, focal = data_loader()
    K = get_intrinsic_matrix(focal, images_train.shape[1], images_train.shape[2])
    dataset = RaysData(images_train, K, c2ws_train, focal)
    model_path = "./saved_models/model_iter_{a}.pt".format(a=start_iter)
    model = NeRF().to(device)

    if trian_existing_model:
        try:
            model.load_state_dict(torch.load(model_path))
            tqdm.write(f"Model loaded from {model_path}")
        except:
            tqdm.write(f"Model not found at {model_path}")
    else:
        tqdm.write(f"Training new model")
        start_iter = 0

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )
    # lr_decay = 500
    # decay_rate = 0.1
    # decay_steps = lr_decay * 1000
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    loss_fn = nn.MSELoss()
    avg_loss = 0
    for iter in tqdm(range(start_iter, start_iter + num_iterations)):
        rays_o, rays_d, target = dataset.sample_rays(rays_num)
        t_vals = sample_t_vals(near=2.0, far=6.0, n_samples=n_samples, perturb=True)
        points, step_size = sample_along_rays(rays_o, rays_d, t_vals)

        points_train = pos_encoding(points, 10).to(device)
        rays_d_train = pos_encoding(rays_d[..., None, :].expand(points.shape), 4).to(
            device
        )
        x = torch.cat([points_train, rays_d_train], dim=-1)
        # print("x.shape", x.shape)

        alpha, rgb = model(x)
        # print("alpha.shape", alpha.shape)
        # print("rgb.shape", rgb.shape)
        colours, _ = volrend(alpha, rgb, step_size.to(device))

        optimizer.zero_grad()
        loss = loss_fn(colours.to(device), target.float().to(device))

        loss.backward()
        optimizer.step()
        scheduler.step()
        avg_loss += loss.item()

        # new_lr = learning_rate * (decay_rate ** (iter / decay_steps))
        # for param_group in optimizer.param_groups:
        #     param_group["lr"] = new_lr

        if (iter + 1) % 20 == 0:
            avg_loss /= 20
            tqdm.write(
                f"iter: {iter+1}, avg_Loss: {avg_loss}, PSNR: {mse2psnr(avg_loss)}"
            )
            avg_loss = 0
        if (iter + 1) % log_freq == 0 and save_dir is not None:
            # img = render_img(model, c2ws_train[0], H, W, K)
            # img.save(f"{img_dir}/{epoch+1}.png")
            torch.save(model.state_dict(), f"{save_dir}/model_iter_{iter+1}.pt")
            tqdm.write(f"Model saved at {save_dir}/model_iter_{iter+1}.pt")
            img = render_img(model, c2ws_val[0], 200, 200, K)
            img.save(f"{img_dir}/{iter+1}.png")

    return model


def train_coarse_to_fine(
    num_iterations=5000,
    start_iter=0,
    n_samples=64,
    ray_num=2000,
    log_freq=500,
    trian_existing_model=True,
    img_dir="./img/",
    model_path="./saved_models/",
    learning_rate=2e-4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images_train, c2ws_train, images_val, c2ws_val, c2ws_test, focal = data_loader()
    K = get_intrinsic_matrix(focal, images_train.shape[1], images_train.shape[2])
    dataset = RaysData(images_train, K, c2ws_train, focal)
    coarse_model = NeRF().to(device)
    fine_model = NeRF().to(device)
    if trian_existing_model:
        try:
            coarse_model.load_state_dict(torch.load(f"saved_models\model_iter_5500.pt"))
            fine_model.load_state_dict(torch.load(f"saved_models\model_iter_5500.pt"))
            tqdm.write(f"Model loaded from {model_path}")
        except:
            tqdm.write(f"Model not found at {model_path}")
    else:
        tqdm.write(f"Training new model")
        start_iter = 0

    parms = list(coarse_model.parameters()) + list(fine_model.parameters())
    optimizer = torch.optim.Adam(
        parms,
        lr=learning_rate,
        betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    loss_fn = nn.MSELoss()
    avg_loss = {fine_model: 0, coarse_model: 0}
    for iter in tqdm(range(start_iter, start_iter + num_iterations)):
        rays_o, rays_d, target = dataset.sample_rays(ray_num)
        t_vals = sample_t_vals(
            near=2.0, far=6.0, n_samples=n_samples, perturb=True, batch_size=ray_num
        ).to(device)
        points, step_size = sample_along_rays(rays_o, rays_d, t_vals)
        points_train = pos_encoding(points, 10).to(device)
        rays_d_train = pos_encoding(rays_d[..., None, :].expand(points.shape), 4).to(
            device
        )

        x = torch.cat([points_train, rays_d_train], dim=-1)
        alpha, rgb = coarse_model(x)
        colours, weights = volrend(alpha, rgb, step_size.to(device))

        t_mid = (t_vals[..., 1:] + t_vals[..., :-1]) / 2
        t_fine = sample_pdf(
            t_mid, weights[..., 1:-1], n_sample=n_samples, perturb=True, device=device
        ).detach()
        t_fine = t_fine.to(t_vals)

        # print("t_fine.shape", t_fine.shape)
        # print("t_vals.shape", t_vals.shape)
        # print("''''''''''''''''''''''''''")
        t_fine = torch.sort(torch.cat([t_vals, t_fine], dim=-1), dim=-1)[0]
        points_fine, step_size_fine = sample_along_rays(rays_o, rays_d, t_fine)

        points_fine_train = pos_encoding(points_fine, 10).to(device)
        rays_d_fine_train = pos_encoding(
            rays_d[..., None, :].expand(points_fine.shape), 4
        ).to(device)
        x_fine = torch.cat([points_fine_train, rays_d_fine_train], dim=-1)
        alpha_fine, rgb_fine = fine_model(x_fine)
        colours_fine, _ = volrend(alpha_fine, rgb_fine, step_size_fine.to(device))

        optimizer.zero_grad()
        coarse_loss = loss_fn(colours.to(device), target.float().to(device))
        avg_loss[coarse_model] += coarse_loss.item()
        fine_loss = loss_fn(colours_fine.to(device), target.float().to(device))
        avg_loss[fine_model] += fine_loss.item()
        loss = coarse_loss + fine_loss

        loss.backward()
        optimizer.step()
        scheduler.step()

        if (iter + 1) % 20 == 0:
            avg_loss[coarse_model] /= 20
            avg_loss[fine_model] /= 20
            tqdm.write(
                f"iter: {iter+1}, coarse_avg_Loss: {avg_loss[coarse_model]}, coarse_PSNR: {mse2psnr(avg_loss[coarse_model])}, fine_avg_Loss: {avg_loss[fine_model]}, fine_PSNR: {mse2psnr(avg_loss[fine_model])}"
            )
            avg_loss[coarse_model] = 0
            avg_loss[fine_model] = 0

        if (iter + 1) % log_freq == 0:
            torch.save(
                coarse_model.state_dict(), f"./saved_models/coarse_model_{iter+1}.pt"
            )
            torch.save(
                fine_model.state_dict(), f"./saved_models/fine_model_{iter+1}.pt"
            )
            tqdm.write(f"Model saved at ./saved_models/coarse_model.pt")
            tqdm.write(f"Model saved at ./saved_models/fine_model.pt")
            coarse_img, fine_img = render_img(
                coarse_model,
                fine_model=fine_model,
                c2w=c2ws_val[0],
                K=K,
                H=200,
                W=200,
            )
            coarse_img.save(f"{img_dir}/coarse_{iter+1}.png")
            fine_img.save(f"{img_dir}/fine_{iter+1}.png")
