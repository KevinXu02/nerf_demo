import numpy as np
import torch
import viser
import time
import os
from PIL import Image


def transform(c2w, x_c):
    # support batch transform
    # c2w: 4x4 matrix
    # x_c: 3x1 vector or Nx3 matrix
    # if x_c is a vector, return a 3x1 vector
    # if x_c is a matrix, return a Nx3 matrix
    if len(x_c.shape) == 1:
        x_c = np.append(x_c, 1)
        x_w = np.dot(c2w, x_c)
        return x_w[:3]
    else:
        N = x_c.shape[0]
        x_c = np.hstack((x_c, np.ones((N, 1))))
        x_w = np.dot(x_c, c2w.T)
        return x_w[:, :3]


def pixel_to_camera(K, uv, s):
    # uv: 2x1 vector or Nx2 matrix
    # if uv is a vector, return a 3x1 vector
    # if uv is a matrix, return a Nx3 matrix
    K_inv = np.linalg.inv(K)
    if len(uv.shape) == 1:
        uv = np.append(uv, 1)
        x_c = np.dot(K_inv, uv) * s
        return x_c[:3]
    else:
        N = uv.shape[0]
        uv = np.hstack((uv, np.ones((N, 1))))
        x_c = np.dot(uv, K_inv.T) * s
        return x_c[:, :3]


def get_intrinsic_matrix(focal, H, W):
    #  return the intrinsic matrix
    #  focal: float
    #  H: int
    #  W: int
    return np.array([[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1]])


def pixel_to_ray(K, c2w, xy):
    #  return the ray direction
    #  K: 3x3 matrix
    #  c2w: 4x4 matrix
    #  uv: 2x1 vector
    #  return: 3x1 vector ray_o,ray_d
    # check uv is between 0 and 1
    uv = uv - 0.5
    s = 1.0
    K_inv = np.linalg.inv(K)
    ray_o = transform(c2w, np.zeros(3))
    ray_d = np.dot(K_inv, np.append(uv, 1))
    ray_d = np.dot(c2w[:3, :3], ray_d)
    ray_d = ray_d / np.linalg.norm(ray_d)
    return ray_o, ray_d


def sample_along_rays(rays_o, rays_d, near=2.0, far=6.0, n_samples=64, perturb=False):
    #  sample points along rays
    #  rays_o: Nx3 matrix
    #  rays_d: Nx3 matrix
    #  perturb: bool
    #  return sampled points: Nxn_samplesx3 matrix
    #  return step_size: Nxn_samplesx1 matrix
    N = rays_d.shape[0]
    t = np.linspace(near, far, n_samples)
    t_width = t[1] - t[0]
    if perturb:
        t = t + np.random.random(n_samples) * t_width
    # step_size = t_(i+1) - t_i
    step_size = np.zeros((N, n_samples))
    step_size[:, 1:] = t[1:] - t[:-1]
    for i in range(N):
        points = rays_o[i] + np.outer(t, rays_d[i])
        if i == 0:
            points_all = points
        else:
            points_all = np.vstack((points_all, points))

    return points_all.reshape(N, n_samples, 3), step_size.reshape(N, n_samples, 1)


def volrend(sigmas, rgbs, step_size, white_bkgd=False):
    # sigmas: Nxn_samplesx1 tensor
    # rgbs: Nxn_samplesx3 tensor
    # step_size: Nxn_samplesx1 tensor
    # return: N x 3 tensor
    if white_bkgd:
        sigmas = torch.cat([sigmas, torch.ones_like(sigmas[:, :1])], dim=-1)
        rgbs = torch.cat([rgbs, torch.zeros_like(rgbs[:, :1])], dim=-2)
    alpha = 1.0 - torch.exp(-sigmas * step_size)
    alpha = alpha.squeeze()
    T_i = torch.cat([torch.ones_like(alpha[:, :1]), 1.0 - alpha], dim=-1)[:, :-1]
    T_i = torch.cumprod(T_i, dim=-1)
    weights = alpha * T_i
    colors = torch.sum(weights.unsqueeze(-1) * rgbs, dim=-2)
    return colors


def mse2psnr(mse):
    return -10.0 * torch.log(mse) / torch.log(torch.Tensor([10.0])).to(mse.device)


def get_rays_full_image(W, H, focal, c2w):
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing="xy"
    )
    rays_o = np.zeros((W * H, 3))
    rays_d = np.zeros((W * H, 3))
    


def viser_visualize(images_train, c2ws_train, rays_o, rays_d, points, H, W, K):
    server = viser.ViserServer(share=True)
    for i, (image, c2w) in enumerate(zip(images_train, c2ws_train)):
        server.add_camera_frustum(
            f"/cameras/{i}",
            fov=2 * np.arctan2(H / 2, K[0, 0]),
            aspect=W / H,
            scale=0.15,
            wxyz=viser.transforms.SO3.from_matrix(c2w[:3, :3]).wxyz,
            position=c2w[:3, 3],
            image=image,
        )
    for i, (o, d) in enumerate(zip(rays_o, rays_d)):
        server.add_spline_catmull_rom(
            f"/rays/{i}",
            positions=np.stack((o, o + d * 6.0)),
        )
        # visualize rays_o
        server.add_point_cloud(
            f"/rays_o/{i}",
            colors=np.zeros_like(o).reshape(-1, 3),
            points=o.reshape(-1, 3),
            point_size=0.02,
        )
    server.add_point_cloud(
        f"/samples",
        colors=np.zeros_like(points).reshape(-1, 3),
        points=points.reshape(-1, 3),
        point_size=0.02,
    )
    time.sleep(1000)


def pos_encoding(x, L):
    # apply a serious of sinusoidal functions to the input cooridnates, to expand its dimensionality
    # pe(x)={x,sin(πx),cos(πx),sin(2^1πx),cos(2^1πx),...,sin(2^(L-1)πx),cos(2^(l-1)πx)}
    # x: [N, 3]
    # L: int
    # return: [N, 6* L]
    x = x.unsqueeze(-1)
    l = torch.arange(L, dtype=torch.float32, device=x.device)
    l = 2**l
    x = x * l * torch.pi
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    # change the type of x to float32
    x = x.type(torch.float32)
    return x.flatten(-2)


def render_img(nerf_model, c2w, W, H, K):
    # Compute ray origins and directions for each pixel in the image
    rays_o, rays_d = get_rays_full_image(W, H, K[0, 0], c2w)

    # Sample points along each ray
    points, step_size = sample_along_rays(rays_o, rays_d)
    points = torch.tensor(points)
    rays_d = torch.tensor(np.tile(rays_d[:, np.newaxis, :], (1, 64, 1)))
    x = torch.cat([points, rays_d], dim=-1)
    # Predict color and opacity of each sampled point using the NeRF model
    sigmas, rgbs = nerf_model(x)

    # Combine colors and opacities into a single image
    img = volrend(sigmas, rgbs, step_size)
    img = img * 255
    # save the image to local
    img = img.cpu().numpy().astype(np.uint8)
    img = Image.fromarray(img)
    img.save("test.png")

    return img
