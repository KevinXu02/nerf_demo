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


def sample_along_rays_np(
    rays_o, rays_d, near=2.0, far=6.0, n_samples=64, perturb=False
):
    #  sample points along rays
    #  rays_o: Nx3 matrix
    #  rays_d: Nx3 matrix
    #  perturb: bool
    #  return sampled points: Nxn_samplesx3 matrix
    #  return step_size: Nxn_samplesx1 matrix
    # handle case where rays_o and rays_d are vectors
    N = rays_d.shape[0]
    t = np.linspace(near, far, n_samples)
    t_width = t[1] - t[0]
    if perturb:
        t = t + np.random.random(n_samples) * t_width
    # step_size = t_(i+1) - t_i
    step_size = np.zeros((N, n_samples))
    step_size = t - np.concatenate([np.zeros_like(t[:1]), t[:-1]])
    # print("step_size.shape", step_size.shape)
    for i in range(N):
        points = rays_o[i] + np.outer(t, rays_d[i])
        if i == 0:
            points_all = points
        else:
            points_all = np.vstack((points_all, points))

    return points_all.reshape(N, n_samples, 3), step_size.reshape(N, n_samples, 1)


def sample_along_rays(rays_o, rays_d, t_vals):
    #  sample points along rays
    #  rays_o: Nx3 matrix
    #  rays_d: Nx3 matrix
    #  t_vals: n_samples tensor
    #  return sampled points: Nxn_samplesx3 matrix
    #  return step_size: Nxn_samplesx1 matrix
    # print("rays_o.shape", rays_o.shape)
    # print("rays_d.shape", rays_d.shape)
    # print("rays_d.shape", rays_d.shape)
    t = t_vals.to(rays_o)
    # step_size = t_(i+1) - t_i
    # print("t.shape", t.shape)
    step_size = torch.cat([(t[:, 1:] - t[:, :-1]), torch.zeros_like(t[:, :1])], dim=-1)
    # print("step_size.shape", step_size.shape)
    points_all = rays_o[..., None, :] + t[..., :, None] * rays_d[..., None, :]

    return points_all, step_size


def sample_t_vals(near=2.0, far=6.0, n_samples=64, perturb=False, batch_size=1):
    #  sample points along rays
    #  rays_o: Nx3 matrix
    #  rays_d: Nx3 matrix
    #  perturb: bool
    #  return sampled points: Nxn_samplesx3 matrix
    #  return step_size: Nxn_samplesx1 matrix
    t = torch.linspace(near, far, n_samples).expand(batch_size, n_samples)
    t_width = (far - near) / n_samples
    if perturb:
        t = t + torch.rand(batch_size, n_samples) * t_width

    # print the fisrt 2 rows of t
    # print(t[0:2])
    return t


def sample_pdf(bins, weight, n_sample, device, perturb=True):
    # this part is borrowed from original nerf code
    """
    -input
    bins=(Batch,Nc-1)
    weight=(Batch,Nc-2)
    N_sample is Nf in Original paper

    -output
    (Batch,Nf)
    *Batch can be just (Batch size) or also (Batch size, *, H, W)
    """
    bins = bins.to(device)
    weight = weight + 1e-5  # prevent NAN
    pdf = weight / torch.sum(weight, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(weight[..., :1]), cdf], dim=-1)  # (Batch, Nc-1)
    NcMinusOne = cdf.shape[-1]

    if perturb:
        u = torch.rand(list(weight.shape[:-1]) + [n_sample])
    else:
        u = torch.linspace(0.0, 1.0, steps=n_sample)
        u = u.expand(list(weight.shape[:-1]) + [n_sample])

    u = u.contiguous()  # (Batch, Nf)
    u = u.to(device)
    idxs = torch.searchsorted(cdf, u, right=True)  # (Batch, Nf)
    below = torch.max(torch.zeros_like(idxs), idxs - 1)
    above = torch.min(torch.ones_like(idxs) * (NcMinusOne - 1), idxs)
    inds_g = torch.stack([below, above], dim=-1)  # (Batch, Nf, 2)

    matched_shape = list(inds_g.shape[:-1]) + [NcMinusOne]  # (Batch, Nf, Nc-1)

    cdf_g = torch.gather(cdf[..., None, :].expand(matched_shape), dim=-1, index=inds_g)
    bins_g = torch.gather(
        bins[..., None, :].expand(matched_shape), dim=-1, index=inds_g
    )

    denom = cdf_g[..., 1] - cdf_g[..., 0]  # (Batch, Nf)
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)

    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
    return samples


def volrend(sigmas, rgbs, step_size, white_bkgd=False):
    # sigmas: Nxn_samplesx1 tensor
    # rgbs: Nxn_samplesx3 tensor
    # step_size: Nxn_samples tensor
    # return: colors: Nx3 tensor
    # return: weights: Nxn_samples tensor
    sigmas = sigmas.squeeze()
    if white_bkgd:
        # injection of white background color
        rgbs = torch.cat([rgbs, torch.ones_like(rgbs[:, :1, :])], dim=1)
        sigmas = torch.cat([sigmas, 99999 * torch.ones_like(sigmas[:, :1])], dim=1)
        step_size = torch.cat([step_size, torch.ones_like(step_size[:, :1])], dim=1)
    alpha = 1.0 - torch.exp(-sigmas * step_size)
    alpha = alpha.squeeze()
    T_i = torch.cat([torch.ones_like(alpha[:, :1]), 1.0 - alpha], dim=-1)[:, :-1]
    T_i = torch.cumprod(T_i, dim=-1)
    weights = alpha * T_i
    colors = torch.sum(weights.unsqueeze(-1) * rgbs, dim=-2)
    if white_bkgd:
        weights = weights[:, :-1]
    return colors, weights


def mse2psnr(mse):
    return 20 * np.log10(1.0) - 10 * np.log10(mse)


def get_rays_full_image(H, W, focal, c2w):
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing="xy"
    )
    rays_o = c2w[:3, 3]
    rays_o = np.tile(rays_o, (H * W, 1))
    rays_d = np.stack(
        [(i - W * 0.5) / focal, (j - H * 0.5) / focal, np.ones_like(i)], axis=-1
    )
    rays_d = np.dot(c2w[:3, :3], rays_d.reshape(-1, 3).T).T
    rays_d = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)
    # print(rays_d.shape)
    # print(rays_o.shape)
    return rays_o, rays_d


def get_rays_full_image_torch(H, W, focal, c2w):
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32),
        torch.arange(H, dtype=torch.float32),
        indexing="xy",
    )
    if not isinstance(c2w, torch.Tensor):
        c2w = torch.from_numpy(c2w).float()

    dirs = torch.stack(
        [(i - W * 0.5) / focal, (j - H * 0.5) / focal, torch.ones_like(i)], dim=-1
    )
    rays_d = dirs @ (c2w[:3, :3].t())
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # print(rays_d.shape)
    # print(rays_o.shape)
    rays_o = c2w[:3, 3].expand(rays_d.shape)
    return rays_o.view(-1, 3), rays_d.view(-1, 3)


def viser_visualize(images_train, c2ws_train, rays_o, rays_d, points, targets, H, W, K):
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
        f"/samples",
        colors=targets.reshape(-1, 3),
        points=points.reshape(-1, 3),
        point_size=0.02,
    )
    time.sleep(1000)


# def pos_encoding(x, L, include_input=True):
#     # apply a serious of sinusoidal functions to the input cooridnates, to expand its dimensionality
#     # pe(x)={x,sin(πx),cos(πx),sin(2^1πx),cos(2^1πx),...,sin(2^(L-1)πx),cos(2^(l-1)πx)}
#     # x: [N, 3]
#     # L: int
#     # return: [N, 6* L]
#     x_0 = x.unsqueeze(-1)
#     x = x_0
#     l = torch.arange(L, dtype=torch.float32, device=x.device)
#     l = 2**l
#     x = x * l * torch.pi
#     x = torch.cat([x.sin(), x.cos()], dim=-1)
#     if include_input:
#         x = torch.cat([x_0, x], dim=-1)
#     # change the type of x to float32
#     return x.flatten(-2).float()


def pos_encoding(x, L, include_input=True):
    embed_fns = []
    encode_fn = [torch.sin, torch.cos]
    if include_input:
        embed_fns.append(lambda x: x)
    for res in range(L):
        res = 2**res
        for fn in encode_fn:
            embed_fns.append(lambda x, fn_=fn, res_=res: fn_(res_ * x))
    return torch.cat([fn(x) for fn in embed_fns], dim=-1)


@torch.no_grad()
def render_img(
    coarse_model,
    c2w,
    H,
    W,
    K,
    chunk=1024,
    fine_model=None,
    n_samples=64,
    white_bkgd=False,
):
    # Compute ray origins and directions for each pixel in the image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rays_o, rays_d = get_rays_full_image_torch(H, W, K[0, 0], c2w)
    t_vals = sample_t_vals(
        near=2.0, far=6.0, n_samples=64, perturb=False, batch_size=H * W
    )
    points, step_size = sample_along_rays(
        rays_o,
        rays_d,
        t_vals,
    )

    points_coarse = pos_encoding(points, 10).to(device)
    rays_d_coarse = pos_encoding(rays_d[..., None, :].expand(points.shape), 4).to(
        device
    )

    # Predict color and opacity of each sampled point using the NeRF model
    all_alpha = []
    all_rgb = []
    for i in range(0, points.shape[0], chunk):
        points_chunk = points_coarse[i : i + chunk]
        rays_d_chunk = rays_d_coarse[i : i + chunk]
        x_chunk = torch.cat([points_chunk, rays_d_chunk], dim=-1)
        alpha_chunk, rgb_chunk = coarse_model(x_chunk)
        all_alpha.append(alpha_chunk)
        all_rgb.append(rgb_chunk)
    alpha = torch.cat(all_alpha, dim=0)
    rgb = torch.cat(all_rgb, dim=0)
    # Combine colors and opacities into a single image
    coarse_img, weights = volrend(
        alpha, rgb, step_size.to(device), white_bkgd=white_bkgd
    )
    depth_img = torch.sum(weights * t_vals.to(weights), dim=-1)
    depth_img = (depth_img.reshape(H, W) * 40).detach().cpu().numpy().astype(np.uint8)
    depth_img = Image.fromarray(depth_img)

    coarse_img = (
        (coarse_img.reshape(H, W, 3) * 255).detach().cpu().numpy().astype(np.uint8)
    )
    coarse_img = Image.fromarray(coarse_img)

    # fine model
    if fine_model is None:
        return coarse_img, depth_img
    else:
        t_mid = (t_vals[..., 1:] + t_vals[..., :-1]) / 2
        t_fine = sample_pdf(
            t_mid, weights[..., 1:-1], n_sample=n_samples, perturb=True, device=device
        ).detach()
        t_fine = t_fine.to(t_vals)
        # print("t_fine.shape", t_fine.shape)
        # print("t_vals.shape", t_vals.shape)
        t_fine = torch.sort(torch.cat([t_vals, t_fine], dim=-1), dim=-1)[0]
        points_fine, step_size_fine = sample_along_rays(rays_o, rays_d, t_fine)
        points_fine_input = pos_encoding(points_fine, 10).to(device)
        rays_d_fine_input = pos_encoding(
            rays_d[..., None, :].expand(points_fine.shape), 4
        ).to(device)
        # Predict color and opacity of each sampled point using the NeRF model
        all_alpha = []
        all_rgb = []
        for i in range(0, points_fine.shape[0], chunk):
            points_chunk = points_fine_input[i : i + chunk]
            rays_d_chunk = rays_d_fine_input[i : i + chunk]
            x_chunk = torch.cat([points_chunk, rays_d_chunk], dim=-1)
            alpha_chunk, rgb_chunk = fine_model(x_chunk)
            all_alpha.append(alpha_chunk)
            all_rgb.append(rgb_chunk)
        alpha = torch.cat(all_alpha, dim=0)
        rgb = torch.cat(all_rgb, dim=0)
        # Combine colors and opacities into a single image
        fine_img, weights = volrend(
            alpha, rgb, step_size_fine.to(device), white_bkgd=white_bkgd
        )
        fine_depth_img = torch.sum(weights * t_fine.to(weights), dim=-1)
        fine_depth_img = (
            (fine_depth_img.reshape(H, W) * 40).detach().cpu().numpy().astype(np.uint8)
        )
        fine_depth_img = Image.fromarray(fine_depth_img)
        fine_img = (
            (fine_img.reshape(H, W, 3) * 255).detach().cpu().numpy().astype(np.uint8)
        )
        fine_img = Image.fromarray(fine_img)
        print("image rendered")
    return coarse_img, fine_img, depth_img, fine_depth_img


@torch.no_grad()
def render_gif(
    coarse_model, c2ws, H, W, K, chunk=1024, fine_model=None, white_bkgd=False
):
    # Compute ray origins and directions for each pixel in the image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # render a set of images and save them as gif
    coarse_imgs = []
    fine_imgs = []
    depth_imgs = []
    fine_depth_imgs = []
    for c2w in c2ws:
        if fine_model is None:
            img, depth_img = render_img(
                coarse_model, c2w, H, W, K, chunk, white_bkgd=white_bkgd
            )
            coarse_imgs.append(img)
            depth_imgs.append(depth_img)
        else:
            coarse_img, fine_img, depth_img, fine_depth_img = render_img(
                coarse_model, c2w, H, W, K, chunk, fine_model, white_bkgd=white_bkgd
            )
            coarse_imgs.append(coarse_img)
            fine_imgs.append(fine_img)
            depth_imgs.append(depth_img)
            fine_depth_imgs.append(fine_depth_img)
    # save the images to local, if exist, create a new name
    name_c = "img_coarse_rendered.gif"
    i = 0
    while os.path.exists(name_c):
        name_c = f"img_coarse_rendered_{i}.gif"
        i += 1
    #
    name_f = "img_fine_rendered.gif"
    i = 0
    while os.path.exists(name_f):
        name_f = f"img_fine_rendered_{i}.gif"
        i += 1

    coarse_imgs[0].save(
        name_c,
        save_all=True,
        append_images=coarse_imgs[1:],
        duration=100,
        loop=0,
    )
    print("img_coarse_rendered.gif saved to ./img_coarse_rendered.gif")
    if fine_model is not None:
        fine_imgs[0].save(
            name_f,
            save_all=True,
            append_images=fine_imgs[1:],
            duration=100,
            loop=0,
        )
        print("img_fine_rendered.gif saved to ./img_fine_rendered.gif")

    # save the depth images to local, if exist, create a new name
    name_d = "img_depth_rendered.gif"
    i = 0
    while os.path.exists(name_d):
        name_d = f"img_depth_rendered_{i}.gif"
        i += 1
    #
    name_fd = "img_fine_depth_rendered.gif"
    i = 0
    while os.path.exists(name_fd):
        name_fd = f"img_fine_depth_rendered_{i}.gif"
        i += 1
    depth_imgs[0].save(
        name_d,
        save_all=True,
        append_images=depth_imgs[1:],
        duration=100,
        loop=0,
    )
    print("img_depth_rendered.gif saved to ./img_depth_rendered.gif")
    if fine_model is not None:
        fine_depth_imgs[0].save(
            name_fd,
            save_all=True,
            append_images=fine_depth_imgs[1:],
            duration=100,
            loop=0,
        )
        print("img_fine_depth_rendered.gif saved to ./img_fine_depth_rendered.gif")
