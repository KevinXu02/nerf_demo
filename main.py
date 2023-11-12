import numpy as np
from data_processing.data_loader import data_loader
from data_processing.rays import RaysData
from models.model import NeRF
import torch
from PIL import Image
from utils.utils import *
from utils.train_utils import *


if __name__ == "__main__":
    # --- You Need to Implement These ------
    images_train, c2ws_train, images_val, c2ws_val, c2ws_test, focal = data_loader()
    K = get_intrinsic_matrix(focal, images_train.shape[1], images_train.shape[2])
    # dataset = RaysData(images_train, K, c2ws_train, focal)
    H, W = images_train.shape[1:3]

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
    # model = train_coarse_to_fine()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    coarse_model = NeRF()
    coarse_model.load_state_dict(torch.load(f"./saved_models/coarse_model_5000.pt"))
    coarse_model.to(device)

    fine_model = NeRF()
    fine_model.load_state_dict(torch.load(f"./saved_models/fine_model_5000.pt"))
    fine_model.to(device)
    fine_model.eval()

    img_c, img_f, depth, depth_fine = render_img(
        coarse_model=coarse_model,
        fine_model=fine_model,
        c2w=c2ws_val[0],
        H=H,
        W=W,
        K=K,
        chunk=512,
        white_bkgd=False,
    )
    img_c.save("./test_img/img_c.png")
    img_f.save("./test_img/img_f.png")
    depth.save("./test_img/depth.png")
    depth_fine.save("./test_img/depth_fine.png")
    # render_gif(
    #     coarse_model=coarse_model,
    #     fine_model=fine_model,
    #     c2ws=c2ws_test,
    #     H=H,
    #     W=W,
    #     K=K,
    #     chunk=1024,
    #     white_bkgd=False,
    # )


# rays_o, rays_d, target = dataset.sample_rays(500)
# t_vals = sample_t_vals(near=0, far=6.0, n_samples=64, perturb=False)
# points, step_size = sample_along_rays(rays_o, rays_d, t_vals)
# # transform all the data to np
# rays_o = rays_o.cpu().numpy()
# rays_d = rays_d.cpu().numpy()
# points = points.cpu().numpy()
# target = target.cpu().numpy()
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
