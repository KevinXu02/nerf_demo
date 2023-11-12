import viser, time  # pip install viser
import numpy as np
from data_processing.data_loader import data_loader
from utils.utils import sample_along_rays
from utils.utils import get_intrinsic_matrix
from data_processing.rays import RaysData

# --- You Need to Implement These ------
images_train, c2ws_train, _, _, _, focal = data_loader()
K = get_intrinsic_matrix(focal, images_train.shape[1], images_train.shape[2])
dataset = RaysData(images_train, K, c2ws_train)
rays_o, rays_d, pixels = dataset.sample_rays(100)
points = sample_along_rays(rays_o, rays_d, perturb=True)
H, W = images_train.shape[1:3]
# ---------------------------------------
print("points shape:", points.shape)
print("rays_o shape:", rays_o.shape)
print("rays_d shape:", rays_d.shape)
print("pixels shape:", pixels.shape)
print("K shape:", K.shape)
print("H:", H)
print("W:", W)

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
    # visualize rays_d
    server.add_point_cloud(
        f"/rays_d/{i}",
        # red color
        colors=np.ones_like(d).reshape(-1, 3),
        points=d.reshape(-1, 3),
        point_size=0.02,
    )
server.add_point_cloud(
    f"/samples",
    colors=np.zeros_like(points).reshape(-1, 3),
    points=points.reshape(-1, 3),
    point_size=0.02,
)
time.sleep(1000)
