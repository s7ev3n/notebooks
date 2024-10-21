import viser
import torch

from camera_model import PinholeCameraModel
from imageio import imread_v2

def fake_camera_data():
    image = imread_v2('./camera/data/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402927612460.jpg')
    image = torch.from_numpy(image)
    image_size_hw = image.shape[:2] # (900, 1600)
    fx, fy, cx, cy = 1264.0, 1264.0, 450.0, 800.0
    intrinsics = torch.tensor([[fx , 0.0, cx],
                               [0.0, fy , cy],
                               [0.0, 0.0, 1.0]])
    extrinsics = torch.zeros((4, 4))  
    extrinsics[:3, :3] = torch.rand(3, 3, dtype=torch.float)
    extrinsics[3, 3] = torch.tensor(1.0)

    return image, image_size_hw, intrinsics, extrinsics

fake_image, fake_image_size_hw, fake_intrinsics, fake_extrinsics = \
    fake_camera_data()

camera_model = PinholeCameraModel(
    intrinsics=fake_intrinsics.unsqueeze(0).unsqueeze(0),
    extrinsics=fake_extrinsics.unsqueeze(0).unsqueeze(0),
    image_size=fake_image_size_hw
)

sight_rays_cam = camera_model.img2cam()

