import viser
import torch
import time

from camera_model import PinholeCameraModel
import imageio.v3 as iio

def fake_camera_data():
    image = iio.imread('./data/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402927612460.jpg')
    # image = torch.from_numpy(image)
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

# camera_model = PinholeCameraModel(
#     intrinsics=fake_intrinsics.unsqueeze(0).unsqueeze(0),
#     extrinsics=fake_extrinsics.unsqueeze(0).unsqueeze(0),
#     image_size=fake_image_size_hw
# )

# sight_rays_cam = camera_model.img2cam()

def main() -> None:
    server = viser.ViserServer()

    server.scene.add_frame('world', 
                        wxyz=(1, 0, 0, 0), 
                        position=(0, 0, 0))
    # NOTE: how to calculate the world/camera wxyz
    # 1. wxyz is camera rotation in world coordinate
    # 2. use Euler angles (order: XYZ) to rotate world coordinate to camera coordinate,
    #    in this case, X:-90, Y:90, Z:0, and use the tool https://www.andre-gaschler.com/rotationconverter/
    #    to do the conversion from Euler angles to wxyz (quaternion)
    # 3. Euler angle postive angle: clockwise, negative angle: counterclockwise
    server.scene.add_frame('world/camera', 
                        wxyz=( -0.5, 0.5, -0.5, 0.5),
                        position=(3.0, 0.0, 0.0))
    
    server.scene.add_image('world/camera/image', 
                           fake_image, 
                           8, 
                           4.5,
                           format='jpeg',
                           wxyz=(1, 0, 0, 0),
                           position=(0.0, 0.0, 1.0))
    
    while True:
        time.sleep(0.2)

if __name__ == "__main__":
    main()
