import viser
import torch
import time
import numpy as np
import imageio.v3 as iio
import trimesh

from camera_model import PinholeCameraModel, CylindricalCameraModel
from rectification import get_grid_rectification, perform_rectification

fx = 1264.0

def fake_camera_data():
    image = iio.imread('./camera/data/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402927612460.jpg')
    # image = torch.from_numpy(image)
    image_size_hw = image.shape[:2] # (900, 1600)
    fx, fy, cx, cy = 1252.8131, 1252.8131, 826.5881, 469.9846
    intrinsics = torch.tensor([[fx , 0.0, cx],
                               [0.0, fy , cy],
                               [0.0, 0.0, 1.0]])
    extrinsics = torch.zeros((4, 4))  
    extrinsics[:3, :3] = torch.rand(3, 3, dtype=torch.float)
    extrinsics[3, 3] = torch.tensor(1.0)

    return image, image_size_hw, intrinsics, extrinsics

fake_image, fake_image_size_hw, fake_intrinsics, fake_extrinsics = \
    fake_camera_data()

fake_image_cuda = torch.from_numpy(fake_image.transpose(2,0,1)).to("cuda")
fake_intrinsics = fake_intrinsics.to("cuda")
fake_extrinsics = fake_extrinsics.to("cuda")

camera_model_mappings = {
    "Pinhole": PinholeCameraModel,
    "Cylindrical": CylindricalCameraModel
}

def update_focal_scaling_image(old_intrinsics, new_focal_length):
    new_intrinsics = old_intrinsics.clone()
    new_intrinsics[0, 0] = new_focal_length
    new_intrinsics[1, 1] = new_focal_length

    grid_uv, _, _ = get_grid_rectification(
        fake_intrinsics,
        fake_extrinsics,
        PinholeCameraModel,
        fake_image_size_hw,
        new_intrinsics,
        fake_extrinsics,
        PinholeCameraModel,
        fake_image_size_hw,
        do_normalization=False
    )

    image_rect = perform_rectification(fake_image_cuda.float(), grid_uv)
    
    return image_rect.cpu().numpy().astype(np.uint8).transpose(1, 2, 0)

def create_arrow(start=[0,0,0], end=[0,0,1], radius=0.01, head_length=0.2, head_radius=0.05):

    direction = np.array(end) - np.array(start)
    length = np.linalg.norm(direction)
    direction_normalized = direction / length
    print(length)
    cylinder = trimesh.creation.cylinder(radius=radius, height=length)
    # cylinder.apply_translation(start + direction_normalized * (length - head_length/2))    
    cone = trimesh.creation.cone(radius=head_radius, height=head_length)
    cone.apply_translation(start + direction_normalized * length / 2) 

    arrow = trimesh.util.concatenate([cylinder, cone])
    return arrow

def main() -> None:
    server = viser.ViserServer()

    # Camera model switch
    default_camera_model = "Pinhole"
    with server.gui.add_folder("Camera Model"):
        gui_camera_model_display = server.gui.add_text(
            "Camera Model: ",
            default_camera_model,
            disabled=True
        )
        gui_camera_model_switch = server.gui.add_button_group(
            "Camera Model:",
            ["Pinhole", "Cylindrical"]
        )

    # Camera parameters adjustment
    with server.gui.add_folder("Adjust Intrinsics"):  
        gui_focal_display = server.gui.add_number(
            "Current Focal Length: ",
            initial_value=fx,
            disabled=True
        )
        gui_focal_slider = server.gui.add_slider(
                                "Focal Length Scaling",
                                min=1,
                                max=1.8,
                                step=0.1,
                                initial_value=1
                            )

    def update_image(image = None) -> None:
        server.scene.add_image('world/camera/image', 
                                image if image is not None else fake_image, 
                                8, 
                                4.5,
                                format='jpeg',
                                wxyz=(1, 0, 0, 0),
                                position=(0.0, 0.0, 1.0))
    
    def update_camera_model_switch_image(from_camera, to_camera):
        from_camera_model = camera_model_mappings[from_camera]
        to_camera_model = camera_model_mappings[to_camera]
        
        to_intrinsics = fake_intrinsics.clone()
        to_intrinsics[0, 0] = torch.tensor(gui_focal_display.value, device=to_intrinsics.device)
        to_intrinsics[1, 1] = torch.tensor(gui_focal_display.value, device=to_intrinsics.device)
        grid_uv, _, _ = get_grid_rectification(
            fake_intrinsics,
            fake_extrinsics,
            from_camera_model,
            fake_image_size_hw,
            to_intrinsics,
            fake_extrinsics,
            to_camera_model,
            fake_image_size_hw,
            do_normalization=False
            )

        image_rect = perform_rectification(fake_image_cuda.float(), grid_uv)
        
        return image_rect.cpu().numpy().astype(np.uint8).transpose(1, 2, 0)


    @gui_focal_slider.on_update
    def _(_) -> None:
        scale = gui_focal_slider.value
        new_focal_length = fx * scale
        gui_focal_display.value = new_focal_length

        new_image = update_focal_scaling_image(fake_intrinsics, new_focal_length)
        update_image(new_image)
        
    @gui_camera_model_switch.on_click
    def _(_):
        if gui_camera_model_switch.value != gui_camera_model_display.value:
            new_image = update_camera_model_switch_image(
                gui_camera_model_display.value, 
                gui_camera_model_switch.value)
            gui_camera_model_display.value = gui_camera_model_switch.value
            update_image(new_image)            

    with server.gui.add_folder("Adjust Extrinsics"):
        gui_camera_tran = server.gui.add_vector3(
            "Translation: ",
            initial_value=(0.0, 0.0, 1.0),
            step=0.1
        )
        gui_camera_rot = server.gui.add_vector3(
            "Rotation (Euler angles (deg) XYZ ):",
            initial_value=(0.0, 0.0, 0.0),
            step = 1
        )



    server.scene.add_frame('world', 
                        wxyz=(1, 0, 0, 0), 
                        position=(0, 0, 0))
    server.scene.add_grid('world/ground',
                          width=20,
                          height=20,
                          width_segments=80,
                          height_segments=80,
                          plane='xy',
                          wxyz=(1, 0, 0, 0),
                          position=(0.0, 0.0, 0.0))
    
    server.scene.add_mesh_trimesh(
        "arrow", 
        create_arrow(),
        scale=1.0,
        wxyz=(1, 0, 0, 0),
        position=(0.0, 0.0, 2.0))

    # NOTE: how to calculate the world/camera wxyz
    # 1. wxyz is camera rotation in world coordinate
    # 2. use Euler angles (order: XYZ) to rotate world coordinate to camera coordinate,
    #    in this case, X:-90, Y:90, Z:0, and use the tool https://www.andre-gaschler.com/rotationconverter/
    #    to do the conversion from Euler angles to wxyz (quaternion)
    # 3. Euler angle postive angle: clockwise, negative angle: counterclockwise
    server.scene.add_frame('world/camera', 
                        wxyz=( -0.5, 0.5, -0.5, 0.5),
                        position=(3.0, 0.0, 5.0))
    
    fov = 2 * np.arctan2(900, fx)
    aspect = 1600 / 900
    server.scene.add_camera_frustum(
        'world/camera/frustum',
        fov=fov,
        aspect=aspect,
        scale=0.15,
        wxyz=(1,0,0,0),
        position=(0.0, 0.0, 0.0))
    
    
    update_image()

    while True:
        time.sleep(0.2)

if __name__ == "__main__":
    main()
