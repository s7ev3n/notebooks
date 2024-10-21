import torch
from scipy.spatial.transform import Rotation as R
from torch import nn

from camera.camera_model import CameraModel, CylindricalCameraModel, PinholeCameraModel, SphericalCameraModel


def normalize_rotation(
    rot_cam2world: torch.Tensor, up_vector_DIN70k_target: torch.Tensor = torch.Tensor([0.0, 0.0, 1.0])
):
    device = rot_cam2world.device
    # Get camera up vector in DIN70k convention by taking negative y axis of camera (defined in CV coordinates)
    up_vector_DIN70k_current = -rot_cam2world[:, 1]
    # Normalize current camera up vector (just in case...)
    up_vector_DIN70k_current = torch.nn.functional.normalize(up_vector_DIN70k_current, p=2.0, dim=0)
    # Normalize target camera up vector (in DIN70k convention)
    up_vector_DIN70k_target = torch.nn.functional.normalize(up_vector_DIN70k_target.to(device), p=2.0, dim=0)
    # Get rotation axis (orthogonal vector to current and target axis vector)
    axis = torch.cross(up_vector_DIN70k_current, up_vector_DIN70k_target)
    # Normalize vector
    axis_normalized = torch.nn.functional.normalize(axis, p=2.0, dim=0)
    # Get rotation angle between current and target axis vector
    angle = torch.dot(up_vector_DIN70k_current, up_vector_DIN70k_target).acos()
    # Scale rotation axis with angle
    axis_normalized_angle = angle * axis_normalized
    # Rotate real rotation matrix to target rotation matrix
    rot_virtual_cam2world = (
        torch.Tensor(R.from_rotvec(axis_normalized_angle.cpu()).as_matrix()).to(device) @ rot_cam2world
    )
    return rot_virtual_cam2world


def perform_rectification(
    img_in: torch.Tensor,
    grid_uv: torch.Tensor,
):
    """Rectification of input image on givn grid.

    Parameters
    ----------
        img_in(torch.Tensor(C, H, W)): Input image with C channels, height H and width W.
        grid_uv(torch.Tensor(H, W, 2)): Grid for resampling

    Returns
    -------
        img_rect(torch.Tensor(C, H_OUT, W_OUT)): Rectified image of virtual camera with
                    C channels, height H_OUT and width W_OUT (matching virtual_image_size).
    """
    # Normalize grid
    _, h, w = img_in.shape
    grid_uv[..., 0] = (2 / w) * grid_uv[..., 0] - 1
    grid_uv[..., 1] = (2 / h) * grid_uv[..., 1] - 1
    # Clamp grid
    grid_uv = grid_uv.clamp(-1.1, 1.1)

    # Interpolate transformed features on regular BEV grid
    img_rect = nn.functional.grid_sample(
        img_in.unsqueeze(0),
        grid_uv.unsqueeze(0),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return img_rect


def get_grid_rectification(
    actual_intrinsic: torch.Tensor,
    actual_extrinsic: torch.Tensor,
    actual_camera_model: CameraModel,
    actual_image_size: tuple,
    virtual_intrinsic: torch.Tensor,
    virtual_extrinsic: torch.Tensor,
    virtual_camera_model: CameraModel,
    virtual_image_size: tuple,
    aug_trafo_intrinsic: torch.Tensor = None,
    aug_trafo_extrinsic: torch.Tensor = None,
    do_normalization: bool = True,
    up_vector_DIN70k_target: torch.Tensor = torch.Tensor([0.0, 0.0, 1.0]),
    use_auto_focal_length: bool = False,
):
    """Creation of grid for rectification to virtual camera with perfectly upright camera.

    Parameters
    ----------
        actual_intrinsic(torch.Tensor(3, 3)): Intrinsic camera matrix of actual camera.
        actual_extrinsic(torch.Tensor(4, 4)): Extrinsic camera matrix of actual camera.
        actual_camera_model(CameraModel): Actual camera model.
        actual_image_size(tuple(2)): Dimension of actual image (height, width).
        virtual_intrinsic(torch.Tensor(3, 3)): Intrinsic camera matrix of virtual camera.
        virtual_extrinsic(torch.Tensor(4, 4)): Extrinsic camera matrix of virtual camera.
        virtual_camera_model(CameraModel): Virtual camera model.
        virtual_image_size(tuple(2)): Dimension of virtual image (height, width).
        aug_trafo_extrinsic(torch.Tensor(4, 4)): Augmentation matrix for extrinsic.
        do_normalization(bool): Switch for camera normalization.
        up_vector_DIN70k_target(torch.Tensor(3)): Definition of target camera up vector
                                                  for camera normalization.
        use_auto_focal_length(bool): Automatic optimization of focal length.

    Returns
    -------
        grid_uv(torch.Tensor(H_OUT, W_OUT, 2)): Grid for rectification to virtual camera,
                    height H_OUT and width W_OUT (matching virtual_image_size).
        virtual_intrinsic(torch.Tensor(3, 3)): Intrinsic camera matrix of virtual camera.
        virtual_extrinsic(torch.Tensor(4, 4)): Extrinsic camera matrix of virtual camera.
    """
    # Define actual camera
    # Add dimension for batch size and number of cameras
    actual_camera = actual_camera_model(
        intrinsics=actual_intrinsic.unsqueeze(0).unsqueeze(0),
        extrinsics=actual_extrinsic.unsqueeze(0).unsqueeze(0),
        image_size=actual_image_size,
    )
    if do_normalization:
        # Normalize rotation matrix to virtual camera
        virtual_extrinsic[:3, :3] = normalize_rotation(virtual_extrinsic[:3, :3], up_vector_DIN70k_target)
    if aug_trafo_extrinsic is not None:
        # Apply extrinsic augmentation
        virtual_extrinsic = virtual_extrinsic @ aug_trafo_extrinsic
    if use_auto_focal_length:
        if issubclass(virtual_camera_model, (CylindricalCameraModel, SphericalCameraModel)):
            f_x_new = virtual_image_size[1] / actual_camera.get_camera_fov()[0][0, 0]
        elif issubclass(virtual_camera_model, PinholeCameraModel):
            f_x_new = virtual_image_size[1] / 2 / torch.tan(actual_camera.get_camera_fov()[0][0, 0] / 2)
        else:
            raise ValueError(f"Unknown camera model: {virtual_camera_model}")
        f_x = virtual_intrinsic[..., 0, 0]
        f_y = virtual_intrinsic[..., 1, 1]
        assert f_x.lt(0).type(f_x.dtype) == f_y.lt(0).type(f_y.dtype), "If scaling, then in both dimensions."
        do_scale = f_x.lt(0).type(f_x.dtype)
        dont_scale = 1 - do_scale
        virtual_intrinsic[..., 0, 0] = f_x_new * (dont_scale - do_scale * f_x)
        virtual_intrinsic[..., 1, 1] = f_x_new * (dont_scale * f_y / f_x - do_scale * f_y)  # f_x_new is no bug here
    if aug_trafo_intrinsic is not None:
        # Apply intrinsic augmentation
        virtual_intrinsic = aug_trafo_intrinsic @ virtual_intrinsic
    # Define virtual camera
    # Add dimension for batch size and number of cameras
    virtual_camera = virtual_camera_model(
        intrinsics=virtual_intrinsic.unsqueeze(0).unsqueeze(0),
        extrinsics=virtual_extrinsic.unsqueeze(0).unsqueeze(0),
        image_size=virtual_image_size,
    )
    # Unproject to output camera
    sight_rays_virtual = virtual_camera.img2cam()
    # Rotate
    sight_rays_rot_virtual = virtual_camera.cam2world_rot(sight_rays_virtual)
    # Apply inverse (extrinsic) rotation
    sight_rays_cam = actual_camera.world2cam_rot(sight_rays_rot_virtual)
    # Project camera world coordinates to image/vector pixel/array position
    grid_uv, _, _ = actual_camera.cam2img(sight_rays_cam)
    # Reshape
    grid_uv = grid_uv.view(virtual_image_size[0], virtual_image_size[1], 2)
    return grid_uv, virtual_intrinsic, virtual_extrinsic
