import numpy as np
from typing import List 


class Gaussian3D:
    def __init__(self) -> None:
        self.init_default()

    def init_default(self):
        self._mean = np.zeros(3,)
        self._opacity = np.array(0.)
        scale = [1.0, 1.0, 1.0]
        quaternion = [1.0, 0.0, 0.0, 0.0]
        self._covariance = np.array(scale.extend(quaternion))
        self._color = np.array([0., 0., 0.]) # TODO: use spher harmonic to represent color
    
    def set_mean(self, mean):
        self._mean = mean
    
    def set_opacity(self, opacity):
        self._opacity = opacity
    
    def get_opacity(self):
        return self._opacity

    def set_covariance(self, covariance):
        self._covariance = covariance

    def set_color(self, color):
        self._color = color

    def get_color(self):
        return self._color

    @classmethod
    def create_from(cls, mean, opacity, covariance, color):
        cls.set_mean(mean)
        cls.set_opacity(opacity)
        cls.set_covariance(covariance)
        cls.set_color_rgb(color)
        return cls

    def covariance_matrix(self):
        '''
        3D Gaussian covariance matrix could be decomposed into 2 matrix
        scale (diag matrix) and rotation matrix (sym matrix)
        $\Sigma = RSS^TR^T$
        ''' 
        scale, quaternion = self.covariance[:3], self.covariance[3:]
        scale_matrix = np.zeros((3,3))
        scale_matrix[0,0] = scale[0]
        scale_matrix[1,1] = scale[1]
        scale_matrix[2,2] = scale[2]
        rotation_matrix = self._quaternion_to_rotation_matrix(quaternion)
        matrix_33 = rotation_matrix @ scale_matrix
        
        return matrix_33 @ matrix_33.T

    def to_covariance_2d(self, jacobian, world_to_image_transform):
        '''
        Project covariance 3d to 2d:
        $\Sigma_{2D} = JW \Sigma_{3D} W^TJ^T$
        J is Jacobian of the affine approximation of the projective transformation.
        '''
        return jacobian @ world_to_image_transform @ self.covariance_matrix() @ world_to_image_transform.T @ jacobian.T

    def _compute_Jocobian(self):
        pass

    def _quaternion_to_rotation_matrix(q):
        w, x, y, z = q
        R = np.array([
            [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
        ])
        return R
    
class Camera:
    def __init__(self, intrinsics, extrinsics, image_size_hw) -> None:
        self.intrinsics_33 = intrinsics
        self.extrinsics_44 = extrinsics
        self.image_size_hw = image_size_hw
        self.grid = self._build_grid()
        self._image = np.zeros((*self.image_size_hw, 3))

    def set_image(self, image):
        self._image = image
    
    def get_image(self):
        return self._image
    
    def _build_grid(self):
        h, w = self.image_size_hw
        offset = 0.5
        x_range = offset + np.linspace(0, w, w)
        y_range = offset + np.linspace(0, h, h)
        xx, yy = np.meshgrid(y_range, x_range, 'ij')
        grid_HW2 = np.concatenate((xx, yy), dim=1) # h,w,2
        return grid_HW2
    
    def grid_norm(self):
        # normalize grid coordinates to (-1, 1)
        half_h = self.image_size_hw[0] / 2
        half_w = self.image_size_hw[1] / 2
        grid_h_HW1 = (self.grid[:,:,1] - half_h) / half_h
        grid_w_HW1 = (self.grid[:,:,0] - half_w) / half_w
        grid_norm_HW2 = np.concatenate((grid_h_HW1, grid_w_HW1), dim=1)
        return grid_norm_HW2
    
    def world_to_camera(self, points_world_N4):
        # project points in world coordinate to camera coordinate using extrinsics
        if points_world_N4.shape[1] == 3:
            ones = np.ones((points_world_N4.shape[0], 1))
            points_world_N4 = np.concatenate((points_world_N4, ones), dim=1)
        points_camera_N3 = self.extrinsics_44 @ points_world_N4[:, :3]
        
        return points_camera_N3
    
    def world_to_camera_matrix(self):
        return self.extrinsics_44

    def camera_to_image(self, points_camera_N3):
        points_image_3N = self.intrinsics_33 @ points_camera_N3.T
        points_image_homo_N2 = (points_image_3N[:2, :] / points_image_3N[2, :]).T
        return points_image_homo_N2

    def camera_to_image_matrix(self):
        return self.intrinsics_33

    def world_to_image(self, points_world_N4):
        points_image_N2 = self.camera_to_image(self.world_to_camera(points_world_N4))
        return points_image_N2
    
    def world_to_image_matrix(self):
        return self.camera_to_image_matrix @ self.world_to_camera
    
class GaussianRaterizer:
    def __init__(self, camera: Camera) -> None:
        self.camera = camera
    
    def _compute_alpha(self, opacity, x2d_mean, cov2d):
        x = self.camera.grid_norm
        alpha = opacity * np.exp(-0.5 * (x-x2d_mean).T @ np.linalg.inv(cov2d) @ (x - x2d_mean))
        return alpha

    def alpha_blending_render(self, gaussians: List[Gaussian3D]):
        image = self.camera.get_image() # empty image
        # sort gaussians by depth, i.e. z
        gaussians = sorted(gaussians, key=lambda x : x.mean[2])
        T = np.array(1.0)
        for g3d in gaussians:
            color = g3d.get_color()
            g2d_mean = self.camera.world_to_camera(g3d.mean)
            jacobian = self._compute_Jacobian()
            world_to_image_transform = self.camera.world_to_image_matrix
            g2d_cov = g3d.to_covariance_2d(jacobian, world_to_image_transform)
            g2d_cov_22 = g2d_cov[:2, :2]
            alpha = self._compute_alpha(g3d.get_opacity(), g2d_mean, g2d_cov_22)
            image += color * alpha * T 
            T *= (1 - alpha)
        
        self.camera.set_image(image)

        return image