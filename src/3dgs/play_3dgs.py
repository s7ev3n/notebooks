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

    def get_mean(self):
        return self._mean

    def set_opacity(self, opacity):
        self._opacity = opacity

    def get_opacity(self):
        return self._opacity

    def set_covariance(self, covariance):
        self._covariance = covariance

    def get_covariance(self):
        return self._covariance

    def set_color(self, color):
        self._color = color

    def get_color(self):
        return self._color

    @classmethod
    def create_from(cls, mean, opacity, covariance, color):
        instance = cls()
        instance.set_mean(mean)
        instance.set_opacity(opacity)
        instance.set_covariance(covariance)
        instance.set_color(color)
        return instance

    def covariance_matrix(self):
        '''
        3D Gaussian covariance matrix could be decomposed into 2 matrix
        scale (diag matrix) and rotation matrix (sym matrix)
        $\Sigma = RSS^TR^T$
        ''' 
        scale, quaternion = self._covariance[:3], self._covariance[3:]
        scale_matrix = np.zeros((3,3))
        scale_matrix[0,0] = scale[0]
        scale_matrix[1,1] = scale[1]
        scale_matrix[2,2] = scale[2]
        rotation_matrix = self._quaternion_to_rotation_matrix(quaternion)
        matrix_33 = rotation_matrix @ scale_matrix
        
        return matrix_33 @ matrix_33.T

    def _quaternion_to_rotation_matrix(self, q):
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
        self._image = np.zeros((3, *self.image_size_hw))

    def set_image(self, image):
        self._image = image
    
    def get_image(self):
        return self._image
    
    def _build_grid(self):
        h, w = self.image_size_hw
        offset = 0.5
        w_range = offset + np.linspace(0, w-1, w)
        h_range = offset + np.linspace(0, h-1, h)
        ww, hh = np.meshgrid(w_range, h_range, indexing='ij')
        grid_HW2 = np.stack((ww, hh), axis=-1) 
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
            points_world_N4 = np.hstack((points_world_N4, ones))
        points_camera_N3 = (self.extrinsics_44 @ points_world_N4.T)[:3, :].T
        
        return points_camera_N3
    
    @property
    def world_to_camera_matrix(self):
        return self.extrinsics_44

    def camera_to_image(self, points_camera_N3):
        points_image_3N = self.intrinsics_33 @ points_camera_N3.T
        points_image_homo_N2 = (points_image_3N[:2, :] / points_image_3N[2, :]).T
        return points_image_homo_N2

    @property    
    def camera_to_image_matrix(self):
        return self.intrinsics_33

    def world_to_image(self, points_world_N4):
        points_image_N2 = self.camera_to_image(self.world_to_camera(points_world_N4))
        return points_image_N2
    
    @property
    def world_to_image_matrix(self):
        intrinsics_44 = np.eye(4)
        intrinsics_44[:3, :3] = self.intrinsics_33

        projection_matrix = intrinsics_44 @ self.extrinsics_44
        
        return projection_matrix
    
class GaussianRaterizer:
    def __init__(self, camera: Camera) -> None:
        self.camera = camera
    
    def _compute_alpha(self, opacity, x2d_mean, cov2d):
        x = self.camera.grid_norm
        alpha = opacity * np.exp(-0.5 * (x-x2d_mean).T @ np.linalg.inv(cov2d) @ (x - x2d_mean))
        return alpha

    def alpha_blending_render(self, gaussians: List[Gaussian3D]):
        # sort gaussians by depth, i.e. z
        gaussians = sorted(gaussians, key=lambda x : x.get_mean()[2])
        cov3ds = np.vstack([g.covariance_matrix()[None, ...] for g in gaussians])
        mean3ds = np.vstack([g.get_mean() for g in gaussians])
        mean2ds = self.camera.world_to_image(mean3ds)
        cov2ds = self.to_covariance_2d(mean3ds, cov3ds)[:2, :2]
        colors = np.vstack([g.get_color() for g in gaussians])
        opacities = np.vstack([g.get_opacity() for g in gaussians])

        image = self.camera.get_image() # empty image
        T = np.ones((1, *self.camera.image_size_hw))
        for mean_2d, cov_2d, color, opacity in zip(mean2ds, cov2ds, colors, opacities):
            alpha = self._compute_alpha(opacity, mean_2d, cov_2d)
            image += color * alpha * T 
            T *= (1 - alpha)
        
        self.camera.set_image(image)

        return image

    def to_covariance_2d(self, mean_3d, cov_3d):
        '''
        Project covariance 3d to 2d:
        $\Sigma_{2D} = JW \Sigma_{3D} W^TJ^T$
        J is Jacobian of the affine approximation of the projective transformation.
        '''
        
        jacobian = self.compute_Jocobian(mean_3d)
        world_to_image_transform = self.camera.world_to_image_matrix[:3, :3]
        return jacobian @ world_to_image_transform[None, ...] @ cov_3d @ world_to_image_transform.T @ jacobian.T

    def compute_Jocobian(self, means_3d):
        '''
        Compute the Jacobian of the affine approximation of the projective transformation.
        '''
        t = self.camera.world_to_camera(means_3d)
        l = np.linalg.norm(t, axis=1, keepdims=True).flatten()
        # Compute the jacobian according to (29) from EWA Volume Splatting M.Zwicker et. al (2001)
        jacobian = np.zeros((t.shape[0], 3, 3))
        jacobian[:, 0, 0] = 1/t[:, 2]
        jacobian[:, 0, 2] = -t[:, 0]/t[:, 2]**2
        jacobian[:, 1, 1] = 1/t[:, 2]
        jacobian[:, 1, 2] = -t[:, 1]/t[:, 2]**2
        jacobian[:, 2, 0] = t[:, 0]/l
        jacobian[:, 2, 1] = t[:, 1]/l
        jacobian[:, 2, 2] = t[:, 2]/l

        return jacobian

def fake_gaussians(num=3):
    red = np.array([1., 0., 0.])
    green = np.array([0., 1., 0.])
    blue = np.array([0., 0., 1.])
    colors = [red, green, blue]
    gaussians = []
    for i in range(num):
        g = Gaussian3D.create_from(
            mean=np.array([0, 0, np.random.rand()]),
            opacity=np.random.rand(1,),
            covariance=np.array([*np.random.rand(3), 1, 0, 0, 0]),
            color=colors[i%3]
        )
        
        gaussians.append(g)
    
    return gaussians
    
if __name__ == '__main__':
    fx, fy, cx, cy = 1252.8131, 1252.8131, 826.5881, 469.9846
    intrinsics = np.array([[fx , 0.0, cx],
                           [0.0, fy , cy],
                           [0.0, 0.0, 1.0]])
    extrinsics = np.eye(4)
    extrinsics[:3, 3] = np.array([0., 0., -3.0])
    camera = Camera(
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        image_size_hw=(900, 1600)
    )
    gaussians = fake_gaussians()
    raterizer = GaussianRaterizer(camera)
    image = raterizer.alpha_blending_render(gaussians)