import numpy as np
from PIL import Image
from typing import List 


class Gaussian3D:
    def __init__(self) -> None:
        self.init_default()

    def init_default(self):
        self._mean = np.zeros(3,)[..., None]
        self._opacity = np.zeros(1,)[..., None]
        scale = [1.0, 1.0, 1.0]
        quaternion = [1.0, 0.0, 0.0, 0.0]
        self._covariance = np.array(scale.extend(quaternion))[..., None]
        self._color = np.array([0., 0., 0.])[..., None] # TODO: use spher harmonic to represent color
    
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
    def __init__(self, intrinsics, extrinsics, image_size_hw, normalize_grid=True) -> None:
        self.intrinsics_33 = intrinsics
        self.extrinsics_44 = extrinsics
        self.image_size_hw = image_size_hw
        self.normalize_grid = normalize_grid # return image grid normalized to [-1, 1]
        self.grids = self._build_grid(self.normalize_grid)
        self._image = np.zeros((3, *self.image_size_hw))

    def set_image(self, image):
        self._image = image
    
    def get_image(self):
        return self._image
    
    def _build_grid(self, normalize_grid=True):
        h, w = self.image_size_hw
        offset = 0.5
        w_range = offset + np.linspace(0, w-1, w)
        h_range = offset + np.linspace(0, h-1, h)
        if normalize_grid:
            half_w, half_h = w/2, h/2
            w_range = (w_range - half_w) / half_w
            h_range = (h_range - half_h) / half_h
        hh, ww = np.meshgrid(h_range, w_range, indexing='ij')
        grid_HW2 = np.stack((hh, ww), axis=-1) 
        return grid_HW2
    
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
        points_image_N2 = (points_image_3N[:2, :] / points_image_3N[2, :]).T
        return points_image_N2

    @property    
    def camera_to_image_matrix(self):
        return self.intrinsics_33

    def world_to_image(self, points_world_N4):
        points_image_N2 = self.camera_to_image(self.world_to_camera(points_world_N4))
        if self.normalize_grid:
            h, w = self.image_size_hw
            half_w, half_h = w/2, h/2
            points_image_N2[:, 0] = (points_image_N2[:, 0] - half_w) / half_w
            points_image_N2[:, 1] = (points_image_N2[:, 1] - half_h) / half_h
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
    
    def _compute_alpha(self, opacity, xmean2d_20, cov2d_22):
        x_HW2 = self.camera.grids
        d_HW2 = x_HW2-xmean2d_20[None, None,...]
        alpha_1HW = opacity[None, None, ...] * np.exp(-0.5 * d_HW2[..., None, :] @ np.linalg.inv(cov2d_22) @ d_HW2[..., None]).squeeze()
        return alpha_1HW

    def alpha_blending_render(self, gaussians: List[Gaussian3D]):
        # sort gaussians by depth, i.e. z
        gaussians = sorted(gaussians, key=lambda x : x.get_mean()[2])
        cov3d_N33 = np.vstack([g.covariance_matrix()[None, ...] for g in gaussians])
        mean3d_N3 = np.vstack([g.get_mean() for g in gaussians])
        mean2d_N2 = self.camera.world_to_image(mean3d_N3)
        cov2d_N22 = self.to_covariance_2d(mean3d_N3, cov3d_N33)[:, :2, :2]
        colors_N0 = np.vstack([g.get_color() for g in gaussians])
        opacity_N0 = np.vstack([g.get_opacity() for g in gaussians])

        image = self.camera.get_image() # empty image
        T = np.ones((1, *self.camera.image_size_hw))
        for mean_2d, cov_2d, color, opacity in zip(mean2d_N2, cov2d_N22, colors_N0, opacity_N0):
            alpha = self._compute_alpha(opacity, mean_2d, cov_2d)
            image += T * alpha * color[..., None, None]
            T = T*(1 - alpha)
        
        self.camera.set_image(image)

        return image

    def to_covariance_2d(self, mean3d_N3, cov3d_N33):
        '''
        Project covariance 3d to 2d:
        $\Sigma_{2D} = JW \Sigma_{3D} W^TJ^T$
        J is Jacobian of the affine approximation of the projective transformation.
        '''
        
        jacobian = self.compute_Jocobian(mean3d_N3)
        R = self.camera.world_to_camera_matrix[:3, :3]
        return jacobian @ R[None, ...] @ cov3d_N33 @ R.T[None, ...] @ jacobian.swapaxes(1, 2)

    def compute_Jocobian(self, mean3d_N3):
        '''
        Compute the Jacobian of the affine approximation of the projective transformation.
        '''
        t = self.camera.world_to_camera(mean3d_N3)
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
            mean=np.array([(i + 1)**2, 0, 0]),
            opacity=np.array([0.7]),
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
    extrinsics = np.zeros((4,4))
    extrinsics[:3, :3] = np.array([[ 0.0,  -1.0,  0.0],
                                   [ 0.0,  0.0,  -1.0],
                                   [ 1.0, 0.0,  0.0]])
    extrinsics[:3, 3] = np.array([0, 0, 0])
    camera = Camera(
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        image_size_hw=(900, 1600)
    )
    gaussians = fake_gaussians()
    raterizer = GaussianRaterizer(camera)
    image = 255 * raterizer.alpha_blending_render(gaussians)
    # save image
    image = image.transpose(1, 2, 0)
    image = Image.fromarray(image.astype('uint8'))
    image.save('output.png')