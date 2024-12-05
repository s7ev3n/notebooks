# 3D Gaussian Splatting

## Background
3DGS有很多图像学中的背景，这里做一些背景知识的补充。

### Splatting 泼溅

Splatting称为泼溅，是图形学中的一种渲染技术，很形象的说就是将3D的点或者体素“泼洒”到2D图像上。为了完成渲染有三个关键的技术点：
1. 投影，将3D的点投影到图像平面
2. 核函数，即该点的影响强度和范围。一般有：1）高斯核；2）EWA核，椭圆加权平均
3. 混合(Blending)，即在图像区域重叠的地方可能有多个点的影响，需要正确的混合，可以考虑按照深度进行排序确定贡献的大小

### Rasterization 栅格化

> 之前在做点云3维目标检测的时候，会将空间栅格化，当然更多的是使用Voxelization，但是好像也有类似使用Rastserization的用词。

图形学中的Rasterization值得是把矢量图变为离散的像素值的过程，所以抗锯齿(anti-aliasing)技术来减少离散化过程中的边缘的锯齿化现象，可以wiki一下看看。


## 3D Gaussian Modelling
3DGS中，使用空间中的3D Gaussian来对场景进行显示的建模，每个3D Gaussian表示为:
$$G(\mathbf{x_i})=o_i \cdot e^{-\frac{1}{2}(\mathbf{x_i}-\mu_i)^T\Sigma_{i}^{-1}(\mathbf{x_i}-\mu_i)}$$

其中，$\mathbf{x_i}$表示的是需要渲染的像素点的坐标位置(归一化到$(-1,1)$区间)，$\mu$ 表示平均值，即3D Gaussian的中心位置，$\Sigma$表示的是多元高斯分布的协方差矩阵。$o_i$表示的是不透明度(opacity)，注意$o_i$同样定义在3D空间中，在进行alpha blending渲染时，需要转换到2D平面。

除了3D Gaussian的参数$\mu$，$\alpha$和$\Sigma$外，渲染时对颜色的建模采用的是称为球谐函数(spherical harmonic coefficients)的参数，场景中的所有3D Gaussian的参数构成了模型训练的权重参数。

> 注意：<ins>**场景的模型权重就是场景中的所有3D Gaussian的参数，这里没有MLP，和NeRF隐式的表征场景不一样，这里3D Gaussian参数本身就是场景的权重参数**，你可以理解这里是更灵活的、非结构化的Voxel表征。</ins>

### 3D Gaussian参数各项

##  Rasterizer

### Volume Rendering

$C = \sum_{i=1}^{N}  \cdot c_i \cdot \alpha_i \cdot \prod_{j=1}^{i-1}(1-\alpha_j)$

### Jacobian
在EWA volume splatting方法中，计算Jacobian的过程是为了近似非线性透视投影变换。Jacobian是一个矩阵，它包含了变换的一阶偏导数信息，用于描述在局部区域内，一个函数如何将一个空间映射到另一个空间。在EWA volume splatting中，Jacobian用于将重建核（reconstruction kernel）从相机空间（camera space）映射到光线空间（ray space）。

具体来说，Jacobian的计算如下：

1. **定义Jacobian矩阵**：Jacobian矩阵 $J_k$ 是在相机空间中某点 $u_k$ 处的透视投影变换的偏导数矩阵。这个矩阵描述了从相机空间到图像空间的局部线性近似。

2. **计算Jacobian矩阵的元素**：Jacobian矩阵 $J_k$ 的元素是通过在点 $u_k$ 处对透视投影变换 $Pi$ 的偏导数来计算的。对于一个给定的相机空间坐标 $u_k = (u_{k0}, u_{k1}, u_{k2})$，Jacobian矩阵$J_k$ 可以表示为：
$$
   J_k = \begin{bmatrix}
   \frac{1}{u_{k2}} & 0 & -\frac{u_{k0}}{u_{k2}^2} \\
   0 & \frac{1}{u_{k2}} & -\frac{u_{k1}}{u_{k2}^2} \\
   \frac{u_{k0}}{\|u_k\| u_{k2}} & \frac{u_{k1}}{\|u_k\| u_{k2}} & \frac{u_{k2}}{\|u_k\| u_{k2}}
   \end{bmatrix}
$$
其中 $\|u_k\|$ 是 $u_k$ 的欧几里得范数，即 $\sqrt{u_{k0}^2 + u_{k1}^2 + u_{k2}^2}$。

1. **使用Jacobian矩阵**：在EWA volume splatting中，Jacobian矩阵$J_k$ 用于将重建核从相机空间变换到光线空间。这个变换涉及到将重建核的方差矩阵$V_k$从对象空间变换到光线空间，得到新的方差矩阵$V'_k$，计算公式为：
$$
   V'_k = J_k W V''_k W^T J_k^T
$$
其中 $W$ 是视图变换中旋转部分的3x3矩阵，$V''_k$ 是对象空间中重建核的3x3方差矩阵。

通过上述步骤，EWA volume splatting方法能够有效地处理透视投影带来的采样率变化问题，从而减少混叠（aliasing）伪影，并提高体积渲染的质量。
