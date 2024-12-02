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
3DGS中，使用空间中的3D Gaussian分布，对场景进行显示的建模，即
3D Gaussian参数：3D position$x,y,z$, opacity$\alpha$, anisotropic covariance, spherical hamonic coefficients
**场景的模型权重就是场景中的所有3D Gaussian的参数，这里没有MLP，和NeRF隐式的表征场景不一样，这里3D Gaussian本身就是场景的表示**，你可以理解这里是更灵活的、非结构化的Voxel表征。


##  Rasterizer

### Volume Rendering

$C = \sum_{i=1}^{N}  \cdot c_i \cdot \alpha_i \cdot \prod_{j=1}^{i-1}(1-\alpha_j)$
