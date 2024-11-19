# 相机模型 Camera Model

相机模型是一个抽象的数学模型，建立了三维欧式空间点和相机图像像素间的关系。相机模型的文章有很多，这里针对自己的理解和需求，记下一些笔记和可视化程序助于理解。



## Notation
相机图像中的像素坐标使用小写的$(x,y)$表示，相机坐标系下的点使用$(X_c, Y_c, Z_c)$表示，世界坐标系下的点使用$(X_w, Y_w, Z_w)$表示，相机的标定焦距使用$f$表示

## Camera Model
从理想的针孔相机(Pinhole camera)模型开始，图片，利用相似三角形原理可以得到相机坐标系（三维欧式空间点）到相机图像的投影：

$$x=f \frac{X_{c}}{Z_{c}} + c_x， y=f \frac{Y_{c}}{Z_{c}} + c_y$$

可以写成矩阵的形式：

$$
\begin{bmatrix}
    x \\
    y
\end{bmatrix}
$$

这样的投影不方便计算，为此，引入一个新的坐标表示法，叫做齐次坐标系(homo coordinate system)，我们只需要在任何一个坐标点的最后一维加上1，就会变成齐次坐标系表示了，例如像素坐标点$(x,y)$加上1成为$(x,y,1)$就是齐次坐标系。
> 齐次坐标系有一个特性，对任意的$w$，$(wx, wy, w)$都表示的是同一个像素坐标点，那如果想要求任意齐次坐标系$(x, y, w)$对应的空间中点，需要对最后一维归一才可以，即$(x/w, y/w, 1)$。

> 另外，需要注意的是齐次坐标系只是一种坐标表示法，并不代表对应真正的空间，例如$(x,y,1)$还是二维欧式空间中的点，并不是三维欧式空间中的点。

我们将上面的公示使用齐次坐标系表示，并从左向右推导：
$$
\begin{bmatrix}
    x \\
    y \\
    1
\end{bmatrix}
=
\begin{bmatrix}
    f \frac{X_{c}}{Z_{c}} + c_x\\
    f \frac{Y_{c}}{Z_{c}} + c_y \\
    1
\end{bmatrix}
=
\begin{bmatrix}
    fX_c + Z_c c_x\\
    fY_c + Z_c c_y \\
    Z_c
\end{bmatrix}
$$
注意对于齐次坐标表示，对每个值乘以$Z_c$还是同一个点，接下来分解成矩阵和向量线性乘积：
$$
\begin{bmatrix}
    x \\
    y \\
    1
\end{bmatrix}
=
\begin{bmatrix}
    fX_c + Z_c c_x\\
    fY_c + Z_c c_y \\
    Z_c
\end{bmatrix}
=
\begin{bmatrix}
    f & 0 & c_x \\
0 & f & c_y \\
    0 & 0 & 1
\end{bmatrix}
\cdot
\begin{bmatrix}
    X_c \\
    Y_c \\
    Z_c
\end{bmatrix}
$$
由于像素坐标是由齐次坐标系表示的，对应的相机坐标系也由齐次坐标系表示，则有：
$$
\begin{bmatrix}
    x \\
    y \\
    1
\end{bmatrix}
=
\begin{bmatrix}
    f & 0 & c_x & 0\\
    0 & f & c_y & 0\\
    0 & 0 & 1   & 0
\end{bmatrix}
\cdot
\begin{bmatrix}
    X_c \\
    Y_c \\
    Z_c \\
    1
\end{bmatrix}
$$
左侧的$3\times3$矩阵 $\begin{bmatrix}
    f & 0 & c_x \\
    0 & f & c_y \\
    0 & 0 & 1   
\end{bmatrix}$，称为相机的内参(intrinsics)矩阵。