# Differentiable RenderLayer

Renders geometry and texture into an image viewed from provided camera pose.
Includes support for rendering on differentiable background images and
boundary-aware propagation of gradients w.r.t. vertices across depth
discontinuities.

# Simple Example
```python
import torch as th
import numpy as np
import cv2

from drtk.renderlayer import RenderLayer

b = 3
h = 1024
w = 1024

# Create camera pose. These matrices define 3 views, one straight-ahead and two
# from the left/right.
camrot = th.cat([th.eye(3)[None].cuda(),
    th.from_numpy(cv2.Rodrigues(np.array([0, np.pi/6, 0]))[0]).float().cuda()[None],
    th.from_numpy(cv2.Rodrigues(np.array([0, -np.pi/6, 0]))[0]).float().cuda()[None]], dim=0)
campos = th.cat([th.FloatTensor([0, 0, -5])[None].cuda(),
                 th.FloatTensor([0, 0, -5])[None].cuda(),
                 th.FloatTensor([0, 0, -5])[None].cuda()], dim=0)
focal = th.FloatTensor([[w/2, 0],
                        [0, h/2]])[None].cuda().expand(b, -1, -1)
princpt = th.FloatTensor([w/2, h/2])[None].cuda().expand(b, -1)

# Create a simple texture which is half white and half pink texels.
tex = 255*th.ones(b, 3, 512, 512).cuda()
tex[:, 1, :256] = 0

# Define vertex and index arrays for a square at the origin, facing the camera.
v = th.FloatTensor([[-1, 1, 0],
                    [1, 1, 0],
                    [1, -1, 0],
                    [-1, -1, 0],
                    ])[None].cuda()
v = v.expand(b, -1, -1)
vt = th.FloatTensor([[0.2, 0.2],
                     [0.8, 0.2],
                     [0.8, 0.8],
                     [0.2, 0.8],
                     ])
vi = th.IntTensor([[0, 1, 2],
                   [2, 3, 0],
                   ])
vti = th.IntTensor([[0, 1, 2],
                    [2, 3, 0],
                    ])

# Create a random background image.
bg_img = 255*th.rand(b, 3, h, w).cuda()

# Create the renderlayer module.
rl = RenderLayer(h, w, vt, vi, vti, boundary_aware=True).cuda()

# Render the views and save the resulting images.
output = rl(v, tex, campos, camrot, focal, princpt, background=bg_img, ksize=3)
render = output["render"]

views = th.cat([render[0], render[1], render[2]], dim=1)
cv2.imwrite("views.png", views.data.cpu().numpy().transpose(1,2,0)[..., ::-1])
```
