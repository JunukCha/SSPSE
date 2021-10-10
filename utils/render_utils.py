import torch
import numpy as np
import neural_renderer as nr
import config

# from models import SMPL
from lib.models.smpl import get_smpl_faces

class IMGRenderer():
    """Renderer used to render segmentation masks and part segmentations.
    Internally it uses the Neural 3D Mesh Renderer
    """
    def __init__(self, focal_length=5000., render_res=224):
        # Parameters for rendering
        self.focal_length = focal_length
        self.render_res = render_res
        # We use Neural 3D mesh renderer for rendering masks and part segmentations
        self.neural_renderer = nr.Renderer(dist_coeffs=None, orig_size=self.render_res,
                                           image_size=render_res,
                                           light_intensity_ambient=1,
                                           light_intensity_directional=0,
                                           anti_aliasing=False)
        self.faces = torch.from_numpy(get_smpl_faces().astype(np.int32)).cuda()

    def __call__(self, vertices, camera, textures):
        """Wrapper function for rendering process."""
        # Estimate camera parameters given a fixed focal length
        cam_t = torch.stack([camera[:,1], camera[:,2], 2*self.focal_length/(self.render_res * camera[:,0] +1e-9)],dim=-1)
        batch_size = vertices.shape[0]
        K = torch.eye(3, device=vertices.device)
        K[0,0] = self.focal_length 
        K[1,1] = self.focal_length 
        K[2,2] = 1
        K[0,2] = self.render_res / 2.
        K[1,2] = self.render_res / 2.
        K = K[None, :, :].expand(batch_size, -1, -1)
        R = torch.eye(3, device=vertices.device)[None, :, :].expand(batch_size, -1, -1)
        faces = self.faces[None, :, :].expand(batch_size, -1, -1)
        parts, _, mask =  self.neural_renderer(vertices, faces, textures=textures.expand(batch_size, -1, -1, -1, -1, -1), K=K, R=R, t=cam_t.unsqueeze(1))
        return mask, parts