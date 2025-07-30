import torch.nn as nn
import torch

class AvgColorLineLoss(nn.Module):
    def __init__(self):
        super(AvgColorLineLoss,self).__init__()
    def forward(self,illum_map):
        return

class FlipColorLineLoss(nn.Module):
    """
    Calculates angle between two vectors on RB plane.
    Two vectors are derived by subtraction
    of original illumination map and fliped version of that.
    """
    def __init__(self):
        super(FlipColorLineLoss,self).__init__()
        self.cos = nn.CosineSimilarity()
    def forward(self,illum_map):
        # illum_map : (b,c,w,h)
        mean_ang_error = 0

        illum_map_hflip = torch.flip(illum_map,[2])
        illum_map_vflip = torch.flip(illum_map,[3])

        v1 = (illum_map_hflip - illum_map).permute(0,2,3,1).reshape((-1,2))
        v2 = (illum_map_vflip - illum_map).permute(0,2,3,1).reshape((-1,2))

        v3 = (illum_map - illum_map_hflip).permute(0,2,3,1).reshape((-1,2))
        v4 = (illum_map_vflip - illum_map_hflip).permute(0,2,3,1).reshape((-1,2))

        v5 = (illum_map_hflip - illum_map_vflip).permute(0,2,3,1).reshape((-1,2))
        v6 = (illum_map - illum_map_vflip).permute(0,2,3,1).reshape((-1,2))

        cos_similarity_1 = self.cos(v1,v2)
        cos_similarity_2 = self.cos(v3,v4)
        cos_similarity_3 = self.cos(v5,v6)

        mean_ang_error += torch.mean(1-torch.abs(cos_similarity_1))
        mean_ang_error += torch.mean(1-torch.abs(cos_similarity_2))
        mean_ang_error += torch.mean(1-torch.abs(cos_similarity_3))

        return mean_ang_error