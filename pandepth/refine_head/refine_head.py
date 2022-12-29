import torch
from torch import nn
import torch.nn.functional as F
from inplace_abn import InPlaceABN

class RefineHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # self.out_conv_depth = nn.Conv2d(num_classes + 1, num_classes, kernel_size=1)

        self.out_conv_depth_1 = nn.Sequential(
            nn.Conv2d(num_classes+1, 256, kernel_size=3),
            InPlaceABN(256)
        )

        self.out_conv_depth_2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3),
            InPlaceABN(256)
        )

        self.out_conv_depth_3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3),
            InPlaceABN(256)
        )

        self.out_conv_depth_4 = nn.Sequential(
            nn.Conv2d(256, num_classes, kernel_size=1),
            InPlaceABN(num_classes)
        )

        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, x, depth_map, output_size, semantic_gt=None):
       

        #----------------------------
        x = torch.cat((x, depth_map), dim=1)

        x = F.leaky_relu(self.out_conv_depth_1(x))
        x = F.leaky_relu(self.out_conv_depth_2(x))

        x = F.leaky_relu(self.out_conv_depth_3(x))
        x = F.leaky_relu(self.out_conv_depth_4(x))

        x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)

        if semantic_gt is not None:
            return x, self.loss(x, semantic_gt)
        else:
            return x, {}
    
    def loss(self, inputs, targets):

        refine_loss = self.cross_entropy_loss(inputs, targets)

        return {
            "refine_loss": refine_loss
        }

