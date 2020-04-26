import torch.nn as nn
from .dla import DLASeg
from .evolve import Evolution
from lib.utils import data_utils
from lib.utils import snake_decode
import torch


class Network(nn.Module):
    def __init__(self, num_layers, heads, head_conv=256, down_ratio=4):
        super(Network, self).__init__()

        self.dla = DLASeg('dla{}'.format(num_layers), heads,
                          pretrained=True,
                          down_ratio=down_ratio,
                          final_kernel=1,
                          last_level=5,
                          head_conv=head_conv)
        self.gcn = Evolution()

    @staticmethod
    def decode_detection(output, h, w):
        ct_hm = output['ct_hm']
        wh = output['wh']
        ct, detection = snake_decode.decode_ct_hm(torch.sigmoid(ct_hm), wh)
        detection[..., :4] = data_utils.clip_to_image(detection[..., :4], h, w)
        output.update({'ct': ct, 'detection': detection})
        return ct, detection

    def forward(self, x, batch=None):
        output, cnn_feature = self.dla(x)
        self.decode_detection(output, cnn_feature.size(2), cnn_feature.size(3))
        # with torch.no_grad():
        #     ct, detection = self.decode_detection(output, cnn_feature.size(2), cnn_feature.size(3))
        output = self.gcn(output, cnn_feature, batch)
        return output

