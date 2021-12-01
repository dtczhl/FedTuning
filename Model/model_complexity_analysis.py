
import numpy as np
import torch.cuda

from Model.ResNet import ResNet
from Model.VGG import VGG
from Dataset.speech_command import SPEECH_COMMAND_N_CLASS, SPEECH_COMMAND_N_INPUT_FEATURE

from ptflops import get_model_complexity_info

# depth_arr = [10, 18, 26, 34] # resnet
depth_arr = [11, 13, 16, 19]  # vgg
macs_arr = np.zeros(len(depth_arr))
params_arr = np.zeros(len(depth_arr))

with torch.cuda.device(0):

    for i in range(len(depth_arr)):

        depth = depth_arr[i]

        net = VGG(num_input_feature=SPEECH_COMMAND_N_INPUT_FEATURE, depth=depth, num_classes=SPEECH_COMMAND_N_CLASS)
        macs, params = get_model_complexity_info(net, (1, 32, 32), as_strings=False,
                                                 print_per_layer_stat=False, verbose=True)

        # print('{:<30}  {}'.format('Computational complexity: ', macs))
        # print('{:<30}  {}'.format('Number of parameters: ', params))

        macs_arr[i] = macs
        params_arr[i] = params

np.set_printoptions(suppress=True, formatter={'float_kind': '{:0.2f}'.format})
print('{:<20}'.format('Depth Layers: '), depth_arr)
print('{:<20}'.format('FLOPs: '), macs_arr)
print('{:<20}'.format('Params: '), params_arr)