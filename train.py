# %%
import os
import os.path
import sys
import torch.multiprocessing as mp
from argparse import ArgumentParser
from train_scripts import train_effps_plus
from train_scripts import train_effps
from train_scripts import train_effps_semantic
from train_scripts import train_effps_instance
from train_scripts import train_effps_depth
from train_scripts import train_effps_sem_depth
from train_scripts import train_effps_pan_depth

# # from ignite.contrib.handlers.param_scheduler import PiecewiseLinear


MODELS = ["EfficientPS",
          "EfficientPS_Plus",
          "EfficientPS_semantic",
          "EfficientPS_instance",
          "EfficientPS_depth",
          "EfficientPS_sem_depth",
          "EfficientPS_pan_depth"]

def get_train_loop(model_name):
    if model_name == "EfficientPS_Plus":
        return train_effps_plus.train
    if model_name == "EfficientPS":
        return train_effps.train
    if model_name == "EfficientPS_semantic":
        return train_effps_semantic.train
    if model_name == "EfficientPS_instance":
        return train_effps_instance.train
    if model_name == "EfficientPS_depth":
        return train_effps_depth.train
    if model_name == "EfficientPS_sem_depth":
        return train_effps_sem_depth.train
    if model_name == "EfficientPS_pan_depth":
        return train_effps_pan_depth.train

    
          

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--nodes', default=1, type=int)
    parser.add_argument('--local_ranks', default=0, type=int,
                        help="Node's order number in [0, num_of_nodes-1]")
    parser.add_argument('--ip_adress', type=str, required=True,
                        help='ip address of the host node')

    parser.add_argument('--model_name', type=str, required=True, help="Name of the model to train. Look up in models.py")

    parser.add_argument('--config', type=str, required=True, help="Config file from configs/")

    parser.add_argument('--fast_dev', action='store_true')

    parser.add_argument('--tune', action='store_true')
  
    
    args = parser.parse_args()

    print(args)
    if args.model_name not in MODELS:
        raise ValueError("model_name must be one of: ", MODELS)
    train_loop = get_train_loop(args.model_name)

    # add the ip address to the environment variable so it can be easily avialbale
    os.environ['MASTER_ADDR'] = args.ip_adress
    print("ip_adress is", args.ip_adress)
    os.environ['MASTER_PORT'] = '12355'
    train_loop(args)