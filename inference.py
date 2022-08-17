# %%
import os
import os.path
import sys
import torch.multiprocessing as mp
from argparse import ArgumentParser
from inference_scripts import inference_effps
from inference_scripts import inference_effps_instance

# # from ignite.contrib.handlers.param_scheduler import PiecewiseLinear

MODELS = ["EfficientPS", "EfficientPS_instance"]

def get_inference_loop(model_name):
    if model_name == "EfficientPS":
        return inference_effps.inference
    if model_name == "EfficientPS_instance":
        return inference_effps_instance.inference
    
          

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--nodes', default=1, type=int)
    parser.add_argument('--local_ranks', default=0, type=int,
                        help="Node's order number in [0, num_of_nodes-1]")
    parser.add_argument('--ip_adress', type=str, required=True,
                        help='ip address of the host node')

    parser.add_argument('--ngpus', default=4, type=int,
                        help='number of gpus per node')

    parser.add_argument('--model_name', type=str, required=True, help="Name of the model to train. Look up in models.py")

    parser.add_argument('--config', type=str, required=True, help="Config file from configs/")
    
    parser.add_argument('--fast_dev', action='store_true')

    parser.add_argument('--tune', action='store_true')
    
    args = parser.parse_args()

    print(args)
    if args.model_name not in MODELS:
        raise ValueError("model_name must be one of: ", MODELS)
    inference_loop = get_inference_loop(args.model_name)

    # Total number of gpus availabe to us.
    args.world_size = args.ngpus * args.nodes
    # add the ip address to the environment variable so it can be easily avialbale
    os.environ['MASTER_ADDR'] = args.ip_adress
    print("ip_adress is", args.ip_adress)
    os.environ['MASTER_PORT'] = '12355'
    os.environ['WORLD_SIZE'] = str(args.world_size)
    # nprocs: number of process which is equal to args.ngpu here
    inference_loop(args)
    # mp.spawn(train_loop, nprocs=args.ngpus, args=(args,))