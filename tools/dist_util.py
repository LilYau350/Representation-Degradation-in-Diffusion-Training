"""
Helpers for distributed training.
"""

import io
import os
import socket

#import blobfile as bf
#from mpi4py import MPI
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
#GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3


# def setup_dist():
#     """
#     Setup a distributed process group.
#     """
#     if dist.is_initialized():
#         return
    
#     visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "")
#     GPUS_PER_NODE = len(visible_devices.split(",")) if visible_devices else th.cuda.device_count()

#     local_rank = int(os.getenv('LOCAL_RANK', 0))

#     if not visible_devices:
#         os.environ["CUDA_VISIBLE_DEVICES"] = f"{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}"
    
#     th.cuda.set_device(local_rank)
    
#     comm = MPI.COMM_WORLD
#     backend = "gloo" if not th.cuda.is_available() else "nccl"

#     if backend == "gloo":
#         hostname = "localhost"
#     else:
#         hostname = socket.gethostbyname(socket.getfqdn())
        
#     os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
#     os.environ["RANK"] = str(comm.rank)
#     os.environ["WORLD_SIZE"] = str(comm.size)

#     port = comm.bcast(_find_free_port(), root=0)
#     os.environ["MASTER_PORT"] = str(port)
    
#     dist.init_process_group(backend=backend, init_method="env://")

def is_main_process():
    """Helper function to check if the current process is the main one."""
    # 如果没有初始化分布式环境，直接返回 True，因为在单卡模式下，当前进程就是主进程
    if not dist.is_available() or not dist.is_initialized():
        return True
    # 在多卡分布式环境下，只有 rank 为 0 的进程是主进程
    return dist.get_rank() == 0

def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return

    # 动态获取 GPU 数量
    visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "")
    GPUS_PER_NODE = len(visible_devices.split(",")) if visible_devices else th.cuda.device_count()

    local_rank = int(os.getenv('LOCAL_RANK', 0))
    th.cuda.set_device(local_rank)

    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "127.0.0.1")  
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12345")  

    # 设置 RANK 和 WORLD_SIZE
    os.environ["RANK"] = os.getenv("RANK", str(local_rank))  
    os.environ["WORLD_SIZE"] = os.getenv("WORLD_SIZE", str(GPUS_PER_NODE))  

    # 初始化进程组，使用 nccl 进行 GPU 通信
    dist.init_process_group(backend="nccl", init_method="env://")

def cleanup_dist():
    """
    Cleanup a distributed process group.
    """
    if dist.is_initialized():
        dist.destroy_process_group()
        
def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda")
    return th.device("cpu")


# def load_state_dict(path, **kwargs):
#     """
#     Load a PyTorch file without redundant fetches across MPI ranks.
#     """
#     chunk_size = 2 ** 30  # MPI has a relatively small size limit
#     if MPI.COMM_WORLD.Get_rank() == 0:
#         with bf.BlobFile(path, "rb") as f:
#             data = f.read()
#         num_chunks = len(data) // chunk_size
#         if len(data) % chunk_size:
#             num_chunks += 1
#         MPI.COMM_WORLD.bcast(num_chunks)
#         for i in range(0, len(data), chunk_size):
#             MPI.COMM_WORLD.bcast(data[i : i + chunk_size])
#     else:
#         num_chunks = MPI.COMM_WORLD.bcast(None)
#         data = bytes()
#         for _ in range(num_chunks):
#             data += MPI.COMM_WORLD.bcast(None)

#     return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
