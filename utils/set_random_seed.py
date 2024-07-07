import torch

def set_random_seed(seed):
    # Set the random seed for CPU
    torch.manual_seed(seed)

    # Set the random seed for all GPUs (if using CUDA)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

        # Enable deterministic algorithms for cuDNN operations
        torch.backends.cudnn.deterministic = True

        # Disable cuDNN benchmark mode
        torch.backends.cudnn.benchmark = False