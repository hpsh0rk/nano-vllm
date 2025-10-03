import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    """
    Load model weights from safetensors files into a PyTorch model.

    This function loads pre-trained weights from .safetensors files located in the specified
    directory into the given PyTorch model. It handles both regular parameter loading and
    packed/sharded parameter loading through custom weight loaders.

    Args:
        model (nn.Module): The PyTorch model to load weights into
        path (str): Directory path containing .safetensors files

    Process:
        1. Retrieves packed module mappings from model attributes (if available)
        2. Iterates through all .safetensors files in the directory
        3. For each weight tensor in the file:
           - Checks if it's part of a packed module (sharded parameter)
           - If packed: uses custom weight loader with shard_id
           - If regular: uses default weight loader
    """
    # Get mapping of packed module names to their corresponding parameter names and shard IDs
    # This allows handling of sharded parameters (e.g., when a large weight matrix is split across files)
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})

    # Iterate through all safetensors files in the specified directory
    for file in glob(os.path.join(path, "*.safetensors")):
        # Open the safetensors file in read mode, using CPU as target device
        with safe_open(file, "pt", "cpu") as f:
            # Process each weight tensor in the current file
            for weight_name in f.keys():
                # Check if this weight belongs to a packed module
                for packed_key in packed_modules_mapping:
                    if packed_key in weight_name:
                        # This is a sharded parameter - get the actual parameter name and shard ID
                        actual_param_name, shard_id = packed_modules_mapping[packed_key]
                        param_name = weight_name.replace(packed_key, actual_param_name)

                        # Get the target parameter from the model
                        param = model.get_parameter(param_name)

                        # Use the custom weight loader to handle sharded loading
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    # This is a regular (non-sharded) parameter
                    param = model.get_parameter(weight_name)

                    # Use custom weight loader if available, otherwise use default
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
