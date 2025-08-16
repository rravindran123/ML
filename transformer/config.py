from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 30,
        "lr": 1e-4,
        "seqlen": 350,
        "d_model": 512,
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload":22,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
        "datasource": '.'
    }

def get_weights_file_path(config, epoch):
    """
    Get the file path for saving or loading model weights.
    
    Args:
        config: Configuration object containing model settings.
        epoch: The epoch number for which the weights are being saved or loaded.
    
    Returns:
        str: The file path for the model weights.
    """
    print("here --- getting file path: ")
    model_folder = f"{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    path= str(Path('.') / model_folder / model_filename)
    print(f"here --- Model path: {path}")
    return path

def get_total_parmeters(model):
    """
    Calculate the total number of parameters in the model.
    
    Args:
        model: The model for which to calculate the parameters.
        
    Returns:
        int: The total number of parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def latest_weights_file_path(config):
    model_folder = f"{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])