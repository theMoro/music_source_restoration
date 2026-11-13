import torch

def load_ckpt(path, model):
    """
    Load checkpoint into model.
    
    Args:
        path: Path to checkpoint file
        model: Model or LightningModule to load checkpoint into
    Returns:
        dict: Full checkpoint dictionary (includes EMA state if present)
    """
    full_ckpt = torch.load(path, weights_only=False, map_location='cpu')
    model_ckpt = full_ckpt['state_dict']
    if set(model.state_dict().keys()) != set(model_ckpt.keys()): # remove prefix, incase the ckpt is of lightning module
        # one_model_key = next(iter(model.state_dict().keys()))
        for one_model_key in list(model.state_dict().keys()):
            if "separation_model" in one_model_key:
                ckpt_corresponding_key = [k for k in model_ckpt.keys() if k.endswith(one_model_key)]
                if ckpt_corresponding_key:
                    prefix = ckpt_corresponding_key[0][:-len(one_model_key)]
                    model_ckpt = {k[len(prefix):]: v for k, v in model_ckpt.items() if k.startswith(prefix)  }  # remove prefix
                    break
    
    model_state = model.state_dict()

    # Only load keys that exist in the model (missing keys keep their initialization)
    filtered_ckpt = {}
    missing_keys = []
    unexpected_keys = []
    
    for key in model_state.keys():
        if key in model_ckpt:
            filtered_ckpt[key] = model_ckpt[key]
        else:
            missing_keys.append(key)
    
    for key in model_ckpt.keys():
        if key not in model_state:
            unexpected_keys.append(key)
    
    if missing_keys:
        print(f"Warning: {len(missing_keys)} keys missing from checkpoint")
    if unexpected_keys:
        print("-" * 30)
        print(f"Warning: {len(unexpected_keys)} unexpected keys in checkpoint (will be ignored)")
        print("unexpected keys found in checkpoint:")
        for key in unexpected_keys:
            print(key)
        print("-" * 30)
    
    # Load with strict=False to allow missing parameters
    model.load_state_dict(filtered_ckpt, strict=False)
    
    return full_ckpt