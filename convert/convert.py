import os
import torch
from collections import OrderedDict
from transformers import AutoConfig
from models.modeling_hunyuan import HunYuanForCausalLM
from models.configuration_hunyuan import HunYuanConfig

def convert_hunyuan_checkpoint(checkpoint_dir, output_dir):
    """
    Convert HunYuan checkpoint to HuggingFace format
    
    Args:
        checkpoint_dir: Directory containing original .bin files
        output_dir: Output directory for converted model
    """
    # Load the config
    config = HunYuanConfig.from_json_file(os.path.join(checkpoint_dir, "config.json"))
    
    # Initialize model with config
    model = HunYuanForCausalLM(config)
    
    # Load state dict
    num_shards = 80  # Based on file count
    state_dict = OrderedDict()
    
    # Load each shard
    for i in range(1, num_shards + 1):
        shard_file = f"pytorch_model-{i:05d}-of-{num_shards:05d}.bin"
        shard_path = os.path.join(checkpoint_dir, shard_file)
        shard_state = torch.load(shard_path, map_location="cpu")
        state_dict.update(shard_state)

    # Load weights into model
    model.load_state_dict(state_dict)
    
    # Save in HF format
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    
    # Copy other required files
    import shutil
    files_to_copy = [
        "tokenizer_config.json",
        "hy.tiktoken", 
        "configuration_hunyuan.py",
        "modeling_hunyuan.py",
        "tokenization_hy.py"
    ]
    for file in files_to_copy:
        src = os.path.join(checkpoint_dir, file)
        dst = os.path.join(output_dir, file)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            
    print(f"Model converted and saved to {output_dir}")

if __name__ == "__main__":
    checkpoint_dir = "."  # Directory containing the .bin files
    output_dir = "hunyuan-hf"
    convert_hunyuan_checkpoint(checkpoint_dir, output_dir)