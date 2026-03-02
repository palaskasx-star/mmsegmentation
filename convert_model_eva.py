import torch

# 1. Path to your raw checkpoint
input_path = 'pretrained/dinov3_tiny_gaussian_cls/checkpoint_epoch_300.pth'
output_path = 'pretrained/dinov3_tiny_gaussian_cls/checkpoint_epoch_300_mmseg.pth'

# 2. Load the full checkpoint
checkpoint = torch.load(input_path, map_location='cpu')

# 3. Extract just the weights (your log shows they are under the 'model' key)
state_dict = checkpoint.get('model', checkpoint)

# 4. Create a new dictionary with the correct MMSegmentation prefix
new_state_dict = {}
for key, value in state_dict.items():
    # Remove any prefixes your custom training code might have added 
    # (e.g., if your keys look like 'backbone.blocks.0...', this cleans them)
    clean_key = key.replace('backbone.', '').replace('module.', '').replace('encoder.', '')
    
    # Add the prefix that TIMMBackbone is explicitly asking for in the logs
    new_key = f'timm_model.model.{clean_key}'
    new_state_dict[new_key] = value

# 5. Save the clean, perfectly formatted weights!
torch.save(new_state_dict, output_path)
print(f"✅ Successfully converted and saved to: {output_path}")