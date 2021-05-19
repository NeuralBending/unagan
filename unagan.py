import torch
import numpy as np
from generate import  HierarchicalGenerator, read_yaml
import os, sys
import src.training_manager as manager

feat_dim = 80
sr = 22050
hop_length = 256
z_dim = 20
z_scale_factors = [2, 2, 2, 2]
z_total_scale_factor = np.prod(z_scale_factors)

def unagan(data_type = 'singing'):
  vocoder_dir = f'models/{data_type}/vocoder/'
  vocoder_config_fp = os.path.join(vocoder_dir, 'args.yml')
  vocoder_config = read_yaml(vocoder_config_fp)

  param_fp = f'models/{data_type}/params.generator.hierarchical_with_cycle.pt'
  mean_fp = f'models/{data_type}/mean.mel.npy'
  std_fp = f'models/{data_type}/std.mel.npy'

  mean = torch.from_numpy(np.load(mean_fp)).float().view(1, feat_dim, 1)
  std = torch.from_numpy(np.load(std_fp)).float().view(1, feat_dim, 1)
  mean = mean.cuda(0)
  std = std.cuda(0)

  n_mel_channels = vocoder_config.n_mel_channels
  ngf = vocoder_config.ngf
  n_residual_layers = vocoder_config.n_residual_layers

  ## Generator
  generator = HierarchicalGenerator(n_mel_channels, z_dim, z_scale_factors)
  manager.load_model(param_fp, generator, device_id='cpu')
  generator = generator.cuda(0)
  generator.eval()
  for p in generator.parameters():
    p.requires_grad = False

  #Vocoder
  vocoder_model_dir = f'models/{data_type}/vocoder/'
  sys.path.append(vocoder_model_dir)
  import modules
  vocoder_name = 'GRUGenerator'
  MelGAN = getattr(modules, vocoder_name)
  vocoder = MelGAN(n_mel_channels, ngf, n_residual_layers)
  vocoder.eval()
  vocoder_param_fp = os.path.join(vocoder_model_dir, 'params.pt')
  vocoder.load_state_dict(torch.load(vocoder_param_fp))
  vocoder = vocoder.cuda(0)
  return generator, vocoder, mean, std
