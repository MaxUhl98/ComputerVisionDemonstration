import torch
from demo_configuration import DemonstrationConfig
from classification.demonstrate import demonstrate

if __name__ == '__main__':
    torch.set_default_device('cpu')
    cfg = DemonstrationConfig()
    demonstrate(cfg)





