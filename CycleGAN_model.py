import torch.nn as nn
from Base_model import BaseModel
import Network

class CycleGanModel(BaseModel):
    def __init__(self, option):
        BaseModel.__init__(self, option)    
        
    def forward(self, x):
        pass