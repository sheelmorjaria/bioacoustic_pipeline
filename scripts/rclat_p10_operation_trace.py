"""Record selected ATen operand metadata without retaining tensors."""
import json
from pathlib import Path
import torch
from torch.utils._python_dispatch import TorchDispatchMode

class FamilyTrace(TorchDispatchMode):
    def __init__(self):
        super().__init__();self.recipes=[];self.lookup={};self.sequence=[]
    def encode(self,x):
        if isinstance(x,torch.Tensor):
            if x.dtype!=torch.float32:raise ValueError('Unexpected replay dtype')
            return {'tensor':True,'shape':list(x.shape),'stride':list(x.stride())}
        if isinstance(x,(tuple,list)):return [self.encode(v) for v in x]
        if isinstance(x,dict):return {k:self.encode(v) for k,v in x.items()}
        if x is None or isinstance(x,(int,float,bool,str)):return x
        raise TypeError(type(x))
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        kwargs=kwargs or {};name=str(func)
        selected=name in ['aten.convolution.default','aten.mm.default','aten.bmm.default','aten.addmm.default']
        if selected:
            recipe=dict(op=name,family='convolution' if 'convolution' in name else 'matrix',args=self.encode(args),kwargs=self.encode(kwargs))
            key=json.dumps(recipe,sort_keys=True)
            if key not in self.lookup:self.lookup[key]=len(self.recipes);self.recipes.append(recipe)
            self.sequence.append(self.lookup[key])
        return func(*args,**kwargs)
    def save(self,path):
        Path(path).write_text(json.dumps(dict(recipes=self.recipes,sequence=self.sequence,limitation='Only selected ATen calls; no tensor data, identities, storage offsets or internal cache keys'),separators=(',',':'))+'\n')
