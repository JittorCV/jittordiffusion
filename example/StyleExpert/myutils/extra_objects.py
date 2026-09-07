import os
import torch.nn as nn
from jittor_runtime import load_safetensors

class ExtraModules(nn.Module):
    def __init__(self, **modules_dict):
        super().__init__()
        self._modules_dict = {}
        if modules_dict:
            for name, module in modules_dict.items():
                self.add_module(name, module)
                self._modules_dict[name] = module

    def add_modules(self, **modules_dict):
        if modules_dict:
            for name, module in modules_dict.items():
                self.add_module(name, module)
                self._modules_dict[name] = module

    def __getattr__(self, name):
        if '_modules_dict' in self.__dict__ and name in self._modules_dict:
            return self._modules_dict[name]
        return super().__getattr__(name)
    
    def get_module(self, name):
        return self._modules_dict.get(name, None)
    
    def get_unwrap_dict(self, accelerator):
        unwrap = accelerator.unwrap_model
        return {name: unwrap(module) for name, module in self._modules_dict.items()}

    def forward(self):
        pass

    # 🧩 新增方法：自动加载所有模块的权重
    def load_pretrained(self, dir_path, suffix=".safetensors", strict=True, verbose=True):
        """
        自动从给定目录加载 _modules_dict 中的模块。
        文件命名需与模块名一致，如 con_encoder.safetensors。
        """
        loaded, missing = [], []
        for name, module in self._modules_dict.items():
            path = os.path.join(dir_path, f"{name}{suffix}")
            if os.path.exists(path):
                try:
                    state = load_safetensors(path)
                    module.load_state_dict(state, strict=strict)
                    loaded.append(name)
                    if verbose:
                        print(f"[ExtraModules] Loaded pretrained weights for '{name}' from {path}")
                except Exception as e:
                    print(f"[ExtraModules] Failed to load '{name}': {e}")
                    missing.append(name)
            else:
                if verbose:
                    print(f"[ExtraModules] No checkpoint found for '{name}' at {path}")
                missing.append(name)
        return {"loaded": loaded, "missing": missing}


class ExtraItems:
    def __init__(self, **object_dict):
        super().__init__()
        self._objects_dict = {}  # 用于存普通对象

        if object_dict:
            for name, obj in object_dict.items():
                self.add_item(name, obj)

    def add_item(self, name, obj):
        self._objects_dict[name] = obj

    def add_items(self, **items_dict):
        for name, obj in items_dict.items():
            self.add_item(name, obj)

    def __getattr__(self, name):
        # 再返回普通对象
        if '_objects_dict' in self.__dict__ and name in self._objects_dict:
            return self._objects_dict[name]
        return None
    
