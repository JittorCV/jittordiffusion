import argparse
import yaml
from functools import wraps
from types import SimpleNamespace


class SafeNamespace(SimpleNamespace):
    """支持安全访问的 SimpleNamespace"""
    def __getattr__(self, name):
        return None  # 不存在时返回 None


def dict_to_namespace(d: dict):
    """递归把 dict 转换为 SafeNamespace"""
    if isinstance(d, dict):
        return SafeNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(x) for x in d]
    else:
        return d


def load_config(config_path: str):
    with open(config_path, "r") as f:
        if config_path.endswith((".yaml", ".yml")):
            return yaml.safe_load(f)
        elif config_path.endswith(".json"):
            import json
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config file type: {config_path}")


def merge_config(base: dict, overrides: dict):
    """
    用 overrides 覆盖 base（递归方式）
    """
    if overrides is None:
        return base
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = merge_config(base[k], v)
        else:
            base[k] = v
    return base


def config_entry(config_arg_name="config"):
    """
    装饰器：封装 parse_args + load_config + merge_config
    :param config_arg_name: 配置文件参数在 argparse 中的名字
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            parsed_args = func(*args, **kwargs)

            overrides = vars(parsed_args).copy()
            cfg_path = overrides.pop(config_arg_name, None)

            # 先把 argparse 参数变成 dict（默认值）
            cfg = {}
            for k, v in overrides.items():
                if v is None:
                    continue
                keys = k.split(".")
                d = cfg
                for subkey in keys[:-1]:
                    d = d.setdefault(subkey, {})
                d[keys[-1]] = v

            # 如果有 config 文件，就覆盖 argparse 参数
            if cfg_path is not None:
                file_cfg = load_config(cfg_path)
                cfg = merge_config(cfg, file_cfg)

            return dict_to_namespace(cfg)
        return wrapper
    return decorator


@config_entry("cfg_file")
def example_parse_args():
    parser = argparse.ArgumentParser(description="Training Config")

    parser.add_argument("--cfg_file", type=str, help="Path to config file (optional)")
    parser.add_argument("--train.batch_size", type=int, default=1, help="Override batch size")
    parser.add_argument("--train.lr", type=float, help="Override learning rate")
    parser.add_argument("--model.name", type=str, help="Override model name")

    return parser.parse_args()


if __name__ == "__main__":
    args = example_parse_args()

    print("Final Config:")
    # 即使 rank 不存在，也不会报错，返回 None
    print(args.rank)  
    print(args.train.batch_size)
    print(args.model.hidden_dim)  # 不存在 → None
