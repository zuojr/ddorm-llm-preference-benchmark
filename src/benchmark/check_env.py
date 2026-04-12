from __future__ import annotations

import importlib
import os
import platform
import sys


def _version(pkg: str) -> str:
    try:
        mod = importlib.import_module(pkg)
        return getattr(mod, '__version__', 'unknown')
    except Exception as e:
        return f'NOT INSTALLED ({e.__class__.__name__})'


def main() -> None:
    print('=== System ===')
    print('Python:', sys.version.replace('\n', ' '))
    print('Platform:', platform.platform())
    print('HF_HOME:', os.environ.get('HF_HOME', 'unset'))
    print('TRANSFORMERS_CACHE:', os.environ.get('TRANSFORMERS_CACHE', 'unset'))
    print('HF_DATASETS_CACHE:', os.environ.get('HF_DATASETS_CACHE', 'unset'))

    print('\n=== Packages ===')
    for pkg in ['torch', 'transformers', 'datasets', 'trl', 'peft', 'accelerate', 'bitsandbytes']:
        print(f'{pkg}:', _version(pkg))

    print('\n=== CUDA ===')
    try:
        import torch
        print('torch.cuda.is_available():', torch.cuda.is_available())
        print('torch.version.cuda:', torch.version.cuda)
        print('bf16_supported:', torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        print('device_count:', torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(i)
            total_gb = prop.total_memory / (1024 ** 3)
            print(f'GPU {i}: {prop.name} | {total_gb:.1f} GB | cc {prop.major}.{prop.minor}')
    except Exception as e:
        print('CUDA probe failed:', repr(e))


if __name__ == '__main__':
    main()
