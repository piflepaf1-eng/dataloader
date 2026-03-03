# Auto-generated dataset stubs by dino_loader.datasets.stub_gen
# Do not edit manually — run: python -m dino_loader.datasets stubs

from dino_loader.datasets.dataset import Dataset

custom: Dataset = Dataset('custom')
"""
Dataset: custom
Supported Confidentialities: private
Supported Modalities: rgb
Available Strategies: default
Available Splits: train
"""

imagenet: Dataset = Dataset('imagenet')
"""
Dataset: imagenet
Supported Confidentialities: public
Supported Modalities: rgb
Available Strategies: default
Available Splits: train, val
"""
