import os
from urllib.parse import urlsplit

from .helpers import load_checkpoint
from .registry import is_model, model_entrypoint


__all__ = ['create_model']


def create_model(
        model_name: str,
        pretrained: bool = False,
        checkpoint_path: str = '',
        **kwargs,
):
    """Following https://github.com/huggingface/pytorch-image-models/tree/main
    """
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    if not is_model(model_name):
        raise RuntimeError('Unknown model (%s)' % model_name)

    create_fn = model_entrypoint(model_name)
    model = create_fn(
        pretrained=pretrained,
        **kwargs,
    )

    if checkpoint_path:
        load_checkpoint(model, checkpoint_path)

    return model