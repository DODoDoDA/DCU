from .Ours.ours_cifar import ours_cifar
from .Ours.ours_mnist import ours_mnist


def get_learner(model_name, args):
    name = model_name.lower()
    if name == 'ours_mnist':
        return ours_mnist(args)
    elif name == 'ours_cifar':
        return ours_cifar(args)
    else:
        assert 0
