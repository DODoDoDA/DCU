from .Ours.ours_cifar import ours_cifar

def get_learner(model_name, args):
    name = model_name.lower()
    if name == 'ours_cifar':
        return ours_cifar(args)
    else:
        assert 0
