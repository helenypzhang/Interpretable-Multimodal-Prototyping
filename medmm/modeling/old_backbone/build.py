from medmm.utils import Registry, check_availability

PATH_BACKBONE_REGISTRY = Registry("PATH_BACKBONE")


def build_backbone(name, verbose=True, **kwargs):
    avai_backbones = PATH_BACKBONE_REGISTRY.registered_names()
    check_availability(name, avai_backbones)
    if verbose:
        print("Path Backbone: {}".format(name))
    return PATH_BACKBONE_REGISTRY.get(name)(**kwargs)
