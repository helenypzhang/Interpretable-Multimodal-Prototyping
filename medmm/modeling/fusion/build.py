from medmm.utils import Registry, check_availability

FUSION_REGISTRY = Registry("FUSION")


def build_fusion(name, verbose=True, **kwargs):
    avai_mils = FUSION_REGISTRY.registered_names()
    check_availability(name, avai_mils)
    if verbose:
        print("Fusion method: {}".format(name))
    return FUSION_REGISTRY.get(name)(**kwargs)
