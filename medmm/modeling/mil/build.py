from medmm.utils import Registry, check_availability

MIL_REGISTRY = Registry("MIL")


def build_mil(name, verbose=True, **kwargs):
    avai_mils = MIL_REGISTRY.registered_names()
    check_availability(name, avai_mils)
    if verbose:
        print("Multi-instance method: {}".format(name))
    return MIL_REGISTRY.get(name)(**kwargs)
