def subclasses(cls, depth: int = -1, strict: bool = True) -> list[type]:
    """Recursively get all subclasses of a class, to a given inheritance depth.

    Args:
        depth (int, optional): Sets the inheritance depth. Defaults to -1,
            which means all subclasses.
        strict (bool, optional): If True, cls itself is excluded from the results.

    Returns:
        list[type]: An list of subclasses.
    """
    all = []
    if not strict:
        all.append(cls)
    if depth != 0:
        subs = cls.__subclasses__()
        all.extend(subs)
        for sub in subs:
            all.extend(subclasses(sub, depth=depth - 1))
    return all
