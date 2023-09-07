def get_random_ingredients(kind=None):
    """
    Return a list of random ingredients as cheese strings.

    :param kind: Optional "kind" of ingredients.
    :type kind: list[str] or None
    :raise lumache.InvalidKindError: If the kind is invalid.
    :return: The ingredients list.
    :rtype: list[str]

    """
    return ["shells", "gorgonzola", "parsley"]


from numba import jit
@jit(nopython=True)
def principal_axes(theta,phi,psi):
    """
    Return a list of very principled ingredients as cheese strings.

    :param kind: Optional "kind" of ingredients.
    :type kind: list[str] or None
    :raise lumache.InvalidKindError: If the kind is invalid.
    :return: The ingredients list.
    :rtype: list[str]

    """
    
    m1 = sin(phi)*cos(psi) - sin(psi)*cos(phi)*cos(theta)
    m2 = -(cos(phi)*cos(psi) + sin(psi)*sin(phi)*cos(theta))
    m3 = sin(psi)*sin(theta)
    m = [m1,m2,m3]

    n1 = -sin(phi)*sin(psi) - cos(psi)*cos(phi)*cos(theta)
    n2 = cos(phi)*sin(psi) - cos(psi)*sin(phi)*cos(theta)
    n3 = cos(psi)*sin(theta)
    n = [n1,n2,n3]

    return m,n