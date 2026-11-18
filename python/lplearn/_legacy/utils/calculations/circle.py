import numpy as np


def area_from_radius(radius):
    '''Calculate the area of a circle given its radius.'''
    return np.pi * radius ** 2


def radius_from_area(area):
    '''Calculate the radius of a circle given its area.'''
    return np.sqrt(area / np.pi)
