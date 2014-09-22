from b17 import *
from scipy import misc
i = NeuroImage(misc.imread('demo/3.jpg'))
j = NeuroImage(misc.imread('demo/4.jpg'))

k = i.add_from_image_using_features(j)
