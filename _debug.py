from b17 import *
from scipy import misc
i = NeuroImage(misc.imread('demo/1.jpg'))
j = NeuroImage(misc.imread('demo/2.jpg'))

k = i.add_from_image_using_features(j)
print i.offset
