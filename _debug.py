from b17 import *
from scipy import misc
i = NeuroImage(misc.imread('demo/prematch0/match0/21.TIF'))
j = NeuroImage(misc.imread('demo/prematch0/match0/20.TIF'))
# i = NeuroImage(misc.imread('demo/demo0/demo1-0.jpg'))
# j = NeuroImage(misc.imread('demo/demo0/demo2-0.jpg'))


for x in xrange(0,1):
    k = j.add_from_image_using_features(i)
    print str(x) + "\t" + str(k.image.shape[1] - i.image.shape[1])
    io.imshow(k)
    io.show()
