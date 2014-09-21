import skimage.io as io
from skimage import transform
from skimage.color import rgb2gray
from skimage.feature import ORB, match_descriptors
from skimage.measure import ransac

B17_CONST = {
    "INVALID_IMAGE":          "Empty image.",
    "UNSUPPORTED":            "Unsupported feature requested."
}

class B17Exception(Exception):
    pass

class NeuroCanvas:

    def __init__(self):
        self.canvas = []


class NeuroImage(io.Image):
    """skimage-mutant that contains all or part of a neural image."""

    def __init__(self, image):
        self.image = image
        if self.image == []:
            raise B17Exception(B17_CONST["INVALID_IMAGE"])

    def find_blobs(self):
        """
        Convert the image to grayscale, and then threshold the
        found blob-size (from skimage.blob_...). Return a blob[].
        """

    def re_res(self, proportion):
        return NeuroImage(
            transform.rescale(self.image, proportion))



    def add_from_image_using_features(self, that):
        this = rgb2gray(self.image[:, 500:500+1987, :])
        that = rgb2gray(that[:, 500:500+1987, :])

        this = transform.rescale(this, 0.25)
        that = transform.rescale(that, 0.25)​​

        orb = ORB(n_keypoints=1000, fast_threshold=0.05)

        orb.detect_and_extract(this)
        keypoints1 = orb.keypoints
        descriptors1 = orb.descriptors

        orb.detect_and_extract(that)
        keypoints2 = orb.keypoints
        descriptors2 = orb.descriptors

        matches12 = match_descriptors(descriptors1,
                                      descriptors2,
                                      cross_check=True)​​





    def match_edge(self, that, direction):
        """
        Attempts to match two images along the edge specified.
        Edge refers to the edge of the subject. So,

             >>> n.match_edge(m, EAST)

                 ------- -------
                |       |       |
                |   n   |   m   |
                |       |       |
                 ------- -------

        There's a somewhat complex process to do this in order to make
        sure we're not wasting time:

        1   Lower the resolution of both images. Now, overlay the images
            with full overlap (i.e. diagram above).
        2   Subtract the first from the second, and establish a net
            difference along the overlap.
        3   Move the second image down by one voxel.
        4   Minimize the difference of the two pictures.
                        -------
                       |       |
                       |   m   |
                 ------|       |
                |       -------
                |   n   |  ^
                |       |  v
                 -------
        5   The global minimum of this procedure is the 'match'.
        6   Increase the resolution, and resume this process, now only
            moving the distance of 1.5 voxels from optimal low-res position.

        In very high-resolution images, this process can be recursed with
        even lower resolution, so that the starting voxel is 25-percent the
        size of the initial image, the second round is 50, and finally 100.
        """

        if direction == EAST:
            return match_east_edge(that)
        else:
            raise B17Exception(B17_CONST["UNSUPPORTED"])


    def match_east_edge(self, that):
        """
        Matches the east edge of `this` with the west edge of `that`.
        """
        overlap = [0, 0]

