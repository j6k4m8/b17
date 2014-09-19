import skimage.io as io
import skimage.transform as sk_transform

B17_CONST = {
    "INVALID_IMAGE":          "Empty image.",
    "UNSUPPORTED":            "Unsupported feature requested."
}

NORTH = 8
SOUTH = 2
EAST  = 6
WEST  = 4


class B17Exception(Exception):
    pass


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
            sk_transform.resize(
                self.image,
                tuple([(proportion * i) for i in self.image.shape])
            ))

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



