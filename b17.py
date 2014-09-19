import skimage.io as io
import skimage.transform as sk_transform

B17_CONST = {
    "INVALID_IMAGE":          "Empty image."
}


class B17Exception(Exception):
    pass


class NeuroImage(io.Image):
    """skimage-mutant that contains all or part of a neural image."""

    def __init__(self, image):
        self.image = image
        if self.image == []:
            raise B17Exception(B17_CONST.INVALID_IMAGE)

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
