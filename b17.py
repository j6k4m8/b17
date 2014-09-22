import skimage.io as io
from skimage import transform
from skimage.color import rgb2gray
from skimage.feature import ORB, match_descriptors
from skimage.measure import ransac
import numpy as np

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
        that = transform.rescale(that, 0.25)

        orb = ORB(n_keypoints=10000, fast_threshold=0.05)

        orb.detect_and_extract(this)
        keypoints1 = orb.keypoints
        descriptors1 = orb.descriptors

        orb.detect_and_extract(that)
        keypoints2 = orb.keypoints
        descriptors2 = orb.descriptors

        matches12 = match_descriptors(descriptors1,
                                    descriptors2,
                                    cross_check = True)

        src = keypoints2[matches12[:, 1]][:, ::-1]
        dst = keypoints1[matches12[:, 0]][:, ::-1]

        model_robust, inliers = \
            ransac((src, dst), transform.ProjectiveTransform,
                    min_samples=4, residual_threshold=2)

        r, c = that.shape[:2]

        # Note that transformations take coordinates in
        # (x, y) format, not (row, column), in order to be
        # consistent with most literature.
        corners = np.array([[0, 0],
                            [0, r],
                            [c, 0],
                            [c, r]])

        # Warp the image corners to their new positions.
        warped_corners = model_robust(corners)

        # Find the extents of both the reference image and
        # the warped target image.
        all_corners = np.vstack((warped_corners, corners))

        corner_min = np.min(all_corners, axis=0)
        corner_max = np.max(all_corners, axis=0)

        output_shape = (corner_max - corner_min)
        output_shape = np.ceil(output_shape[::-1])

        from skimage.color import gray2rgb
        from skimage.exposure import rescale_intensity
        from skimage.transform import warp
        from skimage.transform import SimilarityTransform

        offset = SimilarityTransform(translation=-corner_min)

        this_ = warp(this, offset.inverse,
                       output_shape=output_shape, cval=-1)

        that_ = warp(that, (model_robust + offset).inverse,
                        output_shape = output_shape, cval = -1)

        def add_alpha(image, background=-1):
            """Add an alpha layer to the image.

            The alpha layer is set to 1 for foreground
            and 0 for background.
            """
            rgb = gray2rgb(image)
            alpha = (image != background)
            return np.dstack((rgb, alpha))

        this_alpha = add_alpha(this_)
        that_alpha = add_alpha(that_)

        merged = (this_alpha + that_alpha)
        alpha = merged[..., 3]

        # The summed alpha layers give us an indication of
        # how many images were combined to make up each
        # pixel.  Divide by the number of images to get
        # an average.
        merged /= np.maximum(alpha, 1)[..., np.newaxis]
        io.imshow(merged)
        io.show()
        return merged





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

