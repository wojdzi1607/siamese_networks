import numpy as np
import scipy.ndimage as ndi
from skimage import exposure
import random
import tensorflow as tf
import albumentations as A
import cv2



class ImageAugmentor:
    """Class that performs image augmentation.

    Big part of this code uses Keras ImageDataGenerator file code. I just reorganized it
    in this class

    Attributes:
        augmentation_probability: probability of augmentation
        shear_range: shear intensity (shear angle in degrees).
        rotation_range: degrees (0 to 180).
        shift_range: fraction of total shift (horizontal and vertical).
        zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
            in the range [1-z, 1+z]. A sequence of two can be passed instead
            to select this range.
    """

    def __init__(self, augmentation_probability, shear_range, rotation_range, shift_range, zoom_range):
        """Inits ImageAugmentor with the provided values for the attributes."""
        self.augmentation_probability = augmentation_probability
        self.shear_range = shear_range
        self.rotation_range = rotation_range
        self.shift_range = shift_range
        self.zoom_range = zoom_range
        self.contrast_stretching = True, #####
        self.red_range = (0.8, 1.2)
        self.green_range = None
        self.blue_range = None

    def __transform_matrix_offset_center(self, transformation_matrix, width, height):
        """ Corrects the offset of tranformation matrix

            Corrects the offset of tranformation matrix for the specified image
            dimensions by considering the center of the image as the central point

            Args:
                transformation_matrix: transformation matrix from a specific
                    augmentation.
                width: image width
                height: image height

            Returns:
                The corrected transformation matrix.
        """

        o_x = float(width) / 2 + 0.5
        o_y = float(height) / 2 + 0.5
        offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
        reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
        transformation_matrix = np.dot(
            np.dot(offset_matrix, transformation_matrix), reset_matrix)

        return transformation_matrix

    # Applies a provided transformation to the image
    def __apply_transform(self, image, transformation_matrix):
        """ Applies a provided transformation to the image

            Args:
                image: image to be augmented
                transformation_matrix: transformation matrix from a specific
                    augmentation.

            Returns:
                The transformed image
        """

        channel_axis = 2
        image = np.rollaxis(image, channel_axis, 0)
        final_affine_matrix = transformation_matrix[:2, :2]
        final_offset = transformation_matrix[:2, 2]

        channel_images = [ndi.interpolation.affine_transform(
            image_channel,
            final_affine_matrix,
            final_offset,
            order=0,
            mode='nearest',
            cval=0) for image_channel in image]

        image = np.stack(channel_images, axis=0)
        image = np.rollaxis(image, 0, channel_axis + 1)

        return image

    def __perform_random_rotation(self, image):
        """ Applies a random rotation

            Args:
                image: image to be augmented

            Returns:
                The transformed image
        """

        theta = np.deg2rad(np.random.uniform(
            low=self.rotation_range[0], high=self.rotation_range[1]))

        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])

        transformation_matrix = self.__transform_matrix_offset_center(
            rotation_matrix, image.shape[0], image.shape[1])
        image = self.__apply_transform(image, transformation_matrix)

        return image

    def __perform_random_shear(self, image):
        """ Applies a random shear

            Args:
                image: image to be augmented

            Returns:
                The transformed image
        """

        shear = np.deg2rad(np.random.uniform(
            low=self.shear_range[0], high=self.shear_range[1]))

        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])

        transformation_matrix = self.__transform_matrix_offset_center(
            shear_matrix, image.shape[0], image.shape[1])
        image = self.__apply_transform(image, transformation_matrix)

        return image

    def __perform_random_shift(self, image):
        """ Applies a random shift in x and y

            Args:
                image: image to be augmented

            Returns:
                The transformed image
        """

        tx = np.random.uniform(-self.shift_range[0],
                               self.shift_range[0])
        ty = np.random.uniform(-self.shift_range[1],
                               self.shift_range[1])

        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])
        transformation_matrix = translation_matrix  # no need to do offset
        image = self.__apply_transform(image, transformation_matrix)

        return image

    def __perform_random_zoom(self, image):
        """ Applies a random zoom

            Args:
                image: image to be augmented

            Returns:
                The transformed image
        """
        zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)

        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])

        transformatiom_matrix = self.__transform_matrix_offset_center(
            zoom_matrix, image.shape[0], image.shape[1])
        image = self.__apply_transform(image, transformatiom_matrix)

        return image

    def __perform_random_contrast_stretching(self, image):
        p2, p98 = np.percentile(image, (2, 98)) #####
        image = exposure.rescale_intensity(image, in_range=(p2, p98)) #####
        return image

    def __perform_histogram_equalization(self, image):
        image = exposure.equalize_hist(image)
        return image

    def __augment_color(self, image):
        """ Takes an input image and returns a modified version of it """
        channel_ranges = (self.red_range, self.green_range, self.blue_range)
        for channel, channel_range in enumerate(channel_ranges):
            if not channel_range:
                continue  # no range set, so don't change that channel
            scale = random.uniform(channel_range[0], channel_range[1])
            image[:, :, channel] = image[:, :, channel * scale]

        image = np.clip(image, 0, 255)
        return image

    def __perform_random_brightness(self, image):
        scale = True
        brightness_range = (0.5, 1.5)
        u = np.random.uniform(brightness_range[0], brightness_range[1])
        image = tf.keras.preprocessing.image.apply_brightness_shift(image, u, scale)
        return image


    def __perform_flip(self, image):
        """Flip the input x horizontally with 100% probability."""
        image = np.fliplr(image)
        return image

    def get_random_transform(self, images):
        """ Applies a random augmentation to pairs of images

            Args:
                images: pairs of the batch to be augmented

            Returns:
                The transformed images
        """

        number_of_pairs_of_images = images[0].shape[0]
        random_numbers = np.random.random(
            size=(number_of_pairs_of_images * 2, 4))

        for pair_index in range(number_of_pairs_of_images):
            image_1 = images[0][pair_index, :, :, :]
            image_2 = images[1][pair_index, :, :, :]

            if random_numbers[pair_index * 2, 0] > 0.5:
                #image_1 = self.__perform_random_rotation(image_1)
                image_1 = self.__perform_random_contrast_stretching(image_1)
            if random_numbers[pair_index * 2, 1] > 0.5:
                image_1 = self.__perform_random_rotation(image_1)
                #image_1 = self.__perform_random_brightness(image_1)

            if random_numbers[pair_index * 2, 2] > 0.5:
                #image_1 = self.__perform_random_shift(image_1)
                image_1 = self.__perform_random_contrast_stretching(image_1)

            if random_numbers[pair_index * 2, 3] > 0.5:
                image_1 = self.__perform_random_shift(image_1)
               # image_1 = self.__perform_histogram_equalization(image_1)

            if random_numbers[pair_index * 2 + 1, 0] > 0.5:
                #image_2 = self.__perform_random_rotation(image_2)
                image_2 = self.__perform_random_contrast_stretching(image_2)

            if random_numbers[pair_index * 2 + 1, 1] > 0.5:
                image_2 = self.__perform_random_rotation(image_2)

                #image_2 = self.__perform_random_contrast_stretching(image_2)
                # image_2 = self.__perform_random_brightness(image_2)

            if random_numbers[pair_index * 2 + 1, 2] > 0.5:
                 #image_2 = self.__perform_random_shift(image_2)
                image_2 = self.__perform_random_contrast_stretching(image_2)

            if random_numbers[pair_index * 2 + 1, 3] > 0.5:
                image_2 = self.__perform_random_shift(image_2)
                #image_2 = self.__perform_histogram_equalization(image_2)

            images[0][pair_index, :, :, :] = image_1
            images[1][pair_index, :, :, :] = image_2

        return images
