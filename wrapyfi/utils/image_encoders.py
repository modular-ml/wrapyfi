import logging
from io import BytesIO

import cv2
import numpy as np


class JpegEncoder(object):
    def __init__(
        self,
        quality: int = 95,
        encoder: str = "opencv",
        logging_level: int = logging.WARNING,
    ):
        """
        Initialize the JPEG encoder with the specified quality and encoder.

        :param quality: int: JPEG quality (0-100)
        :param encoder: str: Encoder to use (opencv, pil, or vips)
        :param logging_level: int: Logging level for the encoder. Defaults to logging.WARNING
        """
        self.encoder = encoder
        self.quality = quality

        if encoder == "vips":
            try:
                import pyvips

                logging.getLogger("pyvips").setLevel(logging_level)
                self._encode_jpg_image = self._encode_jpg_image_vips
            except ImportError:
                raise ImportError(
                    "The 'pyvips' package is required for the 'vips' encoder. "
                    "Install it with 'pip install pyvips'. This also requires installing 'libvips' e.g., "
                    "on Ubuntu 'sudo apt install libvips-dev'."
                )
        elif encoder == "pil":
            try:
                from PIL import Image

                logging.getLogger("PIL").setLevel(logging_level)
                self._encode_jpg_image = self._encode_jpg_image_pil
            except ImportError:
                raise ImportError(
                    "The 'Pillow' package is required for the 'pil' encoder. Install it with 'pip install Pillow'."
                )
        elif encoder == "opencv":
            logging.getLogger("cv2").setLevel(logging_level)
            self._encode_jpg_image = self._encode_jpg_image_opencv
        else:
            raise ValueError("The encoder must be either 'pil', 'opencv', or 'vips'.")

    def encode_jpg_image(self, img: np.ndarray, return_numpy: bool = False):
        """
        Encode an image to JPEG using the specified encoder.

        :param img: np.ndarray: Input image in BGR format (OpenCV default)
        :param return_numpy: bool: Whether to return the encoded image as a numpy array. Default is False.

        Returns:
            img_bytes (bytes or np.ndarray): Encoded JPEG image as bytes or numpy array.
        """
        return self._encode_jpg_image(img, return_numpy=return_numpy)

    def _encode_jpg_image_opencv(self, img: np.ndarray, return_numpy: bool = False):
        """
        Encode an image to JPEG using OpenCV.

        :param img: np.ndarray: Input image in BGR format (OpenCV default)
        :param return_numpy: bool: Whether to return the encoded image as a numpy array. Default is False.

        Returns:
            img_bytes (bytes or np.ndarray): Encoded JPEG image as bytes or numpy array.
        """
        _, img_bytes = cv2.imencode(
            ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        )
        if return_numpy:
            return img_bytes
        else:
            return img_bytes.tobytes()

    def _encode_jpg_image_pil(self, img: np.ndarray, return_numpy: bool = False):
        """
        Encode an image to JPEG using Pillow (PIL).

        :param img: np.ndarray: Input image in BGR format (OpenCV default)
        :param return_numpy: bool: Whether to return the encoded image as a numpy array. Default is False.

        Returns:
            img_bytes (bytes or np.ndarray): Encoded JPEG image as bytes or numpy array.
        """
        from PIL import Image

        # Convert BGR (OpenCV) to RGB (Pillow)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert numpy array to PIL Image
        pil_img = Image.fromarray(img_rgb)

        # Encode to JPEG with the specified quality
        with BytesIO() as buffer:
            pil_img.save(buffer, format="JPEG", quality=self.quality, optimize=True)
            img_bytes = buffer.getvalue()

        if return_numpy:
            return np.frombuffer(img_bytes, dtype=np.uint8)
        else:
            return img_bytes

    def _encode_jpg_image_vips(self, img: np.ndarray, return_numpy: bool = False):
        """
        Encode an image to JPEG using libvips (pyvips).

        :param img: np.ndarray: Input image in BGR format (OpenCV default)
        :param return_numpy: bool: Whether to return the encoded image as a numpy array. Default is False.

        Returns:
            img_bytes (bytes or np.ndarray): Encoded JPEG image as bytes or numpy array.
        """
        import pyvips

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Create a pyvips image from the numpy array
        vips_img = pyvips.Image.new_from_memory(
            img_rgb.data,  # Raw image data
            img_rgb.shape[1],  # Width
            img_rgb.shape[0],  # Height
            img_rgb.shape[2],  # Number of bands (channels)
            "uchar",  # Data type
        )

        # Encode to JPEG with the specified quality
        img_bytes = vips_img.jpegsave_buffer(Q=self.quality)

        if return_numpy:
            return np.frombuffer(img_bytes, dtype=np.uint8)
        else:
            return img_bytes
