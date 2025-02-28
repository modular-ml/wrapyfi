import logging
import importlib.util
import os
import io
import tempfile
import subprocess as sp
from queue import Queue
from typing import Optional, List

import numpy as np
import cv2


# importing pyav directly might cause issues with cv2.imshow() during initialization
def check_av_importable():
    # Check if the module exists
    if importlib.util.find_spec("av") is None:
        raise ImportError("PyAV (av) is not installed.")


class VideoEncoder:
    def __init__(
        self,
        buffer_length: int = 15,
        buffer_type: str = "frames",
        encoder: str = "opencv",
        codec: str = "h264",
        fps: int = 30,
        gop: int = 15,
        quality: int = 1,
        width: Optional[int] = None,
        height: Optional[int] = None,
        logging_level: int = logging.WARNING,
    ):
        """
        Initializes the video frame encoder which compresses image chunks using a specified encoder backend.

        The encoder supports multiple backends such as OpenCV, PyAV (libav), and FFmpeg CLI.
        Encoder-specific parameters (e.g., quality, GOP size) are applied based on the chosen backend.

        :param buffer_length: int: Number of frames, seconds, or bytes to buffer before encoding. Defaults to 15.
        :param buffer_type: str: Type of buffer ('frames', 'time', 'bytes'). Defaults to 'frames'.
        :param encoder: str: Encoder backend to use ('opencv', 'pyav', 'ffmpeg'). Defaults to 'opencv'.
        :param codec: str: Codec to use for encoding. Common values: 'h264', 'mp4v'. Defaults to 'h264'.
        :param fps: int: Frame rate of the video. Defaults to 30.
        :param gop: int: Group of Pictures size. Applicable to 'pyav' and 'ffmpeg' encoders. Defaults to 15.
        :param quality: int: Encoding quality (0-100). Higher values indicate better quality. Mapped to encoder-specific parameters. Defaults to 1.
        :param width: int: Width of the video frames. Must be specified.
        :param height: int: Height of the video frames. Must be specified.
        :param logging_level: int: Logging level for the encoder. Defaults to logging.WARNING.
        """
        self.buffer_length = buffer_length
        self.buffer_type = buffer_type
        self.encoder = encoder
        self.codec = codec.lower()
        self.fps = fps
        self.gop = gop
        self.quality = quality
        self.width = width
        self.height = height
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(logging_level)

        self._buffer: List[np.ndarray] = []
        self._current_buffer_size = 0

        if not self.width or not self.height:
            raise ValueError("Width and height must be specified for all encoders.")

        self._setup_encoder()

    def _setup_encoder(self):
        """Configures the encoder backend based on the specified encoder."""
        if self.encoder == "opencv":
            self._fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self._temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
            self._writer = None
            self._encode_video = self._encode_chunk_opencv
        elif self.encoder == "pyav":
            try:
                check_av_importable()
                self._encode_video = self._encode_chunk_pyav
            except ImportError:
                raise ImportError("PyAV is required for the 'pyav' encoder. Install it with 'pip install av'.")
        elif self.encoder == "ffmpeg":
            self._ffmpeg_process = None
            self._encode_video = self._encode_chunk_ffmpeg
        else:
            raise ValueError(f"Unsupported encoder: {self.encoder}")

    def add_frame(self, frame: np.ndarray) -> Optional[bytes]:
        """
        Adds a frame to the buffer and encodes a chunk if the buffer is full.

        :param frame: np.ndarray: BGR image frame with shape (height, width, 3).
        :return: bytes or None: Encoded video chunk if buffer is full, otherwise None.
        """
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Frame must be a BGR image with shape (height, width, 3).")

        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            raise ValueError(f"Frame dimensions {frame.shape[1]}x{frame.shape[0]} do not match encoder settings {self.width}x{self.height}.")

        self._buffer.append(frame)
        self._update_buffer_size(frame)

        if self._check_buffer_full():
            return self._encode_chunk()
        return None

    def _update_buffer_size(self, frame: np.ndarray):
        """Updates the current buffer size based on the buffer type."""
        if self.buffer_type == "frames":
            self._current_buffer_size += 1
        elif self.buffer_type == "time":
            self._current_buffer_size += 1 / self.fps
        elif self.buffer_type == "bytes":
            self._current_buffer_size += frame.nbytes

    def _check_buffer_full(self) -> bool:
        """Checks if the buffer has reached the specified length."""
        return self._current_buffer_size >= self.buffer_length

    def _encode_chunk(self) -> bytes:
        """Encodes the buffered frames using the configured encoder backend."""
        return self._encode_video()

    def _encode_chunk_opencv(self) -> bytes:
        """Encodes the buffer using OpenCV's VideoWriter."""
        self._writer = cv2.VideoWriter(
            self._temp_file,
            self._fourcc,
            self.fps,
            (self.width, self.height),
        )
        if not self._writer.isOpened():
            raise RuntimeError("Failed to initialize OpenCV VideoWriter.")

        for frame in self._buffer:
            self._writer.write(frame)
        self._writer.release()

        with open(self._temp_file, "rb") as f:
            chunk = f.read()

        os.unlink(self._temp_file)
        self._reset_buffer()
        return chunk

    def _encode_chunk_pyav(self) -> bytes:
        """Encodes the buffer using PyAV (libav)."""
        import av
        buffer = io.BytesIO()

        with av.open(buffer, mode="w", format="mp4") as container:
            stream = container.add_stream(self.codec, rate=self.fps)
            stream.width = self.width
            stream.height = self.height
            stream.pix_fmt = "yuv420p"

            crf = 28 - int(self.quality * 0.10)
            stream.options = {
                "crf": str(crf),
                "preset": "ultrafast",
                "tune": "zerolatency",
                "x264-params": f"keyint={self.gop}:min-keyint={self.gop}"
            }

            for frame in self._buffer:
                av_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
                for packet in stream.encode(av_frame):
                    container.mux(packet)

            for packet in stream.encode():
                container.mux(packet)

        chunk = buffer.getvalue()
        self._reset_buffer()
        return chunk

    def _encode_chunk_ffmpeg(self) -> bytes:
        """Encodes the buffer using FFmpeg CLI."""
        self._ffmpeg_process = sp.Popen(
            [
                "ffmpeg",
                "-y",
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{self.width}x{self.height}",
                "-r", str(self.fps),
                "-i", "-",
                "-c:v", self.codec,
                "-g", str(self.gop),
                "-crf", str(100 - self.quality),
                "-f", "mpegts",
                "pipe:1",
            ],
            stdin=sp.PIPE,
            stdout=sp.PIPE,
            stderr=sp.PIPE,
        )

        for frame in self._buffer:
            self._ffmpeg_process.stdin.write(frame.tobytes())
        self._ffmpeg_process.stdin.close()

        chunk, _ = self._ffmpeg_process.communicate()
        self._reset_buffer()
        return chunk

    def _reset_buffer(self):
        """Resets the buffer and current buffer size."""
        self._buffer = []
        self._current_buffer_size = 0

    def close(self):
        """Releases resources and cleans up temporary files."""
        if self.encoder == "opencv" and os.path.exists(self._temp_file):
            os.unlink(self._temp_file)
        elif self.encoder == "ffmpeg" and self._ffmpeg_process:
            self._ffmpeg_process.terminate()

    def __del__(self):
        """Ensures resources are cleaned up when the encoder is destroyed."""
        self.close()


class VideoDecoder:
    def __init__(
        self,
        codec: str,
        width: int,
        height: int,
        fps: int,
        decoder: str = "auto",
        logging_level: int = logging.WARNING,
    ):
        """
        Initializes the decoder which acquires video chunks and decodes into frames using a specified backend.

        Supports PyAV for efficient decoding of H.264 and other codecs in-memory, and OpenCV for other codecs
        using temporary files.

        :param codec: str: Video codec used in the encoded chunks (e.g., 'h264', 'mp4v').
        :param width: int: Width of the video frames.
        :param height: int: Height of the video frames.
        :param fps: int: Frame rate of the video.
        :param decoder: str: Backend decoder to use ('auto', 'pyav', 'opencv'). Defaults to 'auto'.
        :param logging_level: int: Logging level for the decoder. Defaults to logging.WARNING.
        """
        self.codec = codec.lower()
        self.width = width
        self.height = height
        self.fps = fps
        self.decoder = decoder.lower()
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(logging_level)

        self._cap = None
        self._setup_decoder()


    def _setup_decoder(self):
        """Configures the decoder backend based on the codec and available libraries."""
        if self.decoder == "auto":
            # Prefer PyAV for H.264 codecs if available
            if self.codec in ["h264", "libx264", "avc1"]:
                try:
                    check_av_importable()
                    self.decoder = "pyav"
                    self._decode_video = self._decode_chunk_pyav
                except ImportError:
                    self.log.warning("PyAV not installed, falling back to OpenCV.")
                    self.decoder = "opencv"
                    self._decode_video = self._decode_chunk_opencv
            else:
                self.decoder = "opencv"
                self._decode_video = self._decode_chunk_opencv

        if self.decoder == "pyav":
            try:
                check_av_importable()
                self._decode_video = self._decode_chunk_pyav
            except ImportError:
                raise ImportError("PyAV is required for the 'pyav' decoder. Install with 'pip install av'.")
        elif self.decoder == "opencv":
            self._decode_video = self._decode_chunk_opencv
            pass  # OpenCV is assumed to be available
        else:
            raise ValueError(f"Unsupported decoder: {self.decoder}")

    def decode(self, ret_queue: Queue, chunk: bytes):
        """
        Decodes a video chunk into a list of frames.

        :param chunk: bytes: The encoded video data.
        :return: List of frames in BGR format as numpy arrays.
        """
        self._decode_video(ret_queue, chunk)

    def _decode_chunk_pyav(self, ret_queue: Queue, chunk: bytes):
        """
        Decodes a chunk using PyAV.

        :param ret_queue: Queue: A queue to store the decoded frames and pass them to the main thread.
        :param chunk: bytes: The encoded video input data.
        """
        import av

        buffer = io.BytesIO(chunk)
        try:
            with av.open(buffer, format="mp4") as container:
                video_stream = next(s for s in container.streams if s.type == "video")
                for frame in container.decode(video_stream):
                    cv_frame = frame.to_ndarray(format="bgr24")
                    ret_queue.put(cv_frame)
        except Exception as e:
            self.log.error(f"PyAV decoding failed: {e}")
            return False
        return True

    def _decode_chunk_opencv(self, ret_queue: Queue, chunk: bytes):
        """
        Decodes a chunk using OpenCV by writing to a temporary file.

        :param ret_queue: Queue: A queue to store the decoded frames and pass them to the main thread.
        :param chunk: bytes: The encoded video input data.
        """
        # Write chunk to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file.write(chunk)
        temp_file.close()
        temp_path = temp_file.name

        self._cap = cv2.VideoCapture(temp_path)
        if not self._cap.isOpened():
            self.log.error("Failed to open temporary video file with OpenCV.")
            self._cap.release()
            os.unlink(temp_path)
            return False

        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            ret_queue.put(frame)

        self._cap.release()
        os.unlink(temp_path)
        return True

    def __del__(self):
        """Clean up any resources if necessary."""
        if self._cap:
            self._cap.release()
