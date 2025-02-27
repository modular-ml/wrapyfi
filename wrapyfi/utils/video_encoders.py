import logging
import os
import io
import tempfile
import subprocess as sp
from typing import Optional, Union, List, Tuple

import numpy as np
import cv2


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
        if self.encoder == "opencv":
            self._fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self._temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
            self._writer = None
        elif self.encoder == "pyav":
            try:
                import av
            except ImportError:
                raise ImportError("PyAV is required for the 'pyav' encoder. Install it with 'pip install av'.")
        elif self.encoder == "ffmpeg":
            self._ffmpeg_process = None
        else:
            raise ValueError(f"Unsupported encoder: {self.encoder}")

    def add_frame(self, frame: np.ndarray) -> Optional[bytes]:
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
        if self.buffer_type == "frames":
            self._current_buffer_size += 1
        elif self.buffer_type == "time":
            self._current_buffer_size += 1 / self.fps
        elif self.buffer_type == "bytes":
            self._current_buffer_size += frame.nbytes

    def _check_buffer_full(self) -> bool:
        return self._current_buffer_size >= self.buffer_length

    def _encode_chunk(self) -> bytes:
        if self.encoder == "opencv":
            return self._encode_chunk_opencv()
        elif self.encoder == "pyav":
            return self._encode_chunk_pyav()
        elif self.encoder == "ffmpeg":
            return self._encode_chunk_ffmpeg()
        return b""

    def _encode_chunk_opencv(self) -> bytes:
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
        import av
        buffer = io.BytesIO()

        with av.open(buffer, mode="w", format="mp4") as container:
            stream = container.add_stream("libx264", rate=self.fps)
            stream.width = self.width
            stream.height = self.height
            stream.pix_fmt = "yuv420p"

            # Sensible CRF mapping (0–100 → 28–18)
            crf = 28 - int(self.quality * 0.10)
            stream.options = {
                "crf": str(crf),
                "preset": "ultrafast",  # Faster encoding
                "tune": "zerolatency",  # Reduce buffering
                "x264-params": "keyint={}:min-keyint={}".format(self.gop, self.gop)  # Force keyframes
            }

            for frame in self._buffer:
                av_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
                for packet in stream.encode(av_frame):
                    container.mux(packet)

            # Flush remaining packets
            for packet in stream.encode():
                container.mux(packet)

        chunk = buffer.getvalue()
        with open("debug_output.mp4", "wb") as f:
            f.write(chunk)

        self._reset_buffer()
        return chunk

    def _encode_chunk_ffmpeg(self) -> bytes:
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
        self._buffer = []
        self._current_buffer_size = 0

    def close(self):
        if self.encoder == "opencv" and os.path.exists(self._temp_file):
            os.unlink(self._temp_file)
        elif self.encoder == "ffmpeg" and self._ffmpeg_process:
            self._ffmpeg_process.terminate()

    def __del__(self):
        self.close()