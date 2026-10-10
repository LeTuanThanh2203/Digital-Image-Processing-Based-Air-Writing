# tracking/webcam.py
"""
Webcam capture module using OpenCV.

Provides a robust, object-oriented wrapper around OpenCV's VideoCapture
for reliable camera management, failure detection, and resource cleanup.
"""

import sys
from typing import Optional, Tuple
import cv2
import numpy as np


class Webcam:
    """Manages webcam capture, frame retrieval, and lifecycle."""

    def __init__(
        self,
        camera_index: int = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None,
    ) -> None:
        """Initialize webcam settings.

        Args:
            camera_index: Hardware index for the camera device (default: 0).
            width: Desired capture frame width.
            height: Desired capture frame height.
            fps: Desired capture frame rate.
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        """Open the camera capture device.

        Returns:
            bool: True if the camera opened successfully, False otherwise.
        """
        if self.is_opened():
            return True

        # Use DirectShow backend on Windows for faster initialization if available
        if sys.platform.startswith("win"):
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            # Fallback to default backend if DirectShow fails to open
            if not self.cap.isOpened():
                self.cap.release()
                self.cap = cv2.VideoCapture(self.camera_index)
        else:
            self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            self.cap = None
            return False

        # Apply camera properties if provided
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Zero buffer delay for real-time tracking
        if self.width is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        if self.height is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        if self.fps is not None:
            self.cap.set(cv2.CAP_PROP_FPS, float(self.fps))

        return True


    def is_opened(self) -> bool:
        """Check if the camera capture device is currently opened."""
        return self.cap is not None and self.cap.isOpened()

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a single frame from the camera.

        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success, frame).
                If capture fails or camera is disconnected, returns (False, None).
        """
        if not self.is_opened():
            return False, None

        ret, frame = self.cap.read()
        if not ret or frame is None or frame.size == 0:
            return False, None

        return True, frame

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Alias for read_frame() matching standard cv2.VideoCapture interface."""
        return self.read_frame()

    def get_resolution(self) -> Tuple[int, int]:
        """Get actual resolution (width, height) of the opened camera stream.

        Returns:
            Tuple[int, int]: (actual_width, actual_height)
        """
        if not self.is_opened():
            return (0, 0)
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)

    def release(self) -> None:
        """Release the camera hardware device safely."""
        if self.cap is not None:
            if self.cap.isOpened():
                self.cap.release()
            self.cap = None

    @staticmethod
    def close_windows() -> None:
        """Close all OpenCV GUI display windows."""
        cv2.destroyAllWindows()

    def __enter__(self) -> "Webcam":
        """Context manager entry point."""
        if not self.open():
            raise RuntimeError(
                f"Failed to open camera at index {self.camera_index}. "
                f"Please ensure a webcam is connected and not in use by another application."
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit point ensuring safe release and window cleanup."""
        self.release()
        self.close_windows()
