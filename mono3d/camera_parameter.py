"""Camera Parameter Module"""

from pathlib import Path
from typing import Union

import cv2
import cv2.typing as cvt
import numpy as np
import numpy.typing as npt
from tqdm.auto import trange, tqdm


class CameraParameter:
    """
    A class to store camera parameters

    Attributes:
        intrinsic_mat: npt.NDArray[np.float64]
            The intrinsic matrix of the camera
        distortion_coeffs: npt.NDArray[np.float64]
            The distortion coefficients of the camera

    Methods:
        save_to(file_path: Union[str, Path]) -> None
            Save camera parameters to a file
        load_from(file_path: Union[str, Path]) -> CameraParameter
            Load camera parameters from a file
    """

    def __init__(
        self,
        intrinsic_mat: npt.NDArray[np.float64],  # shape: (3, 3)
        distortion_coeffs: npt.NDArray[np.float64],  # shape: (5, 1)
    ):
        # camera parameters
        self.intrinsic_mat = intrinsic_mat
        self.distortion_coeffs = distortion_coeffs

    def save_to(self, file_path: Union[str, Path]):
        """Save camera parameters to a file"""
        np.savez(
            str(file_path),
            intrinsic_mat=self.intrinsic_mat,
            distortion_coeffs=self.distortion_coeffs,
        )

    @staticmethod
    def load_from(file_path: Union[str, Path]) -> "CameraParameter":
        """Load camera parameters from a file"""
        data = np.load(str(file_path))
        return CameraParameter(
            intrinsic_mat=data["intrinsic_mat"],
            distortion_coeffs=data["distortion_coeffs"],
        )

    @property
    def K(self) -> npt.NDArray[np.float64]:
        """alias of intrinsic_mat"""
        return self.intrinsic_mat

    def undistort_video(
        self, video_path: Union[str, Path], output_path: Union[str, Path]
    ):
        """Undistort a video using the camera parameters

        Args:
            video_path: Union[str, Path]
                The path to the input video
            output_path: Union[str, Path]
                The path to the output video

        Returns:
            None
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            tqdm.write(f"Error: Unable to open video file '{video_path}'")
            return

        # Get source video specs
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Create a VideoWriter object
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        out = cv2.VideoWriter(
            str(output_path), fourcc, fps, (frame_width, frame_height)
        )

        for i in trange(total_frames, desc="Undistorting frames:", leave=False):
            # Read a frame from the video
            ret, frame = cap.read()

            # Break the loop if unable to read the next frame
            if not ret:
                tqdm.write(
                    f"WARNING: Read frame failed at {i}, total frames: {total_frames}."
                    f"If video {video_path.as_posix()} is not corrupted, "
                    f"please try to convert the video to mp4 again with, e.g., ffmpeg."
                )
                break

            # Undistort the frame
            undistorted_frame = self.undistort_image(frame)

            # Write the undistorted frame to the output video
            out.write(undistorted_frame)

        # Release the video capture and video writer objects
        cap.release()
        out.release()

    def undistort_image(self, image: cvt.MatLike) -> cvt.MatLike:
        """Undistort an image using the camera parameters

        Args:
            image: np.ndarray
                The image to undistort

        Returns:
            np.ndarray
                The undistorted image
        """

        return cv2.undistort(image.copy(), self.intrinsic_mat, self.distortion_coeffs)
