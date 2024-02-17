from pathlib import Path
from typing import Union, Optional

import cv2
import numpy as np
import numpy.typing as npt
from tqdm.auto import trange


class CameraParameter:
    """
    A class to store camera parameters

    Attributes:
        intrinsic_mat: np.ndarray
            The intrinsic matrix of the camera
        distortion_coeffs: np.ndarray
            The distortion coefficients of the camera
        rvec: np.ndarray
            The rotation vector of the camera
        tvec: np.ndarray
            The translation vector of the camera

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
            rvec: Optional[npt.NDArray[np.float64]] = None,
            tvec: Optional[npt.NDArray[np.float64]] = None,
    ):
        # camera parameters
        self.intrinsic_mat = intrinsic_mat
        self.distortion_coeffs = distortion_coeffs
        self.rvec = rvec
        self.tvec = tvec

        # self.rotation_mat = rotation_mat
        # self.projection_mat = np.dot(self.intrinsic_mat, np.hstack((self.rotation_mat, self.tvec)))
        # self.fundamental_mat = None  # TODO: implement this later

        # self.intrinsic_mat: np.ndarray = np.zeros((1, 1))
        # self.rvec: np.ndarray = np.zeros((1, 1))
        # self.tvec: np.ndarray = np.zeros((1, 1))
        # self.rotation_mat: np.ndarray = np.zeros((1, 1))
        # self.projection_mat: np.ndarray = np.zeros((1, 1))
        # self.fundamental_mat: dict[str, np.ndarray] = {}

    def save_to(self, file_path: Union[str, Path]):
        """ Save camera parameters to a file """
        np.savez(
            str(file_path),
            intrinsic_mat=self.intrinsic_mat,
            distortion_coeffs=self.distortion_coeffs,
            rvec=self.rvec,
            tvec=self.tvec
            # rotation_mat=self.rotation_mat,
            # projection_mat=self.projection_mat,
        )

    def load_from(self, file_path: Union[str, Path]):
        """ Load camera parameters from a file """
        data = np.load(str(file_path))
        self.intrinsic_mat = data['intrinsic_mat']
        self.distortion_coeffs = data['distortion_coeffs']
        self.rvec = data['rvec']
        self.tvec = data['tvec']
        # self.rotation_mat = data['rotation_mat']
        # self.projection_mat = data['projection_mat']
        return self

    @property
    def K(self) -> np.ndarray:
        """ alias of intrinsic_mat """
        return self.intrinsic_mat

    @property
    def r(self) -> np.ndarray:
        """ alias of rvec """
        return self.rvec

    @property
    def t(self) -> np.ndarray:
        """ alias of tvec """
        return self.tvec

    # @property
    # def R(self) -> np.ndarray:
    #     """ alias of rotation_mat """
    #     return self.rotation_mat
    #
    # @property
    # def F(self) -> dict[str, np.ndarray]:
    #     """ alias of fundamental_mat """
    #     return self.fundamental_mat
    #
    # @property
    # def P(self) -> np.ndarray:
    #     """ alias of projection_mat """
    #     return self.projection_mat

    def undistort_video(self, video_path: Union[str, Path], output_path: Union[str, Path]):
        """ Undistort a video using the camera parameters

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
            print(f"Error: Unable to open video file '{video_path}'")
            return

        # Get source video specs
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

        for i in trange(total_frames, desc="Undistorting frames:", leave=False):
            # Read a frame from the video
            ret, frame = cap.read()

            # Break the loop if unable to read the next frame
            if not ret:
                print(f'WARNING: Read frame failed at {i}, total frames: {total_frames}.'
                      f'If video is not corrupted, please try to convert the video to mp4 again with, e.g., ffmpeg.')
                break

            # Undistort the frame
            undistorted_frame = self.undistort_image(frame)

            # Write the undistorted frame to the output video
            out.write(undistorted_frame)

        # Release the video capture and video writer objects
        cap.release()
        out.release()

    def undistort_image(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """ Undistort an image using the camera parameters

        Args:
            image: np.ndarray
                The image to undistort

        Returns:
            np.ndarray
                The undistorted image
        """
        return cv2.undistort(image.copy(), self.intrinsic_mat, self.distortion_coeffs)
