""" class to build a 3D coordinate system from a cube with ArUco markers """

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import cv2
import cv2.typing as cvt
import numpy as np
from cv2 import aruco
from tqdm.auto import tqdm

from mono3d import CameraParameter
from mono3d.util import process_video


@dataclass
class ArucoDetection:
    """
    Represents an ArUco marker detection result.

    Attributes:
        corners: A np.ndarray containing the corner points of the detected markers.
        ids: A np.ndarray containing the IDs of the detected markers.
    """

    corners: Sequence[cvt.MatLike]
    ids: cvt.MatLike

    def __len__(self):
        assert len(self.corners) == len(
            self.ids
        ), "corners and ids should have same length, detection possibly corrupted."
        return len(self.ids)


class ArucoCube:
    """
    Represents a 3D coordinate system built from a cube with ArUco markers.

    Attributes:
        aruco_dict: aruco.Dictionary
            The dictionary of ArUco markers.
        aruco_params: aruco.DetectorParameters
            The parameters for the ArUco marker detector.
        marker_length: float
            The length of the ArUco markers in millimeters.
        detector: aruco.ArucoDetector
            The ArUco marker detector.
        cube_ids: np.ndarray
            The IDs of the ArUco markers on the cube.
        cube: aruco.Board
            The ArUco board representing the cube.

    Methods:
        detect(image: cvt.MatLike) -> ArucoDetection
            Detect ArUco markers in an image.
        draw(
            image: cvt.MatLike,
            cam_param: Optional[CameraParameter] = None,
            detection: Optional[ArucoDetection] = None,
            draw_cube_axis: bool = True,
            draw_markers: bool = False,
            draw_markers_axes: bool = False,
            min_markers: int = 1,
        ) -> cvt.MatLike
            Draw the axes and detectors on an image with a cube of markers.
    """

    def __init__(
        self,
        aruco_dict: Optional[aruco.Dictionary] = None,
        aruco_params: Optional[aruco.DetectorParameters] = None,
        marker_length: float = 43.0,
    ):
        self.aruco_dict = aruco_dict or aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.aruco_params = aruco_params or aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.marker_length = marker_length
        self.cube_ids = np.array([0, 1, 2, 3, 4, 5])
        c_pt = marker_length / 2
        marker_ids = np.array([[0], [1], [2], [3], [4], [5]], dtype=np.int32)
        # X axis in blue color, Y axis in green color and Z axis in red color.
        # order of corners is clockwise
        marker_corners = [
            np.array(
                [
                    [c_pt, -c_pt, c_pt],
                    [-c_pt, -c_pt, c_pt],
                    [-c_pt, c_pt, c_pt],
                    [c_pt, c_pt, c_pt],
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    [-c_pt, -c_pt, c_pt],
                    [c_pt, -c_pt, c_pt],
                    [c_pt, -c_pt, -c_pt],
                    [-c_pt, -c_pt, -c_pt],
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    [-c_pt, c_pt, c_pt],
                    [-c_pt, -c_pt, c_pt],
                    [-c_pt, -c_pt, -c_pt],
                    [-c_pt, c_pt, -c_pt],
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    [c_pt, c_pt, c_pt],
                    [-c_pt, c_pt, c_pt],
                    [-c_pt, c_pt, -c_pt],
                    [c_pt, c_pt, -c_pt],
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    [c_pt, -c_pt, c_pt],
                    [c_pt, c_pt, c_pt],
                    [c_pt, c_pt, -c_pt],
                    [c_pt, -c_pt, -c_pt],
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    [-c_pt, -c_pt, -c_pt],
                    [c_pt, -c_pt, -c_pt],
                    [c_pt, c_pt, -c_pt],
                    [-c_pt, c_pt, -c_pt],
                ],
                dtype=np.float32,
            ),
        ]
        self.cube = aruco.Board(
            objPoints=marker_corners,
            dictionary=self.aruco_dict,
            ids=marker_ids,
        )

    def detect(self, image: cvt.MatLike) -> ArucoDetection:
        """Detect ArUco markers in an image"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Detect markers
        corners, ids, _ = self.detector.detectMarkers(gray)
        return ArucoDetection(corners=corners, ids=ids)

    @staticmethod
    def _draw_markers(
        image: cvt.MatLike,
        detection: ArucoDetection,
    ) -> cvt.MatLike:
        """Draw the detected ArUco markers on an image"""
        return aruco.drawDetectedMarkers(image, detection.corners, detection.ids)

    def _draw_markers_axes(
        self,
        image: cvt.MatLike,
        cam_param: CameraParameter,
        detection: ArucoDetection,
    ) -> cvt.MatLike:
        """Draw the 3D axis on an image"""
        image = image.copy()
        for corners in detection.corners:
            # Estimate the pose of the marker
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners=corners,
                markerLength=self.marker_length,
                cameraMatrix=cam_param.K,
                distCoeffs=cam_param.distortion_coeffs,
            )
            # Draw the axis
            image = cv2.drawFrameAxes(
                image=image,
                cameraMatrix=cam_param.K,
                distCoeffs=cam_param.distortion_coeffs,
                rvec=rvecs[0],
                tvec=tvecs[0],
                length=self.marker_length,
            )

        return image

    def _draw_cube_axis(
        self,
        image: cvt.MatLike,
        cam_param: CameraParameter,
        detection: ArucoDetection,
    ) -> cvt.MatLike:
        """Draw the 3D axis of the cube on an image"""
        _, rvecs, tvecs = aruco.estimatePoseBoard(
            corners=detection.corners,
            ids=detection.ids,
            board=self.cube,
            cameraMatrix=cam_param.K,
            distCoeffs=cam_param.distortion_coeffs,
            rvec=np.empty(1),
            tvec=np.empty(1),
        )
        return cv2.drawFrameAxes(
            image=image,
            cameraMatrix=cam_param.K,
            distCoeffs=cam_param.distortion_coeffs,
            rvec=rvecs,
            tvec=tvecs,
            length=self.marker_length,
        )

    def plot_on_image(
        self,
        image: cvt.MatLike,
        cam_param: Optional[CameraParameter] = None,
        detection: Optional[ArucoDetection] = None,
        draw_cube_axis: bool = True,
        draw_markers: bool = False,
        draw_markers_axes: bool = False,
        min_markers: int = 1,
    ) -> cvt.MatLike:
        """
        Draw the axes and detectors on an image with a cube of markers.
        By default, it only draws the cube axis.

        Args:
            image (cvt.MatLike):
                The input image
            cam_param (CameraParameter, optional):
                The camera parameters, default to None.
            detection (ArucoDetection, optional):
                The detected markers, default to None.
            draw_cube_axis (bool, optional):
                Whether to draw the cube axis, default to True.
            draw_markers (bool, optional):
                Whether to draw the markers, default to False.
            draw_markers_axes (bool, optional):
                Whether to draw the markers axes, default to False.
            min_markers (int, optional):
                The minimum number of markers to draw, default to 1.

        Returns:
            cvt.MatLike: The output image
        """
        image = image.copy()
        detection = detection or self.detect(image)

        if detection.ids is None or len(detection.ids) < min_markers:
            return image

        if cam_param is None and (draw_markers_axes or draw_cube_axis):
            raise ValueError("cam_param is required to draw the axis.")

        if draw_markers:
            image = self._draw_markers(image, detection)
        if draw_markers_axes:
            image = self._draw_markers_axes(image, cam_param, detection)
        if draw_cube_axis:
            image = self._draw_cube_axis(image, cam_param, detection)

        return image

    def plot_on_video(
        self,
        src_video_path: Path,
        cam_param: Optional[CameraParameter] = None,
        draw_cube_axis: bool = True,
        draw_markers: bool = False,
        draw_markers_axes: bool = False,
        min_markers: int = 1,
        scale: float = 1.0,
    ):
        """Draw the cube axis on a video"""
        process_video(
            src_video_path=src_video_path,
            frame_process_func=functools.partial(
                self.plot_on_image,
                cam_param=cam_param,
                draw_cube_axis=draw_cube_axis,
                draw_markers=draw_markers,
                draw_markers_axes=draw_markers_axes,
                min_markers=min_markers,
            ),
            scale=scale,
        )


if __name__ == "__main__":

    test_video_dir = Path("../../siberian_jay/mapping_videos")
    test_image = cv2.imread("../tests/images/aruco_cube.jpg")
    cam_param = CameraParameter.load_from("../tests/images/cam_param.npz")
    aruco_cube = ArucoCube()
    scale = 0.5
    stream = False
    # visualize the cube axis from a video
    video_paths = list(test_video_dir.glob("*.MP4"))[:1]
    for vp in tqdm(
        video_paths, desc="Processing videos", bar_format="{l_bar}{bar:10}{r_bar}"
    ):
        aruco_cube.plot_on_video(
            src_video_path=vp,
            cam_param=cam_param,
            draw_cube_axis=True,
            draw_markers=False,
            draw_markers_axes=False,
            min_markers=1,
            scale=scale,
        )
