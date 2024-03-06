""" class to build a 3D coordinate system from a cube with ArUco markers """

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import cv2.typing as cvt
import numpy as np
from cv2 import aruco
from tqdm.auto import tqdm

from mono3d import CameraParameter


@dataclass
class ArucoDetection:
    corners: cvt.MatLike
    ids: cvt.MatLike


class ArucoCube:

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
        board_ids = np.array([[0], [1], [2], [3], [4], [5]], dtype=np.int32)
        board_corners = [
            np.array(
                [
                    [-c_pt, c_pt, c_pt],
                    [c_pt, c_pt, c_pt],
                    [c_pt, -c_pt, c_pt],
                    [-c_pt, -c_pt, c_pt],
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
        self.board = aruco.Board(
            objPoints=board_corners,
            dictionary=self.aruco_dict,
            ids=board_ids,
        )

    def detect(self, image: cvt.MatLike) -> ArucoDetection:
        """Detect ArUco markers in an image"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Detect markers
        corners, ids, _ = self.detector.detectMarkers(gray)
        return ArucoDetection(corners=corners, ids=ids)

    def draw_markers(
        self,
        image: cvt.MatLike,
        detection: Optional[ArucoDetection] = None,
    ) -> cvt.MatLike:
        """Draw the detected ArUco markers on an image"""
        detection = detection or self.detect(image)
        return aruco.drawDetectedMarkers(image, detection.corners, detection.ids)

    def draw_axis(
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

    def draw_cube_axis(
        self,
        image: cvt.MatLike,
        cam_param: CameraParameter,
        detection: Optional[ArucoDetection] = None,
        draw_markers: bool = False,
        min_markers: int = 2,
    ) -> cvt.MatLike:
        """Draw the 3D axis on an image with a cube of markers"""
        image = image.copy()
        detection = detection or self.detect(image)
        if detection.ids is None or len(detection.ids) < min_markers:
            return image
        if draw_markers:
            image = self.draw_markers(image, detection)
        _, rvecs, tvecs = aruco.estimatePoseBoard(
            corners=detection.corners,
            ids=detection.ids,
            board=self.board,
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


if __name__ == "__main__":

    test_image = cv2.imread("../tests/images/aruco_cube.jpg")
    aruco_cube = ArucoCube()
    detection = aruco_cube.detect(test_image)

    # visualize the detected markers
    # image_with_markers = aruco_cube.draw_markers(test_image, detection)
    # cv2.imshow("Aruco markers", image_with_markers)
    # cv2.waitKey(0)

    # visualize the detected markers with axis
    image_with_axis = aruco_cube.draw_axis(
        test_image,
        CameraParameter.load_from("../tests/images/cam_param.npz"),
        detection,
    )
    cv2.imshow("Aruco markers with axis", image_with_axis)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
