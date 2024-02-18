""" CharucoBoard class compatible with both OpenCV above and below 4.6.0 """

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Any

import cv2
import numpy as np
import numpy.typing as npt
from tqdm.auto import tqdm

from mono3d import CameraParameter
from mono3d.globals import LEGACY


@dataclass
class CharucoBoardDetection:
    """ Structured storage of detections on a Charuco board """
    aruco_marker_ids: npt.NDArray[np.int32]  # shape: (n, 1)
    aruco_marker_corners: npt.NDArray[np.float32]  # shape: (n, 4, 2)
    charuco_ids: Optional[npt.NDArray[np.int32]] = None  # shape: (n, 1)
    charuco_corners: Optional[npt.NDArray[np.float32]] = None  # shape: (n, 4, 2)


class CharucoBoard:
    """
    A class represents a Charuco board.
    # TODO: not tested with OpenCV > 4.6.0

    Attributes:
        aruco_dict (cv2.aruco.Dictionary): The Aruco dictionary.
        n_horizontal_squares (int): The number of horizontal squares.
        n_vertical_squares (int): The number of vertical squares.
        board_square_length (float): The length of a square on the board.
        aruco_marker_length (float): The length of an aruco marker.
        board (cv2.aruco.CharucoBoard): The Charuco board object.
        aruco_detector (Optional[cv2.aruco.ArucoDetector]):
            Aruco marker detector, only available for OpenCV > 4.6.0
        charuco_detector (Optional[cv2.aruco.CharucoDetector]):
            Charuco board detector, only available for OpenCV > 4.6.0

    Methods:
        detect(image: npt.NDArray[np.uint8]) -> Tuple[int, CharucoBoardDetection]:
            Detects Aruco and Charuco markers in the given image.
        _detect_legacy(image: npt.NDArray[np.uint8]) -> Tuple[int, CharucoBoardDetection]:
            detect() function for OpenCV < 4.6.0
        calibrate_camera(detections: List[Tuple[npt.NDArray[np.uint8],
        CharucoBoardDetection]]) -> CameraParameter:
            Calibrates the camera using Charuco detections.
        draw_detection(image: npt.NDArray[np.uint8], detection: CharucoBoardDetection) -> npt.NDArray[np.uint8]:
            Draw charuco detection on a image
    """

    def __init__(
            self,
            # TODO: unable to use [cv2.aruco.Dictionary] since not exist in OpenCV < 4.6.0, unsure about newer version
            aruco_dict: Optional[Any] = None,
            n_horizontal_squares: int = 5,
            n_vertical_squares: int = 10,
            board_square_length: float = 55.0,
            aruco_marker_length: float = 43.0,
    ):
        if aruco_dict is None:  # TODO: Necessary? Is cv2.aruco.Dictionary mutable?
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        else:
            self.aruco_dict = aruco_dict
        self.n_horizontal_squares = n_horizontal_squares
        self.n_vertical_squares = n_vertical_squares
        self.board_square_length = board_square_length
        self.aruco_marker_length = aruco_marker_length

        if LEGACY:
            self.board = cv2.aruco.CharucoBoard_create(
                squaresX=self.n_horizontal_squares,
                squaresY=self.n_vertical_squares,
                squareLength=self.board_square_length,
                markerLength=self.aruco_marker_length,
                dictionary=self.aruco_dict
            )
            self.aruco_detector = None
            self.charuco_detector = None
        else:
            self.board = cv2.aruco.CharucoBoard(
                size=(self.n_horizontal_squares, self.n_vertical_squares),
                squareLength=self.board_square_length,
                markerLength=self.aruco_marker_length,
                dictionary=self.aruco_dict
            )
            self.aruco_detector = cv2.aruco.ArucoDetector(
                dictionary=self.aruco_dict,
                detectorParams=cv2.aruco.DetectorParameters()
            )
            self.charuco_detector = cv2.aruco.CharucoDetector(board=self.board)

    def detect(self, image: npt.NDArray[np.uint8]) -> Tuple[int, Optional[CharucoBoardDetection]]:
        """
        Detects Aruco and Charuco markers in the given image.
        Detected Aruco markers are refined for better accuracy using cornerSubPix.

        Arg:
            image (npt.NDArray[np.uint8]): The input image for marker detection.

        Returns:
            Tuple[int, CharucoBoardDetection]: A tuple containing the number of detected charuco corners and
                detection details
        """
        # check if image is not empty
        if image.size == 0:
            return 0, None

        # image = self.sharp(image=image.copy())  # TODO: sharpen image?

        if LEGACY:
            return self._detect_legacy(image=image)

        # While OpenCV >= 4.6.0, and LEGACY is False, aruco_detector and charuco_detector are available
        aruco_corners, aruco_ids, _ = self.aruco_detector.detectMarkers(image=image)  # type: ignore

        if aruco_ids is None:
            return 0, None

        # iterates over each detected Aruco corner to find the sub-pixel accurate location
        criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 100, 0.00001)  # 0.00001 come from Alex
        for aruco_corner in aruco_corners:
            aruco_corner = cv2.cornerSubPix(
                image=image,
                corners=aruco_corner,
                winSize=(3, 3),
                zeroZone=(-1, -1),
                criteria=criteria
            )
        # While OpenCV >= 4.6.0, and LEGACY is False, aruco_detector and charuco_detector are available
        charuco_corners, charuco_ids, _, _ = self.charuco_detector.detectBoard(  # type: ignore
            image=image,
            markerCorners=aruco_corners,
            markerIds=aruco_ids,
        )

        # TODO: Do subpix on charuco corners
        n_charuco_corners = len(charuco_corners)
        return n_charuco_corners, CharucoBoardDetection(
            aruco_marker_corners=aruco_corners,
            aruco_marker_ids=aruco_ids,
            charuco_corners=charuco_corners,
            charuco_ids=charuco_ids
        )

    def _detect_legacy(self, image: npt.NDArray[np.uint8]) -> Tuple[int, Optional[CharucoBoardDetection]]:
        """
        detect() function for OpenCV < 4.6.0

        Args:
            image (npt.NDArray[np.uint8])

        Returns:
            Tuple[int, CharucoBoardDetection]
        """
        aruco_corners, aruco_ids, _ = cv2.aruco.detectMarkers(image=image, dictionary=self.aruco_dict)

        if aruco_ids is None:
            return 0, None

        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=aruco_corners,
            markerIds=aruco_ids,
            image=image,
            board=self.board,
            # number of adjacent markers that must be detected to return a charuco corner
            minMarkers=2
        )
        n_charuco_corners = len(charuco_corners)
        # TODO: Do subpix on charuco corners
        return n_charuco_corners, CharucoBoardDetection(
            aruco_marker_corners=aruco_corners,
            aruco_marker_ids=aruco_ids,
            charuco_corners=charuco_corners,
            charuco_ids=charuco_ids
        )

    def calibrate_camera(
            self,
            detections: List[Tuple[npt.NDArray[np.uint8], CharucoBoardDetection]]
    ) -> CameraParameter:
        """
        Calibrates the camera using Charuco detections.

        It collects data from multiple image and detection pairs and computes the camera
        intrinsic parameters and distortion coefficients. The calibration results are returned
        as a CameraParameter object.

        Args:
            detections (List[Tuple[npt.NDArray[np.uint8], CharucoBoardDetection]]): A list of image and
                marker detection tuples.

        Returns:
            CameraParameter: The camera calibration results.
        """

        if not detections:
            print("WARNING: camera calibration receives no detections, returning zero matrix.")
            return CameraParameter(intrinsic_mat=np.zeros((3, 3)), distortion_coeffs=np.zeros((5, 1)))

        all_corners_flatten: List[npt.NDArray[Any]] = []
        all_ids_flatten: List[npt.NDArray[Any]] = []

        # take shape from the first image of the first detection pair
        image_size = detections[0][0].shape[:2]

        for _, detection in detections:
            if detection.charuco_corners is None or detection.charuco_ids is None:
                continue
            all_corners_flatten.append(detection.charuco_corners)
            all_ids_flatten.append(detection.charuco_ids)
        (
            error,
            camera_matrix,
            distortion_coeffs,
            _,
            _,
            intrinsics_std,
            extrinsics_std,
            per_view_errors
        ) = cv2.aruco.calibrateCameraCharucoExtended(
            charucoCorners=all_corners_flatten,
            charucoIds=all_ids_flatten,
            board=self.board,
            imageSize=image_size,
            cameraMatrix=None,
            distCoeffs=None,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001),
        )

        # TODO: iterate to remove high error views (images)?

        # refine the camera matrix based on a free scaling parameter for undistortion
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix,
            distortion_coeffs,
            imageSize=image_size,
            alpha=1,  # all pixels are retained with some extra black images
            newImgSize=image_size
        )

        return CameraParameter(
            intrinsic_mat=new_camera_matrix,
            distortion_coeffs=distortion_coeffs
        )

    @staticmethod
    def draw_detection(
            image: npt.NDArray[np.uint8],
            detection: CharucoBoardDetection
    ) -> npt.NDArray[np.uint8]:
        """
        Draw charuco detection on an image

        Args:
            image (npt.NDArray[np.uint8]): image to be plotted
            detection (CharucoBoardDetection): charuco detection

        Returns:
            npt.NDArray[np.uint8]: image plotted with charuco detection
        """
        plotted = image.copy()
        cv2.aruco.drawDetectedMarkers(
            image=plotted,
            corners=detection.aruco_marker_corners,
            ids=detection.aruco_marker_ids
        )
        cv2.aruco.drawDetectedCornersCharuco(
            image=plotted,
            charucoCorners=detection.charuco_corners,
            charucoIds=detection.charuco_ids,
        )
        return plotted


if __name__ == "__main__":
    # short test for CharucoBoard class in this file
    print(f"Using OpenCV={cv2.__version__}")

    test_image_path = "../tests/images/charuco_board.jpg"
    test_image = cv2.imread(test_image_path)

    board = CharucoBoard()
    n_corners, detection = board.detect(test_image)
    if detection is None:
        print("No detections found")
        exit(1)
    n_markers = len(detection.aruco_marker_ids)

    print(f"Found {n_markers} Aruco markers and {n_corners} Charuco corners")
    # print ids
    print(f"marker ids: {detection.aruco_marker_ids}")
    print(f"charuco ids: {detection.charuco_ids}")
    # print dtype of corners and ids
    print(f"marker corner dtype: {detection.aruco_marker_corners[0].dtype}")
    print(f"marker id dtype: {detection.aruco_marker_ids[0].dtype}")
    if detection.charuco_corners is not None and detection.charuco_ids is not None:
        print(f"charuco corner dtype: {detection.charuco_corners[0].dtype}")
        print(f"charuco id dtype: {detection.charuco_ids[0].dtype}")

    test_calib_image_dir = Path("../tests/images/cam_calib_charuco_images")
    test_calib_image_paths = list(test_calib_image_dir.glob("*.jpg"))

    detections: List[Tuple[npt.NDArray[np.uint8], CharucoBoardDetection]] = []
    write_dir = test_calib_image_dir / "plotted"
    write_dir.mkdir(exist_ok=True)
    for image_path in tqdm(test_calib_image_paths, desc="Calibrating"):
        image = cv2.imread(str(image_path))
        n_corners, detection = board.detect(image)
        if detection is None:
            print(f"No detections found in {image_path}")
            continue
        plotted = CharucoBoard.draw_detection(image=image, detection=detection)
        # write to file
        cv2.imwrite(str(write_dir / f"{image_path.stem}.plotted.jpg"), plotted)

        detections.append((image, detection))

    calibration = board.calibrate_camera(detections)

    np.set_printoptions(precision=8, suppress=True, floatmode='fixed')
    print("Camera matrix:")
    print(calibration.intrinsic_mat.dtype)
    print(calibration.intrinsic_mat)
    print("Distortion coefficients:")
    print(calibration.distortion_coeffs.dtype)
    print(calibration.distortion_coeffs)
