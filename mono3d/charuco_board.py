"""CharucoBoard class"""

from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

import cv2
import cv2.typing as cvt
import numpy as np
import numpy.typing as npt
from tqdm.auto import tqdm

from mono3d import CameraParameter
from mono3d.charuco_board_detection import CharucoBoardDetection


class CharucoBoard:
    """
    A class represents a Charuco board.

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
        detect(image: cvt.MatLike) -> Tuple[int, CharucoBoardDetection]:
            Detects Aruco and Charuco markers in the given image.
        calibrate_camera(detections: Iterable[[CharucoBoardDetection]]) -> CameraParameter:
            Calibrates the camera using Charuco detections.
        draw_detection(image: cvt.MatLike, detection: CharucoBoardDetection) -> cvt.MatLike:
            Draw charuco detection on an image
    """

    def __init__(
        self,
        aruco_dict: Optional[cv2.aruco.Dictionary] = None,
        n_horizontal_squares: int = 5,
        n_vertical_squares: int = 10,
        board_square_length: float = 55.0,
        aruco_marker_length: float = 43.0,
        use_board_legacy_pattern: bool = True,
    ):
        if aruco_dict is None:  # TODO: Necessary? Is cv2.aruco.Dictionary mutable?
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        else:
            self.aruco_dict = aruco_dict
        self.n_horizontal_squares = n_horizontal_squares
        self.n_vertical_squares = n_vertical_squares
        self.board_square_length = board_square_length
        self.aruco_marker_length = aruco_marker_length

        self.board = cv2.aruco.CharucoBoard(
            size=(self.n_horizontal_squares, self.n_vertical_squares),
            squareLength=self.board_square_length,
            markerLength=self.aruco_marker_length,
            dictionary=self.aruco_dict,
        )
        if use_board_legacy_pattern:
            tqdm.write("WARNING: Using the legacy CharucoBoard pattern.")
            self.board.setLegacyPattern(use_board_legacy_pattern)
        self.aruco_detector = cv2.aruco.ArucoDetector(
            dictionary=self.aruco_dict, detectorParams=cv2.aruco.DetectorParameters()
        )
        self.charuco_detector = cv2.aruco.CharucoDetector(board=self.board)

    def detect(self, image: cvt.MatLike) -> Optional[CharucoBoardDetection]:
        """
        Detects Aruco and Charuco markers in the given image.
        Detected Aruco markers are refined for better accuracy using cornerSubPix.

        Arg:
            image (cvt.MatLike): The input image for marker detection.

        Returns:
            Tuple[int, CharucoBoardDetection]:
                A tuple containing the number of detected charuco corners and
                detection details
        """
        # check if the image is empty
        if image.size == 0:
            return None

        # if not gray scale, convert to gray scale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # image = self.sharp(image=image.copy())  # TODO: sharpen image?

        aruco_corners, aruco_ids, _ = self.aruco_detector.detectMarkers(image=image)

        if aruco_ids is None:
            return None

        # iterates over each Aruco corner to find the sub-pixel accurate location
        criteria = (
            cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER,
            100,
            0.00001,
        )
        for aruco_corner in aruco_corners:
            cv2.cornerSubPix(
                image=image,
                corners=aruco_corner,
                winSize=(3, 3),
                zeroZone=(-1, -1),
                criteria=criteria,
            )

        charuco_corners, charuco_ids, _, _ = self.charuco_detector.detectBoard(
            image=image,
            markerCorners=aruco_corners,
            markerIds=aruco_ids,
        )

        # TODO: Do subpix on charuco corners

        if charuco_corners is None or charuco_ids is None:
            return None

        return CharucoBoardDetection(
            aruco_marker_corners=aruco_corners,
            aruco_marker_ids=aruco_ids,
            charuco_corners=charuco_corners,
            charuco_ids=charuco_ids,
        )

    def calibrate_camera(
        self,
        image_size: Tuple[int, ...],
        detections: Iterable[CharucoBoardDetection],
    ) -> Optional[CameraParameter]:
        """
        Calibrates the camera using Charuco detections.

        It collects data from multiple image and detection pairs and computes the camera's intrinsic parameters and
        distortion coefficients.
        The calibration results are returned as a CameraParameter object.

        Args:
            image_size (Tuple[int, ...]): The size of the image used for calibration.
            detections (Iterable[CharucoBoardDetection]): The Charuco board detections.

        Returns:
            CameraParameter: The camera calibration results.
        """

        all_corners_flatten: List[npt.NDArray[Any]] = []
        all_ids_flatten: List[npt.NDArray[Any]] = []

        for detection in detections:
            if detection.charuco_corners is None or detection.charuco_ids is None:
                continue
            all_corners_flatten.append(detection.charuco_corners)
            all_ids_flatten.append(detection.charuco_ids)

        if not all_corners_flatten or not all_ids_flatten:
            tqdm.write("WARNING: No charuco corners found in the detections")
            return None

        (
            error,
            camera_matrix,
            distortion_coeffs,
            _,
            _,
            intrinsics_std,
            extrinsics_std,
            per_view_errors,
        ) = cv2.aruco.calibrateCameraCharucoExtended(
            charucoCorners=all_corners_flatten,
            charucoIds=all_ids_flatten,
            board=self.board,
            imageSize=image_size,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001),
            # default value for type check
            cameraMatrix=np.zeros((3, 3), dtype=np.float64),
            distCoeffs=np.zeros((1, 5), dtype=np.float64),
        )

        # TODO: iterate to remove high error views (images)?

        # refine the camera matrix based on a free scaling parameter for undistortion
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix,
            distortion_coeffs,
            imageSize=image_size,
            alpha=1,  # all pixels are retained with some extra black images
            newImgSize=image_size,
        )

        return CameraParameter(
            # TODO: should change the typing in the class?
            intrinsic_mat=new_camera_matrix.astype(np.float64, copy=False),
            distortion_coeffs=distortion_coeffs.astype(np.float64, copy=False),
        )

    @staticmethod
    def draw_detection(
        image: cvt.MatLike, detection: CharucoBoardDetection
    ) -> cvt.MatLike:
        """
        Draw charuco detection on an image

        Args:
            image (cvt.MatLike): image to be plotted
            detection (CharucoBoardDetection): charuco detection

        Returns:
            cvt.MatLike: image plotted with charuco detection
        """
        plotted = image.copy()
        cv2.aruco.drawDetectedMarkers(
            image=plotted,
            corners=detection.aruco_marker_corners,
            ids=detection.aruco_marker_ids,
        )
        if detection.charuco_corners is None or detection.charuco_ids is None:
            tqdm.write("WARNING: No charuco corners found in the detection")
            return plotted
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
    detection = board.detect(test_image)
    if (
        detection is None
        or detection.charuco_corners is None
        or detection.charuco_ids is None
    ):
        print("No detections found")
        exit(1)
    n_markers = len(detection.aruco_marker_ids)
    n_corners = len(detection.charuco_ids)

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

    detections: List[CharucoBoardDetection] = []
    write_dir = test_calib_image_dir / "plotted"
    write_dir.mkdir(exist_ok=True)
    for image_path in tqdm(test_calib_image_paths, desc="Detecting charuco corners:"):
        image = cv2.imread(str(image_path))
        detection = board.detect(image)
        if detection is None:
            print(f"No detections found in {image_path}")
            continue
        plotted = CharucoBoard.draw_detection(image=image, detection=detection)
        # write to file
        cv2.imwrite(str(write_dir / f"{image_path.stem}.plotted.jpg"), plotted)

        detections.append(detection)

    image_shape = cv2.imread(str(test_calib_image_paths[0])).shape[:2]
    calibration = board.calibrate_camera(image_shape, detections)

    np.set_printoptions(precision=8, suppress=True, floatmode="fixed")
    print("Camera matrix:")
    if calibration is None:
        print("Calibration failed")
        exit(1)
    print(calibration.intrinsic_mat.dtype)
    print(calibration.intrinsic_mat)
    if calibration.distortion_coeffs is None:
        print("No distortion coefficients")
        exit(1)
    print("Distortion coefficients:")
    print(calibration.distortion_coeffs.dtype)
    print(calibration.distortion_coeffs)
