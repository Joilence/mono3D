""" CharucoBoard class compatible with both OpenCV above and below 4.6.0 """

from pathlib import Path
from typing import Optional, Tuple, List, Any, Iterable

import cv2
import cv2.typing as cvt
import numpy as np
import numpy.typing as npt
from tqdm.auto import tqdm

from mono3d import CameraParameter
from mono3d.charuco_board_detection import CharucoBoardDetection
from mono3d.globals import LEGACY


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
        detect(image: cvt.MatLike) -> Tuple[int, CharucoBoardDetection]:
            Detects Aruco and Charuco markers in the given image.
        _detect_legacy(image: cvt.MatLike) -> Tuple[int, CharucoBoardDetection]:
            detect() function for OpenCV < 4.6.0
        calibrate_camera(detections: List[Tuple[npt.NDArray[np.uint8],
        CharucoBoardDetection]]) -> CameraParameter:
            Calibrates the camera using Charuco detections.
        draw_detection(image: cvt.MatLike, detection: CharucoBoardDetection) -> cvt.MatLike:
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
            if use_board_legacy_pattern:
                tqdm.write("WARNING: Using the legacy CharucoBoard pattern.")
                self.board.setLegacyPattern(use_board_legacy_pattern)
            self.aruco_detector = cv2.aruco.ArucoDetector(
                dictionary=self.aruco_dict,
                detectorParams=cv2.aruco.DetectorParameters()
            )
            self.charuco_detector = cv2.aruco.CharucoDetector(board=self.board)

    def detect(self, image: cvt.MatLike) -> Optional[CharucoBoardDetection]:
        """
        Detects Aruco and Charuco markers in the given image.
        Detected Aruco markers are refined for better accuracy using cornerSubPix.

        Arg:
            image (cvt.MatLike): The input image for marker detection.

        Returns:
            Tuple[int, CharucoBoardDetection]: A tuple containing the number of detected charuco corners and
                detection details
        """
        # check if image is not empty
        if image.size == 0:
            return None

        # if not gray scale, convert to gray scale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # image = self.sharp(image=image.copy())  # TODO: sharpen image?

        if LEGACY:
            return self._detect_legacy(image=image)

        # While OpenCV >= 4.6.0, and LEGACY is False, aruco_detector and charuco_detector are available
        aruco_corners, aruco_ids, _ = self.aruco_detector.detectMarkers(image=image)  # type: ignore

        if aruco_ids is None:
            return None

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

        if charuco_corners is None or charuco_ids is None:
            return None

        return CharucoBoardDetection(
            aruco_marker_corners=aruco_corners,
            aruco_marker_ids=aruco_ids,
            charuco_corners=charuco_corners,
            charuco_ids=charuco_ids
        )

    def _detect_legacy(self, image: cvt.MatLike) -> Optional[CharucoBoardDetection]:
        """
        detect() function for OpenCV < 4.6.0

        Args:
            image (cvt.MatLike)

        Returns:
            Tuple[int, CharucoBoardDetection]
        """
        aruco_corners, aruco_ids, _ = cv2.aruco.detectMarkers(image=image, dictionary=self.aruco_dict)

        if aruco_ids is None:
            return None

        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=aruco_corners,
            markerIds=aruco_ids,
            image=image,
            board=self.board,
            # number of adjacent markers that must be detected to return a charuco corner
            minMarkers=2
        )

        if charuco_corners is None or charuco_ids is None:
            return None

        # TODO: Do subpix on charuco corners
        return CharucoBoardDetection(
            aruco_marker_corners=aruco_corners,
            aruco_marker_ids=aruco_ids,
            charuco_corners=charuco_corners,
            charuco_ids=charuco_ids
        )

    def calibrate_camera(
            self,
            image_size: Tuple[int, ...],
            detections: Iterable[CharucoBoardDetection],
    ) -> Optional[CameraParameter]:
        """
        Calibrates the camera using Charuco detections.

        It collects data from multiple image and detection pairs and computes the camera
        intrinsic parameters and distortion coefficients. The calibration results are returned
        as a CameraParameter object.

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
            image: cvt.MatLike,
            detection: CharucoBoardDetection
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
            ids=detection.aruco_marker_ids
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
    if detection is None or detection.charuco_corners is None or detection.charuco_ids is None:
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

    np.set_printoptions(precision=8, suppress=True, floatmode='fixed')
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
