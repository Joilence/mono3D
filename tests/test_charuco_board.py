"""Unit tests for the CharucoBoard class"""

from typing import Any, Dict

import cv2
import numpy as np
import numpy.typing as npt
import pytest

from mono3d.charuco_board import CharucoBoard, CharucoBoardDetection


@pytest.fixture
def charuco_board_image() -> npt.NDArray[np.uint8]:
    """
    Fixture to load the charuco board image
    """
    return cv2.imread("images/charuco_board.jpg")


@pytest.mark.parametrize(
    "test_id, configs",
    [
        (
                "happy_path_custom_dict",
                {"aruco_dict": cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250),  # Custom value
                 "n_horizontal_squares": 7,
                 "n_vertical_squares": 5,
                 "board_square_length": 60.0,
                 "aruco_marker_length": 45.0}
        ),
    ])
def test_CharucoBoard_init(test_id: str, configs: Dict[str, Any]):
    # Arrange
    def test_aruco_dict_equal(aruco_dict1, aruco_dict2):
        return (aruco_dict1.bytesList == aruco_dict2.bytesList).all()

    # Act
    board = CharucoBoard(**configs)

    # Assert
    assert board.n_horizontal_squares == configs["n_horizontal_squares"], f"Failed on {test_id} - n_horizontal_squares"
    assert board.n_vertical_squares == configs["n_vertical_squares"], f"Failed on {test_id} - n_vertical_squares"
    assert board.board_square_length == configs["board_square_length"], f"Failed on {test_id} - board_square_length"
    assert board.aruco_marker_length == configs["aruco_marker_length"], f"Failed on {test_id} - aruco_marker_length"
    assert test_aruco_dict_equal(board.aruco_dict, configs["aruco_dict"]), f"Failed on {test_id} - aruco_dict"


def test_detect_empty_image():
    # Arrange
    board = CharucoBoard()  # TODO: should be fixture
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    # Act
    n_charuco_corners, detection = board.detect(image)

    # Assert
    assert n_charuco_corners == 0, f"Failed on empty image - n_charuco_corners should be 0, got {n_charuco_corners}"
    assert detection is None, f"Failed on empty image - detection should be None, got {detection}"


def test_detect():
    # TODO: test on all images in the images folder
    # Arrange
    board = CharucoBoard()  # TODO: should be fixture
    image = cv2.imread("tests/images/charuco_board.jpg")  # TODO: should be fixture

    aruco_marker_ids = np.array([[20], [15], [10], [5], [18], [13],
                                 [8], [21], [16], [11], [9], [12],
                                 [0], [23], [3], [6], [1], [24],
                                 [19], [14], [22], [17], [7], [2]])

    charuco_ids = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8],
                            [9], [10], [11], [12], [13], [14], [15], [16],
                            [17], [18], [19], [20], [21], [22], [23], [24],
                            [25], [26], [27], [28], [29], [32], [33]])

    # Act
    n_charuco_corners, detection = board.detect(image)

    # Assert
    assert n_charuco_corners == 32, f"Failed on happy path - n_charuco_corners should be 32, got {n_charuco_corners}"
    assert isinstance(detection, CharucoBoardDetection), \
        f"Failed on happy path - detection should be a CharucoBoardDetection instance, got {detection}"
    assert (detection.aruco_marker_ids == aruco_marker_ids).all(), \
        f"Failed on happy path - aruco_marker_ids should be {aruco_marker_ids}, got {detection.aruco_marker_ids}"
    assert (detection.charuco_ids == charuco_ids).all(), \
        f"Failed on happy path - charuco_ids should be {charuco_ids}, got {detection.charuco_ids}"

# @pytest.mark.parametrize("test_id, detections, expected_calibration", [
#     ("calibrate_camera_happy_path",
#      [(np.zeros((100, 100, 3), dtype=np.uint8), CharucoBoardDetection(np.empty((1,)), np.empty((4, 2,)), None, None))],
#      CameraCalibration(None, None)),
#     # Add more test cases for various realistic values, edge cases, and error cases
# ])
# def test_calibrate_camera(test_id, detections, expected_calibration):
#     # Arrange
#     board = CharucoBoard()
#
#     # Act
#     calibration_result = board.calibrate_camera(detections)
#
#     # Assert
#     assert isinstance(calibration_result,
#                       CameraCalibration), f"Failed on {test_id} - result is not a CameraCalibration instance"
