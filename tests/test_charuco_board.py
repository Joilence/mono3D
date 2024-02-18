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


@pytest.mark.parametrize(
    "test_id, image, expected_n_charuco_corners, expected_detection",
    [
        ("edge_case_detect_empty_image",
         np.zeros((100, 100, 3), dtype=np.uint8),
         0, CharucoBoardDetection()),
        ("happy_path_detect_charuco_board",
         cv2.imread("tests/images/charuco_board.jpg"),  # TODO: should be fixture
         32, CharucoBoardDetection(
            aruco_marker_ids=np.array([[20], [15], [10], [5], [18], [13],
                                       [8], [21], [16], [11], [9], [12],
                                       [0], [23], [3], [6], [1], [24],
                                       [19], [14], [22], [17], [7], [2]]),
            charuco_ids=np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8],
                                  [9], [10], [11], [12], [13], [14], [15], [16],
                                  [17], [18], [19], [20], [21], [22], [23], [24],
                                  [25], [26], [27], [28], [29], [32], [33]]))
         )
    ]
)
def test_detect(test_id, image, expected_n_charuco_corners, expected_detection):
    # Arrange
    board = CharucoBoard()
    assert image is not None, "Failed to find the charuco board image"

    # Act
    n_charuco_corners, detection = board.detect(image)

    # Assert
    assert n_charuco_corners == expected_n_charuco_corners, f"Failed on {test_id} - n_charuco_corners"
    # partial comparison (only ids) of detection
    for attr in ["aruco_marker_ids", "charuco_ids"]:
        computed = np.asarray(getattr(detection, attr)).reshape(-1)
        expected = np.asarray(getattr(expected_detection, attr)).reshape(-1)
        print(computed, expected)
        assert computed.shape == expected.shape, f"Failed on {test_id} - {attr}: shape mismatch"
        assert (computed == expected).all(), \
            f"Failed on {test_id} - {attr}: {computed} != {expected}"

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
