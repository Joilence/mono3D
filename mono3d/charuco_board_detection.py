from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Any

import numpy as np
from numpy import typing as npt


@dataclass(frozen=True)
class CharucoBoardDetection:
    """ Structured storage of detections on a Charuco board """
    aruco_marker_ids: npt.NDArray[np.int32]  # shape: (n, 1)
    aruco_marker_corners: npt.NDArray[np.float32]  # shape: (n, 4, 2)
    charuco_ids: Optional[npt.NDArray[np.int32]] = None  # shape: (n, 1)
    charuco_corners: Optional[npt.NDArray[np.float32]] = None  # shape: (n, 4, 2)

    def save_to(self, file_path: Union[str, Path]):
        """ Save camera parameters to a file """
        if self.charuco_ids is None or self.charuco_corners is None:
            np.savez(
                str(file_path),
                aruco_marker_ids=self.aruco_marker_ids,
                aruco_marker_corners=self.aruco_marker_corners,
            )
        else:
            np.savez(
                str(file_path),
                aruco_marker_ids=self.aruco_marker_ids,
                aruco_marker_corners=self.aruco_marker_corners,
                charuco_ids=self.charuco_ids,
                charuco_corners=self.charuco_corners,
            )

    @staticmethod
    def load_from(file_path: Union[str, Path]) -> "CharucoBoardDetection":
        """ Load camera parameters from a file """
        data = np.load(str(file_path))
        return CharucoBoardDetection(
            data['aruco_marker_ids'],
            data['aruco_marker_corners'],
            data.get('charuco_ids'),
            data.get('charuco_corners'),
        )

    def has_equal_ids(self, other: Any) -> bool:
        """ Check if the ids of the detections are equal, for testing """
        if not isinstance(other, CharucoBoardDetection):
            return False
        equal_aruco_ids = (np.asarray(self.aruco_marker_ids) == np.asarray(other.aruco_marker_ids)).all()
        equal_charuco_ids = (np.asarray(self.charuco_ids) == np.asarray(other.charuco_ids)).all()
        return equal_aruco_ids and equal_charuco_ids


if __name__ == "__main__":
    # Arrange
    aruco_marker_ids = np.array([25, 26, 27, 28, 29, 32, 33])
    aruco_marker_corners = np.array([[[0, 0], [1, 0], [1, 1], [0, 1]]])
    charuco_ids = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                            25, 26, 27, 28, 29, 30, 31])
    charuco_corners = np.array([[[0, 0], [1, 0], [1, 1], [0, 1]]])

    # Act
    detection = CharucoBoardDetection(aruco_marker_ids, aruco_marker_corners, charuco_ids, charuco_corners)

    # Assert
    assert (detection.aruco_marker_ids == aruco_marker_ids).all(), \
        f"Failed on happy path - aruco_marker_ids should be {aruco_marker_ids}, got {detection.aruco_marker_ids}"
    assert (detection.aruco_marker_corners == aruco_marker_corners).all(), \
        f"Failed on happy path - aruco_marker_corners should be {aruco_marker_corners}, got {detection.aruco_marker_corners}"
    assert (detection.charuco_ids == charuco_ids).all(), \
        f"Failed on happy path - charuco_ids should be {charuco_ids}, got {detection.charuco_ids}"
    assert (detection.charuco_corners == charuco_corners).all(), \
        f"Failed on happy path - charuco_corners should be {charuco_corners}, got {detection.charuco_corners}"
