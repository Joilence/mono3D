"""Charuco Board Detection dataclass"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Any, Sequence

import numpy as np
import cv2.typing as cvt


@dataclass(frozen=True)
class CharucoBoardDetection:
    """ Structured storage of detections on a Charuco board """
    aruco_marker_ids: cvt.MatLike  # expected dtype: np.int32, shape: (n, 1)
    aruco_marker_corners: Sequence[cvt.MatLike]  # expected dtype: Sequence[np.float32], shape: (n, 4, 2)
    charuco_ids: Optional[cvt.MatLike] = None  # expected dtype: np.int32, shape: (n, 1)
    charuco_corners: Optional[cvt.MatLike] = None  # expected dtype: np.float32, shape: (n, 4, 2)

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
