from pathlib import Path
from typing import Union, Callable, Optional

import cv2
from cv2 import typing as cvt
from tqdm.auto import trange


def process_video(
    src_video_path: Union[Path, str],
    frame_process_func: Callable[[cvt.MatLike], cvt.MatLike],
    dst_dir: Optional[Union[Path, str]] = None,
    scale: float = 1.0,
    output_suffix: str = ".processed",
) -> None:
    """
    Process a video by applying a frame processing function to each frame.

    Args:
        src_video_path (str or Path): Path to the source video file.
        dst_dir (str or Path, optional):
            Path to the folder where the processed video will be saved.
        frame_process_func (Callable[[cv2.Mat], cv2.Mat]):
            Function that takes an image and returns a processed image.
        scale (float, optional):
            Scale factor for resizing the frames, default to 1.0.
        output_suffix (str, optional):
            Suffix to append to the processed video filename before the extension.
            Defaults to ".processed".
    """
    src_video_path = Path(src_video_path)

    vc = cv2.VideoCapture(str(src_video_path))
    width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vc.get(cv2.CAP_PROP_FPS))
    n_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))

    if scale != 1:
        width = int(width * scale)
        height = int(height * scale)

    dst_dir = dst_dir or src_video_path.parent
    output_path = Path(dst_dir) / f"{src_video_path.stem}{output_suffix}.mp4"
    vw = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter.fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    for _ in trange(
        n_frames,
        desc=f"Processing {src_video_path.name}",
        unit="frames",
        bar_format="{l_bar}{bar:10}{r_bar}",
        leave=False,
    ):
        ret, frame = vc.read()
        if not ret:
            break
        processed_frame = frame_process_func(frame)
        # scale image if necessary
        if scale != 1:
            processed_frame = cv2.resize(processed_frame, (width, height))
        vw.write(processed_frame)

    vc.release()
    vw.release()
