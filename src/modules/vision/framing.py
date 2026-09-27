"""Vision framing engine computing active speaker tracking and 9:16 crop trajectories."""

from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np

from src.core.contracts import CropBox, LayoutMode
from src.core.events import JobContext


class VisionFramingEngine:
    """Detects speakers, computes camera center, and produces smoothed 9:16 crop trajectories."""

    def __init__(self, context: JobContext, alpha: float = 0.15, deadband_px: float = 18.0):
        self.context = context
        self.alpha = alpha
        self.deadband_px = deadband_px

    def compute_crop_trajectory(
        self,
        video_path: Path,
        start_time: float,
        end_time: float,
        layout_mode: LayoutMode = LayoutMode.SINGLE_SPEAKER_SMOOTH,
    ) -> List[CropBox]:
        """
        Generate time-series crop trajectory [(timestamp, crop_x, crop_y, crop_w, crop_h)].
        Converts 16:9 source (e.g. 1920x1080) to vertical 9:16 framing.
        """
        self.context.log_info(
            f"Computing vision crop trajectory from {start_time:.2f}s to {end_time:.2f}s (mode: {layout_mode.value})..."
        )

        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                self.context.log_warning(f"Could not open video file {video_path}. Using static center crop.")
                return self._fallback_center_crops(start_time, end_time)

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)

            # 9:16 crop dimensions inside source frame
            crop_h = frame_height
            crop_w = int(crop_h * 9 / 16)
            if crop_w % 2 != 0:
                crop_w += 1

            max_x = max(0, frame_width - crop_w)
            default_center_x = max_x // 2

            # Initialize MediaPipe Face Detection
            mp_face_detection = None
            face_detection = None
            try:
                import mediapipe as mp
                mp_face_detection = mp.solutions.face_detection
                # model_selection=1 is suited for faces further away than 2 meters
                face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
                self.context.log_info("MediaPipe FaceDetection initialized successfully.")
            except ImportError:
                self.context.log_warning("MediaPipe not installed. Falling back to motion detection.")

            # Sampling at 15fps
            sample_step_frames = max(1, int(fps / 15))
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            current_frame_idx = start_frame

            smoothed_x = float(default_center_x)
            trajectory: List[CropBox] = []
            prev_gray = None

            while current_frame_idx <= end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                current_timestamp = current_frame_idx / fps

                # Only analyze sampled frames for efficiency
                if (current_frame_idx - start_frame) % sample_step_frames == 0:
                    target_x = None
                    try:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        # Tier 1: MediaPipe AI Face Tracking
                        if face_detection is not None:
                            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            results = face_detection.process(rgb)
                            if results.detections:
                                largest_face = max(
                                    results.detections, 
                                    key=lambda d: d.location_data.relative_bounding_box.width * d.location_data.relative_bounding_box.height
                                )
                                bbox = largest_face.location_data.relative_bounding_box
                                face_center_x_rel = bbox.xmin + (bbox.width / 2)
                                target_x = int((face_center_x_rel * frame_width) - (crop_w // 2))

                        # Tier 2: Dynamic visual motion/activity centroid tracking (fallback)
                        if target_x is None and prev_gray is not None:
                            diff = cv2.absdiff(prev_gray, gray)
                            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                            M = cv2.moments(thresh)
                            if M["m00"] > 1000:
                                motion_x = int(M["m10"] / M["m00"])
                                target_x = motion_x - (crop_w // 2)

                        prev_gray = gray
                    except Exception:
                        pass

                    if target_x is not None:
                        target_x = max(0, min(max_x, target_x))
                        delta = abs(target_x - smoothed_x)
                        if delta > self.deadband_px:
                            smoothed_x = (self.alpha * target_x) + ((1.0 - self.alpha) * smoothed_x)
                    else:
                        # Slowly drift toward frame center when stationary
                        smoothed_x = (0.05 * default_center_x) + (0.95 * smoothed_x)

                int_crop_x = int(round(smoothed_x))
                if int_crop_x % 2 != 0:
                    int_crop_x -= 1
                int_crop_x = max(0, min(max_x, int_crop_x))

                trajectory.append(
                    CropBox(
                        timestamp=round(current_timestamp, 3),
                        crop_x=int_crop_x,
                        crop_y=0,
                        crop_w=crop_w,
                        crop_h=crop_h,
                    )
                )

                current_frame_idx += 1

            cap.release()

            if not trajectory:
                return self._fallback_center_crops(start_time, end_time, frame_width, frame_height)

            self.context.log_info(f"Generated {len(trajectory)} crop trajectory points.")
            return trajectory

        except Exception as exc:
            self.context.log_warning(f"Vision tracking fallback triggered ({exc}). Using center crop.")
            return self._fallback_center_crops(start_time, end_time)

    def _fallback_center_crops(
        self,
        start_time: float,
        end_time: float,
        frame_width: int = 1920,
        frame_height: int = 1080,
    ) -> List[CropBox]:
        """Provides static 9:16 center crop when video frames cannot be sampled."""
        trajectory: List[CropBox] = []
        crop_h = frame_height
        crop_w = int(crop_h * 9 / 16)
        if crop_w % 2 != 0:
            crop_w += 1
        crop_x = max(0, (frame_width - crop_w) // 2)
        if crop_x % 2 != 0:
            crop_x -= 1

        t = start_time
        while t <= end_time:
            trajectory.append(
                CropBox(
                    timestamp=round(t, 2),
                    crop_x=crop_x,
                    crop_y=0,
                    crop_w=crop_w,
                    crop_h=crop_h,
                )
            )
            t += 0.1
        return trajectory
