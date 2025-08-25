from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class Alert:
    frame_idx: int
    timestamp_s: float
    type: str
    track_id: int | None
    score: float | None = None


def evaluate_heuristics(
    frame_idx: int,
    fps: float,
    person_tracks: Dict[int, any],
    object_tracks: Dict[int, any],
    loiter_seconds: float,
    abandonment_seconds: float,
    stationary_pixel_tolerance: int,
) -> List[Alert]:
    alerts: List[Alert] = []
    # Loitering: person visible and relatively stationary for >= threshold
    loiter_frames = int(loiter_seconds * fps)
    for tid, tr in person_tracks.items():
        if tr.total_visible_frames >= loiter_frames and tr.stationary_frames >= int(0.7 * loiter_frames):
            alerts.append(Alert(frame_idx=frame_idx, timestamp_s=frame_idx / fps, type="loitering", track_id=tid))

    # Object abandonment: object stationary for threshold without a nearby person
    abandon_frames = int(abandonment_seconds * fps)
    for tid, tr in object_tracks.items():
        if tr.stationary_frames >= abandon_frames:
            alerts.append(Alert(frame_idx=frame_idx, timestamp_s=frame_idx / fps, type="object_abandonment", track_id=tid))

    return alerts


