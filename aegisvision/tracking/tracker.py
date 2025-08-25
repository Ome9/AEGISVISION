from dataclasses import dataclass
from typing import Dict, Tuple, List
import math


@dataclass
class Track:
    id: int
    bbox: Tuple[float, float, float, float]
    last_seen_frame: int
    first_seen_frame: int
    total_visible_frames: int
    last_center: Tuple[float, float]
    stationary_frames: int = 0


class SimpleTracker:
    def __init__(self, max_distance: float = 80.0, max_lost_frames: int = 30):
        self.max_distance = max_distance
        self.max_lost_frames = max_lost_frames
        self.next_id = 1
        self.tracks: Dict[int, Track] = {}

    def _center(self, bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _distance(self, c1: Tuple[float, float], c2: Tuple[float, float]) -> float:
        return math.hypot(c1[0] - c2[0], c1[1] - c2[1])

    def update(self, frame_idx: int, detections: List[Tuple[float, float, float, float]]) -> Dict[int, Track]:
        assigned = set()
        # Greedy assignment by nearest center
        for det in detections:
            det_c = self._center(det)
            best_id = None
            best_dist = float("inf")
            for tid, tr in self.tracks.items():
                dist = self._distance(det_c, tr.last_center)
                if dist < best_dist and dist <= self.max_distance:
                    best_dist = dist
                    best_id = tid
            if best_id is None:
                # new track
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = Track(
                    id=tid,
                    bbox=det,
                    last_seen_frame=frame_idx,
                    first_seen_frame=frame_idx,
                    total_visible_frames=1,
                    last_center=det_c,
                    stationary_frames=0,
                )
                assigned.add(tid)
            else:
                tr = self.tracks[best_id]
                stationary = self._distance(det_c, tr.last_center) < 5.0
                tr.stationary_frames = tr.stationary_frames + 1 if stationary else 0
                tr.bbox = det
                tr.last_seen_frame = frame_idx
                tr.total_visible_frames += 1
                tr.last_center = det_c
                assigned.add(best_id)

        # Remove stale tracks
        to_remove = []
        for tid, tr in self.tracks.items():
            if tid not in assigned and (frame_idx - tr.last_seen_frame) > self.max_lost_frames:
                to_remove.append(tid)
        for tid in to_remove:
            self.tracks.pop(tid, None)

        return self.tracks


