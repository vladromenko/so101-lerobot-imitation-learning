"""Shared state. New geometric values are metres, radians and seconds."""
from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class ObjectState:
    id: str
    color: str
    mask: Any
    pose: np.ndarray
    dimensions: np.ndarray
    confidence: float
    timestamp: float
    status: str = 'observed'
    evidence: dict = field(default_factory=dict)


@dataclass
class BoxState:
    id: str
    pose: np.ndarray
    inner_dimensions: np.ndarray
    wall_height: float | None
    occupied_slots: list = field(default_factory=list)
    evidence: dict = field(default_factory=dict)


@dataclass
class SceneState:
    objects: dict = field(default_factory=dict)
    boxes: dict = field(default_factory=dict)
    obstacles: list = field(default_factory=list)
    surfaces: list = field(default_factory=list)
    version: int = 0
    timestamp: float = 0.
    evidence: dict = field(default_factory=dict)


@dataclass
class GraspCandidate:
    tool_pose: np.ndarray
    opening: float
    pregrasp_pose: np.ndarray
    retreat_direction: np.ndarray
    joint_solution: np.ndarray
    clearance: float
    score: tuple
    native: Any = None


@dataclass
class Placement:
    object_pose: np.ndarray
    support_surface: str
    tolerance: float = .008


@dataclass
class ActionResult:
    success: bool
    reason: str
    object_id: str | None
    updated_scene: SceneState
    evidence: dict = field(default_factory=dict)


@dataclass
class Frame:
    camera_id: str
    frame_id: str
    image: np.ndarray
    timestamp: float
    received_at: float
    capture_interval: tuple
    joints: np.ndarray | None = None
    feedback_error_s: float | None = None
