"""Synchronized camera observations for offline visual world models."""

import numpy as np

from kuafu.data.collection.state_recorder import StateTransitionRecorder, _array


CAMERAS = ("table_cam_0", "table_cam_1")


def configure_rgb_cameras(cfg, width=128, height=72):
    """Keep calibrated poses and field of view; render RGB at a smaller resolution."""
    if width < 1 or height < 1:
        raise ValueError("Camera dimensions must be positive")
    for key in CAMERAS:
        camera = getattr(cfg, key)
        if abs(width / height - camera.width / camera.height) > 1e-6:
            raise ValueError("RGB capture must preserve the calibrated camera aspect ratio")
        camera.width, camera.height = width, height
        camera.data_types = ["rgb"]
    cfg.enable_cameras = True


class RGBTransitionRecorder(StateTransitionRecorder):
    """Record pre-action RGB beside applied joint commands, with no reset frames.

    Only obs/rgb is stored. Consecutive rows supply successors, so the visual
    loader drops the final unpaired action rather than fabricating an image.
    State fields remain available for auditing and are not visual-model inputs.
    """

    def __init__(self, env, path, target, metadata=None):
        super().__init__(env, path, target, {
            **(metadata or {}), "camera_keys": list(CAMERAS),
            "rgb_timing": "pre_action", "rgb_layout": "T,C,H,W,3",
            "rgb_successor": "next row within the same successful episode",
        })

    def record_pre_step(self):
        super().record_pre_step()
        # Explicit refresh also covers asynchronous resets, before any physics.
        self.env.sim.render()
        frames = []
        for key in CAMERAS:
            camera = self.env.scene.sensors[key]
            camera.update(0.0, force_recompute=True)
            frame = _array(camera.data.output["rgb"])[..., :3]
            if frame.dtype != np.uint8 or frame.ndim != 4:
                raise ValueError(f"{key}: expected uint8 [env,height,width,3] RGB")
            frames.append(frame)
        frames = np.stack(frames, axis=1)
        for index, row in enumerate(self.pending):
            row["obs/rgb"] = frames[index]
