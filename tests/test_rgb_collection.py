"""Camera refresh and action alignment in visual transition recording."""

from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from kuafu.data.collection.rgb_recorder import RGBTransitionRecorder, configure_rgb_cameras


def test_rgb_capture_is_pre_action_and_never_includes_reset_frames(tmp_path):
    pixels = torch.zeros((1, 4, 8, 4), dtype=torch.uint8)
    refreshed = []
    camera = SimpleNamespace(data=SimpleNamespace(output={"rgb": pixels}),
                             update=lambda dt, **kw: refreshed.append(kw))
    env = SimpleNamespace(num_envs=1, obs_buf={"full_state": torch.zeros(1, 2)},
                          _logged_relative_joint=torch.ones(1, 9), reward_buf=torch.zeros(1),
                          _cached_success=torch.tensor([True]),
                          scene=SimpleNamespace(sensors={"table_cam_0": camera, "table_cam_1": camera}),
                          sim=SimpleNamespace(render=lambda: None))
    path = tmp_path / "rgb.hdf5"
    recorder = RGBTransitionRecorder(env, path, 1)
    for step in range(3):
        pixels.fill_(step + 10)
        recorder.record_pre_step()
        pixels.fill_(200)
        env.obs_buf["full_state"].add_(1)
        recorder.record_post_step()
    recorder.record_pre_step()
    recorder.record_pre_reset(torch.tensor([0]))
    pixels.fill_(255)
    recorder.record_post_step()
    recorder.close()
    assert all(x == {"force_recompute": True} for x in refreshed)
    with h5py.File(path) as source:
        demo = source["data/demo_0"]
        assert demo["obs/rgb"].shape == (3, 2, 4, 8, 3)
        np.testing.assert_array_equal(demo["obs/rgb"][:, 0, 0, 0, 0], [10, 11, 12])
        assert "next_obs/rgb" not in demo
        assert demo.attrs["success"]


def test_rgb_resolution_preserves_calibrated_field_of_view():
    cameras = [SimpleNamespace(width=640, height=360, data_types=["rgb", "depth"], spawn=object())
               for _ in range(2)]
    cfg = SimpleNamespace(table_cam_0=cameras[0], table_cam_1=cameras[1], enable_cameras=False)
    spawns = [camera.spawn for camera in cameras]
    configure_rgb_cameras(cfg)
    assert cfg.enable_cameras
    assert all((c.width, c.height, c.data_types) == (128, 72, ["rgb"]) for c in cameras)
    assert [c.spawn for c in cameras] == spawns
    with pytest.raises(ValueError, match="aspect ratio"):
        configure_rgb_cameras(cfg, 64, 64)
