import importlib.util
import os

import pytest

def require_stage_runtime():
    for name in ("torch", "diffusers", "zarr"):
        pytest.importorskip(name)
    if not os.environ.get("KUAFU2REAL_ROOT") and importlib.util.find_spec("kuafu2real") is None:
        pytest.skip("Configure optional Kuafu2Real/DP3 checkouts", allow_module_level=True)
    from kuafu.sim2real.dp3.runtime import configure
    configure()


@pytest.fixture(autouse=True)
def small_thread_pool():
    import torch
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def policy_kwargs():
    from diffusers import DDIMScheduler
    from omegaconf import OmegaConf
    return dict(
        shape_meta=OmegaConf.create({"obs": {"point_cloud": {"shape": [12, 3]}, "agent_pos": {"shape": [10]}},
                    "action": {"shape": [10]}}),
        noise_scheduler=DDIMScheduler(num_train_timesteps=10, prediction_type="sample"),
        horizon=8, n_obs_steps=2, n_action_steps=3, num_inference_steps=2,
        diffusion_step_embed_dim=16, down_dims=[16, 32], n_groups=4, kernel_size=3,
        encoder_output_dim=16, pointcloud_encoder_cfg=OmegaConf.create({
            "in_channels": 3, "out_channels": 16, "use_layernorm": True, "final_norm": "layernorm"}),
    )


def make_policy(variant="C", *, simple=False):
    import numpy as np
    import torch
    from kuafu.sim2real.dp3.policy import StageDP3, StageSimpleDP3
    from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
    torch.manual_seed(19)
    policy = StageSimpleDP3 if simple else StageDP3
    model = policy(variant=variant, num_stages=3, stage_hidden_dim=8, stage_embedding_dim=4,
                   **policy_kwargs())
    rng = np.random.default_rng(10)
    normalizer = LinearNormalizer()
    normalizer.fit({"point_cloud": rng.normal(size=(4, 12, 3)).astype(np.float32),
                    "agent_pos": rng.normal(size=(4, 10)).astype(np.float32),
                    "action": rng.normal(size=(4, 10)).astype(np.float32)}, last_n_dims=1)
    model.set_normalizer(normalizer)
    return model


@pytest.fixture
def batch():
    import torch
    torch.manual_seed(2)
    return {"obs": {"point_cloud": torch.randn(2, 2, 12, 3), "agent_pos": torch.randn(2, 2, 10)},
            "action": torch.randn(2, 8, 10), "stage_id": torch.tensor([0, 2]),
            "action_valid": torch.ones(2, 8, dtype=torch.bool)}
