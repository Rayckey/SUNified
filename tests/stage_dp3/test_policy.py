from conftest import require_stage_runtime

require_stage_runtime()

import copy

import pytest
import torch

from diffusion_policy_3d.policy.dp3 import DP3
from diffusion_policy_3d.policy.simple_dp3 import SimpleDP3
from conftest import make_policy, policy_kwargs


@pytest.fixture(params=[False, True], ids=["DP3", "SimpleDP3"])
def simple(request):
    return request.param


def nonzero_grad(module):
    return any(p.grad is not None and torch.count_nonzero(p.grad) for p in module.parameters())


def test_baseline_matches_upstream_loss_and_inference(batch, simple):
    model = make_policy("A", simple=simple)
    baseline = (SimpleDP3 if simple else DP3)(**policy_kwargs())
    baseline.load_state_dict(model.state_dict())
    torch.manual_seed(10)
    expected_loss = baseline.compute_loss(batch)[0]
    torch.manual_seed(10)
    actual_loss = model.compute_loss(batch)[0]
    torch.testing.assert_close(actual_loss, expected_loss)
    torch.manual_seed(11)
    expected = baseline.predict_action(batch["obs"])
    torch.manual_seed(11)
    actual = model.predict_action(batch["obs"])
    torch.testing.assert_close(actual["action"], expected["action"])
    assert not any(name.startswith("stage_") for name in model.state_dict())


@pytest.mark.parametrize("variant", ["B", "C"])
def test_classification_trains_head_and_shared_encoder(variant, batch, simple):
    model = make_policy(variant, simple=simple)
    _, stage_loss = model.loss_terms(batch)
    stage_loss.backward()
    assert nonzero_grad(model.stage_head)
    assert nonzero_grad(model.obs_encoder)
    assert not nonzero_grad(model.model)


def test_action_route_detaches_probabilities_but_trains_embedding(batch, simple):
    model = make_policy("C", simple=simple)
    # Once conditioning weights have learned, embeddings must get action gradients.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in model._expanded_weights:
                param[:, -model.stage_embedding_dim:].fill_(0.1)
    action_loss, _ = model.loss_terms(batch)
    action_loss.backward()
    assert not nonzero_grad(model.stage_head)
    assert nonzero_grad(model.obs_encoder)
    assert torch.count_nonzero(model.stage_embeddings.grad)
    assert nonzero_grad(model.model)


def test_initial_conditioning_preserves_baseline_and_head_initialization(batch, simple):
    a, b, c = [make_policy(variant, simple=simple) for variant in "ABC"]
    for name, param in b.stage_head.state_dict().items():
        torch.testing.assert_close(param, c.stage_head.state_dict()[name])
    c.load_baseline(a.state_dict())
    predictions = []
    for model in (a, b, c):
        torch.manual_seed(3)
        predictions.append(model.predict_action(batch["obs"])["action"])
    for prediction in predictions[1:]:
        torch.testing.assert_close(prediction, predictions[0])


def test_ground_truth_is_supervision_only_and_extra_obs_rejected(batch, simple):
    model = make_policy("C", simple=simple)
    torch.manual_seed(7)
    original_action_loss, original_ce = model.loss_terms(batch)
    changed = {**batch, "stage_id": torch.tensor([1, 1])}
    torch.manual_seed(7)
    changed_action_loss, changed_ce = model.loss_terms(changed)
    torch.testing.assert_close(original_action_loss, changed_action_loss)
    assert not torch.isclose(original_ce, changed_ce)
    with pytest.raises(ValueError, match="only"):
        model.predict_action({**batch["obs"], "stage_id": batch["stage_id"]})


def test_auxiliary_prediction_does_not_affect_actions(batch, simple):
    model = make_policy("B", simple=simple)
    torch.manual_seed(8)
    before = model.predict_action(batch["obs"])["action"]
    with torch.no_grad():
        for param in model.stage_head.parameters():
            param.add_(500)
    torch.manual_seed(8)
    torch.testing.assert_close(before, model.predict_action(batch["obs"])["action"])


def test_checkpoint_reload_and_invalid_labels(batch, simple):
    model = make_policy("C", simple=simple)
    loss, _ = model.compute_loss(batch)
    loss.backward()
    torch.optim.Adam(model.parameters()).step()
    restored = make_policy("C", simple=simple)
    restored.load_state_dict(copy.deepcopy(model.state_dict()))
    torch.manual_seed(4)
    expected = model.predict_action(batch["obs"])
    torch.manual_seed(4)
    actual = restored.predict_action(batch["obs"])
    torch.testing.assert_close(actual["action"], expected["action"])
    torch.testing.assert_close(actual["stage_logits"], expected["stage_logits"])
    with pytest.raises(ValueError, match="vocabulary"):
        model.compute_loss({**batch, "stage_id": torch.tensor([3, 0])})
    missing = {**batch, "stage_id": torch.tensor([-1, -1])}
    assert model.loss_terms(missing)[1].item() == 0


def test_padding_has_no_direct_loss_gradient_and_future_history_is_rejected(batch, simple):
    model = make_policy("C", simple=simple)
    batch["action_valid"][:, -3:] = False
    output = []
    def retain_prediction(module, inputs, result):
        result.retain_grad()
        output.append(result)
    hook = model.model.register_forward_hook(retain_prediction)
    model.compute_loss(batch)[0].backward()
    hook.remove()
    assert torch.count_nonzero(output[0].grad[:, :-3])
    assert torch.count_nonzero(output[0].grad[:, -3:]) == 0
    with pytest.raises(ValueError, match="exactly"):
        model.predict_action({key: value.repeat_interleave(2, dim=1) for key, value in batch["obs"].items()})
