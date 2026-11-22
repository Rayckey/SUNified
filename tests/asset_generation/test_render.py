import numpy as np

from kuafu.asset_generation.render import shade_srgb


def test_srgb_preview_preserves_unlit_input_at_unit_light():
    colors = np.array([[0, 10, 255], [204, 189, 111]], dtype=np.uint8)
    assert np.array_equal(shade_srgb(colors, np.ones(2)), colors)
    assert not shade_srgb(colors, np.zeros(2)).any()


def test_srgb_preview_does_not_multiply_encoded_bytes_as_if_linear():
    result = shade_srgb(np.array([[255, 128, 0]]), np.array([0.5]))
    assert result[0].tolist() == [188, 92, 0]
