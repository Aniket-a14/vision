import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from defectlab.imaging.degrade import InlineCamera, Regime, apply_regime, degrade  # noqa: E402


@pytest.fixture
def synthetic() -> np.ndarray:
    rng = np.random.default_rng(0)
    image = np.full((300, 300), 120, dtype=np.uint8)
    image[100:200, 100:200] = 40
    image += rng.integers(0, 15, image.shape, dtype=np.uint8)
    return image


def test_degradation_preserves_shape_and_dtype(synthetic):
    out = degrade(synthetic, np.random.default_rng(1), InlineCamera())
    assert out.shape == synthetic.shape
    assert out.dtype == np.uint8


def test_degradation_removes_high_frequency_detail(synthetic):
    out = degrade(synthetic, np.random.default_rng(1), InlineCamera())
    assert cv2.Laplacian(out, cv2.CV_64F).var() < cv2.Laplacian(synthetic, cv2.CV_64F).var()


def test_lab_regime_is_a_passthrough(synthetic):
    out = apply_regime(synthetic, Regime.LAB, np.random.default_rng(1))
    np.testing.assert_array_equal(out, synthetic)


def test_inline_regime_changes_the_image(synthetic):
    out = apply_regime(synthetic, Regime.INLINE, np.random.default_rng(1))
    assert not np.array_equal(out, synthetic)


def test_zero_severity_is_a_passthrough(synthetic):
    out = degrade(synthetic, np.random.default_rng(1), InlineCamera(severity=0.0))
    np.testing.assert_array_equal(out, synthetic)


def test_degradation_is_reproducible_for_a_seed(synthetic):
    first = degrade(synthetic, np.random.default_rng(7), InlineCamera())
    second = degrade(synthetic, np.random.default_rng(7), InlineCamera())
    np.testing.assert_array_equal(first, second)


def test_severity_increases_pixel_noise(synthetic):
    mild = degrade(synthetic, np.random.default_rng(3), InlineCamera(severity=0.2))
    harsh = degrade(synthetic, np.random.default_rng(3), InlineCamera(severity=3.0))
    assert harsh.std() > mild.std()


def test_output_stays_in_byte_range(synthetic):
    out = degrade(synthetic, np.random.default_rng(5), InlineCamera(severity=4.0))
    assert out.min() >= 0
    assert out.max() <= 255
