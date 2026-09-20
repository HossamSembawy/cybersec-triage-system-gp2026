import joblib
import numpy as np
import pytest
from sklearn.ensemble import IsolationForest

from src.network_module.preprocessor import NetworkPreprocessor
from src.network_module.predictor import NetworkPredictor


FEATURE_NAMES = [
    "Flow Duration",
    "Flow Bytes/s",
    "Flow Packets/s",
]


def fit_synthetic_model():
    random_generator = np.random.default_rng(42)
    benign_training_flows = random_generator.normal(size=(30, 3))
    model = IsolationForest(n_estimators=10, random_state=42)
    model.fit(benign_training_flows)
    return model


def test_dict_input_strips_dataset_column_whitespace():
    preprocessor = NetworkPreprocessor(FEATURE_NAMES)

    vector = preprocessor.build_feature_vector({
        " Flow Packets/s": 30,
        " Flow Duration": 10,
        " Flow Bytes/s": 20,
    })

    np.testing.assert_array_equal(vector, [[10.0, 20.0, 30.0]])


def test_ordered_list_input_preserves_value_order():
    preprocessor = NetworkPreprocessor(FEATURE_NAMES)

    vector = preprocessor.build_feature_vector([10, 20, 30])

    np.testing.assert_array_equal(vector, [[10.0, 20.0, 30.0]])


def test_dict_input_rejects_missing_features():
    preprocessor = NetworkPreprocessor(FEATURE_NAMES)

    with pytest.raises(ValueError, match="Missing 1 feature"):
        preprocessor.build_feature_vector({
            "Flow Duration": 10,
            "Flow Bytes/s": 20,
        })


def test_ordered_list_rejects_wrong_length():
    preprocessor = NetworkPreprocessor(FEATURE_NAMES)

    with pytest.raises(ValueError, match="Expected 3 features, got 2"):
        preprocessor.build_feature_vector([10, 20])


def test_non_numeric_values_are_rejected():
    preprocessor = NetworkPreprocessor(FEATURE_NAMES)

    with pytest.raises(ValueError):
        preprocessor.build_feature_vector({
            "Flow Duration": 10,
            "Flow Bytes/s": "not-a-number",
            "Flow Packets/s": 30,
        })


def test_non_finite_values_are_replaced_with_zero():
    preprocessor = NetworkPreprocessor(FEATURE_NAMES)

    vector = preprocessor.build_feature_vector([
        np.inf,
        -np.inf,
        np.nan,
    ])

    np.testing.assert_array_equal(vector, [[0.0, 0.0, 0.0]])


def test_saved_feature_order_drives_inference_order(tmp_path):
    model = fit_synthetic_model()

    feature_names = ["third", "first", "second"]
    bundle = {
        "model": model,
        "feature_names": feature_names,
        "score_lo": 0.40,
        "score_hi": 0.60,
        "model_version": "network-isolationforest-test",
    }
    joblib.dump(bundle, tmp_path / "network_model.joblib")

    predictor = NetworkPredictor.from_pretrained(tmp_path)
    vector = predictor.preprocessor.build_feature_vector({
        "first": 1,
        "second": 2,
        "third": 3,
    })

    assert predictor.feature_names == feature_names
    np.testing.assert_array_equal(vector, [[3.0, 1.0, 2.0]])


@pytest.mark.parametrize(
    "score_lo_offset, score_hi_offset, expected_probability",
    [
        (0.20, 0.40, 0.0),
        (-0.20, 0.20, 0.5),
        (-0.40, -0.20, 1.0),
        (0.0, 0.0, 0.0),
    ],
)
def test_predictor_uses_saved_normalization_and_clips_to_range(
    score_lo_offset,
    score_hi_offset,
    expected_probability,
):
    model = fit_synthetic_model()
    test_flow = [0.1, 0.2, 0.3]
    raw_score = -model.score_samples(np.array([test_flow]))[0]

    predictor = NetworkPredictor(
        model=model,
        feature_names=FEATURE_NAMES,
        score_lo=raw_score + score_lo_offset,
        score_hi=raw_score + score_hi_offset,
        preprocessor=NetworkPreprocessor(FEATURE_NAMES),
    )

    result = predictor.predict(test_flow)

    assert result.anomaly_probability == pytest.approx(expected_probability)
    assert 0.0 <= result.anomaly_probability <= 1.0
