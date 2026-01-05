import json
import tempfile
from pathlib import Path

import numpy as np

from src.config import CONFIG
from src.export import build_psdm_payload_from_models, export_psdm_json


def test_export_payload_from_models(small_df, tiny_models):
    p, gmms = tiny_models

    payload = build_psdm_payload_from_models(small_df, p, gmms)

    required_keys = [
        "schema",
        "generated_at",
        "generator",
        "time_model",
        "value_model",
        "parameters",
        "data",
    ]
    for key in required_keys:
        assert key in payload

    assert payload["schema"] == "simonaMarkovLoad:psdm:1.0"

    generator = payload["generator"]
    assert generator["name"] == "simonaMarkovLoad"
    assert generator["version"] == "git:unknown"
    assert "config" in generator

    n_states = CONFIG["model"]["n_states"]
    laplace_alpha = CONFIG["model"]["laplace_alpha"]
    assert generator["config"]["n_states"] == n_states
    assert generator["config"]["laplace_alpha"] == laplace_alpha

    time_model = payload["time_model"]
    assert time_model["bucket_count"] == 2304
    assert "bucket_encoding" in time_model
    assert (
        time_model["bucket_encoding"]["formula"]
        == "bucket = month*192 + is_weekend*96 + quarter_hour"
    )
    assert time_model["sampling_interval_minutes"] == 15
    assert time_model["timezone"] == "Europe/Berlin"

    value_model = payload["value_model"]
    assert value_model["value_unit"] == "normalized"
    assert value_model["normalization"]["method"] == "minmax_per_series"

    discretization = value_model["discretization"]
    assert discretization["states"] == n_states

    expected_thresholds = [(k / 10) ** 2 for k in range(1, 10)]
    assert discretization["thresholds_right"] == expected_thresholds

    parameters = payload["parameters"]
    assert parameters["transitions"]["empty_row_strategy"] == "self_loop"
    assert "gmm" in parameters

    data = payload["data"]
    assert "transitions" in data
    assert "gmms" in data

    transitions = data["transitions"]
    assert transitions["shape"] == list(p.shape)
    assert transitions["shape"] == [2304, n_states, n_states]
    assert transitions["dtype"] == "float32"
    assert transitions["encoding"] == "nested_lists"
    assert "values" in transitions

    gmms_data = data["gmms"]
    assert "buckets" in gmms_data
    assert len(gmms_data["buckets"]) == 2304

    json_str = json.dumps(payload)
    parsed_payload = json.loads(json_str)

    assert parsed_payload["schema"] == payload["schema"]
    assert parsed_payload["time_model"]["bucket_count"] == 2304


def test_export_payload_with_metadata(small_df, tiny_models):
    p, gmms = tiny_models

    meta = {
        "source": "test_data",
        "records": len(small_df),
        "time_range": {"start": "2024-01-01T00:00:00", "end": "2024-02-29T23:45:00"},
    }

    gmm_params = {
        "max_components": 2,
        "min_samples_per_state": 10,
        "covariance_type": "diag",
        "random_seed": 123,
    }

    payload = build_psdm_payload_from_models(
        small_df, p, gmms, meta=meta, gmm_params=gmm_params
    )

    assert "training_data" in payload
    assert payload["training_data"] == meta

    assert payload["parameters"]["gmm"] == gmm_params


def test_export_payload_with_reference_power(small_df, tiny_models):
    p, gmms = tiny_models

    payload = build_psdm_payload_from_models(
        small_df,
        p,
        gmms,
        reference_power_kw=4.5,
        min_power_kw=0.2,
    )

    normalization = payload["value_model"]["normalization"]
    assert normalization["reference_power"] == {"value": 4.5, "unit": "kW"}
    assert normalization["min_power"] == {"value": 0.2, "unit": "kW"}


def test_export_json_file_write(small_df, tiny_models):
    p, gmms = tiny_models

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_export.json"

        result_path = export_psdm_json(
            output_path,
            small_df,
            p,
            gmms,
            pretty=True,
            reference_power_kw=3.3,
        )

        assert result_path == output_path
        assert output_path.exists()
        assert output_path.is_file()

        with open(output_path, encoding="utf-8") as f:
            content = f.read()
            parsed_data = json.loads(content)

        assert parsed_data["schema"] == "simonaMarkovLoad:psdm:1.0"
        assert parsed_data["time_model"]["bucket_count"] == 2304

        assert "\n" in content
        assert "  " in content


def test_export_json_compact(small_df, tiny_models):
    p, gmms = tiny_models

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "compact_export.json"

        export_psdm_json(
            output_path,
            small_df,
            p,
            gmms,
            pretty=False,
            reference_power_kw=3.3,
        )

        with open(output_path, encoding="utf-8") as f:
            content = f.read()

        parsed_data = json.loads(content)
        assert parsed_data["schema"] == "simonaMarkovLoad:psdm:1.0"


def test_transitions_to_json_format(tiny_models):
    from src.export import transitions_to_json

    p, _ = tiny_models

    transitions_json = transitions_to_json(p)

    assert isinstance(transitions_json, list)
    assert len(transitions_json) == p.shape[0]

    for bucket_transitions in transitions_json:
        assert isinstance(bucket_transitions, list)
        assert len(bucket_transitions) == p.shape[1]

        for state_transitions in bucket_transitions:
            assert isinstance(state_transitions, list)
            assert len(state_transitions) == p.shape[2]

            for prob in state_transitions:
                assert isinstance(prob, float)
                assert 0 <= prob <= 1


def test_gmms_to_json_format(tiny_models):
    from src.export import gmms_to_json

    _, gmms = tiny_models

    gmms_json = gmms_to_json(gmms)

    assert isinstance(gmms_json, list)
    assert len(gmms_json) == 2304

    for bucket_data in gmms_json:
        assert isinstance(bucket_data, dict)
        assert "states" in bucket_data

        states = bucket_data["states"]
        assert isinstance(states, list)
        assert len(states) == CONFIG["model"]["n_states"]

        for state_gmm in states:
            if state_gmm is not None:

                assert isinstance(state_gmm, dict)
                assert "weights" in state_gmm
                assert "means" in state_gmm
                assert "variances" in state_gmm

                weights = state_gmm["weights"]
                means = state_gmm["means"]
                variances = state_gmm["variances"]

                assert isinstance(weights, list)
                assert isinstance(means, list)
                assert isinstance(variances, list)

                assert len(weights) == len(means) == len(variances)

                assert abs(sum(weights) - 1.0) < 1e-6

                assert all(w >= 0 for w in weights)

                assert all(v >= 0 for v in variances)


def test_gmm_weight_normalization():
    from src.export import gmms_to_json

    mock_weights = np.array([2.0, 3.0])
    mock_means = np.array([0.2, 0.7])
    mock_variances = np.array([0.01, 0.02])

    gmms = [[None for _ in range(CONFIG["model"]["n_states"])] for _ in range(2)]
    gmms[0][0] = (mock_weights, mock_means, mock_variances)

    gmms_json = gmms_to_json(gmms)

    exported_weights = gmms_json[0]["states"][0]["weights"]
    assert abs(sum(exported_weights) - 1.0) < 1e-6

    expected_weights = [2.0 / 5.0, 3.0 / 5.0]
    assert abs(exported_weights[0] - expected_weights[0]) < 1e-6
    assert abs(exported_weights[1] - expected_weights[1]) < 1e-6
