import json

import pytest

from src.export import build_psdm_payload_from_models


def test_thresholds_right_formula(small_df, tiny_models):
    P, gmms = tiny_models
    payload = build_psdm_payload_from_models(small_df, P, gmms)

    thresholds = payload["value_model"]["discretization"]["thresholds_right"]

    expected_thresholds = [(k / 10) ** 2 for k in range(1, 10)]

    assert thresholds == expected_thresholds

    assert thresholds == [0.01, 0.04, 0.09, 0.16, 0.25, 0.36, 0.49, 0.64, 0.81]
    assert len(thresholds) == 9


def test_bucket_encoding_formula(small_df, tiny_models):
    P, gmms = tiny_models
    payload = build_psdm_payload_from_models(small_df, P, gmms)

    formula = payload["time_model"]["bucket_encoding"]["formula"]
    expected_formula = "bucket = month*192 + is_weekend*96 + quarter_hour"

    assert formula == expected_formula


def test_schema_structure_completeness(small_df, tiny_models):
    P, gmms = tiny_models
    payload = build_psdm_payload_from_models(small_df, P, gmms)

    expected_top_keys = {
        "schema",
        "generated_at",
        "generator",
        "time_model",
        "value_model",
        "parameters",
        "data",
    }
    assert set(payload.keys()) == expected_top_keys

    generator = payload["generator"]
    expected_generator_keys = {"name", "version", "config"}
    assert set(generator.keys()) == expected_generator_keys

    expected_config_keys = {"n_states", "laplace_alpha"}
    assert set(generator["config"].keys()) == expected_config_keys

    time_model = payload["time_model"]
    expected_time_keys = {
        "bucket_count",
        "bucket_encoding",
        "sampling_interval_minutes",
        "timezone",
    }
    assert set(time_model.keys()) == expected_time_keys

    expected_bucket_encoding_keys = {"formula"}
    assert set(time_model["bucket_encoding"].keys()) == expected_bucket_encoding_keys

    value_model = payload["value_model"]
    expected_value_keys = {"value_unit", "normalization", "discretization"}
    assert set(value_model.keys()) == expected_value_keys

    expected_norm_keys = {"method"}
    assert set(value_model["normalization"].keys()) == expected_norm_keys

    expected_disc_keys = {"states", "thresholds_right"}
    assert set(value_model["discretization"].keys()) == expected_disc_keys

    parameters = payload["parameters"]
    expected_param_keys = {"transitions", "gmm"}
    assert set(parameters.keys()) == expected_param_keys

    expected_trans_keys = {"empty_row_strategy"}
    assert set(parameters["transitions"].keys()) == expected_trans_keys

    data = payload["data"]
    expected_data_keys = {"transitions", "gmms"}
    assert set(data.keys()) == expected_data_keys

    transitions = data["transitions"]
    expected_trans_data_keys = {"shape", "dtype", "encoding", "values"}
    assert set(transitions.keys()) == expected_trans_data_keys

    gmms_data = data["gmms"]
    expected_gmm_keys = {"buckets"}
    assert set(gmms_data.keys()) == expected_gmm_keys


def test_schema_value_constraints(small_df, tiny_models):
    P, gmms = tiny_models
    payload = build_psdm_payload_from_models(small_df, P, gmms)

    assert payload["schema"] == "simonaMarkovLoad:psdm:1.0"
    assert payload["generator"]["name"] == "simonaMarkovLoad"
    assert payload["generator"]["version"] == "git:unknown"

    assert payload["time_model"]["bucket_count"] == 2304
    assert payload["time_model"]["sampling_interval_minutes"] == 15
    assert payload["time_model"]["timezone"] == "Europe/Berlin"

    assert payload["value_model"]["value_unit"] == "normalized"
    assert payload["value_model"]["normalization"]["method"] == "minmax_per_series"

    assert payload["parameters"]["transitions"]["empty_row_strategy"] == "self_loop"

    assert payload["data"]["transitions"]["dtype"] == "float32"
    assert payload["data"]["transitions"]["encoding"] == "nested_lists"

    transitions_shape = payload["data"]["transitions"]["shape"]
    assert len(transitions_shape) == 3
    assert transitions_shape[0] == 2304
    assert transitions_shape[1] == transitions_shape[2]

    buckets = payload["data"]["gmms"]["buckets"]
    assert len(buckets) == 2304

    for bucket in buckets:
        assert "states" in bucket
        states = bucket["states"]
        assert len(states) == payload["generator"]["config"]["n_states"]


def test_generated_at_format(small_df, tiny_models):
    P, gmms = tiny_models
    payload = build_psdm_payload_from_models(small_df, P, gmms)

    generated_at = payload["generated_at"]

    assert generated_at.endswith("Z")

    import re

    iso_pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
    assert re.match(iso_pattern, generated_at)

    from datetime import datetime

    try:
        datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    except ValueError:
        pytest.fail(f"Timestamp '{generated_at}' is not parseable")


def test_json_serialization_round_trip(small_df, tiny_models):
    P, gmms = tiny_models
    original_payload = build_psdm_payload_from_models(small_df, P, gmms)

    json_string = json.dumps(original_payload)

    reconstructed_payload = json.loads(json_string)

    assert set(original_payload.keys()) == set(reconstructed_payload.keys())

    assert reconstructed_payload["schema"] == original_payload["schema"]
    assert reconstructed_payload["time_model"]["bucket_count"] == 2304

    orig_shape = original_payload["data"]["transitions"]["shape"]
    recon_shape = reconstructed_payload["data"]["transitions"]["shape"]
    assert orig_shape == recon_shape

    orig_buckets = original_payload["data"]["gmms"]["buckets"]
    recon_buckets = reconstructed_payload["data"]["gmms"]["buckets"]
    assert len(orig_buckets) == len(recon_buckets)


@pytest.mark.skipif(
    True, reason="JSON schema validation skipped - add jsonschema dependency to enable"
)
def test_jsonschema_validation(small_df, tiny_models):

    jsonschema = pytest.importorskip("jsonschema")

    P, gmms = tiny_models
    payload = build_psdm_payload_from_models(small_df, P, gmms)

    schema = {
        "type": "object",
        "required": [
            "schema",
            "generated_at",
            "generator",
            "time_model",
            "value_model",
            "parameters",
            "data",
        ],
        "properties": {
            "schema": {"type": "string", "enum": ["simonaMarkovLoad:psdm:1.0"]},
            "generated_at": {"type": "string"},
            "time_model": {
                "type": "object",
                "required": ["bucket_count"],
                "properties": {"bucket_count": {"type": "integer", "enum": [2304]}},
            },
            "data": {
                "type": "object",
                "required": ["transitions", "gmms"],
                "properties": {
                    "transitions": {
                        "type": "object",
                        "required": ["shape", "dtype", "encoding", "values"],
                        "properties": {
                            "dtype": {"type": "string", "enum": ["float32"]},
                            "encoding": {"type": "string", "enum": ["nested_lists"]},
                        },
                    }
                },
            },
        },
    }

    jsonschema.validate(payload, schema)
