from src.api.main import create_app


def test_all_detection_routes_are_registered():
    app = create_app()
    paths = app.openapi()["paths"]

    assert "/api/v1/email/analyze" in paths
    assert "/api/v1/url/analyze" in paths
    assert "/api/v1/network/analyze" in paths


def test_network_request_example_contains_all_features():
    app = create_app()
    network_request_schema = app.openapi()["components"]["schemas"][
        "NetworkRequest"
    ]
    example = network_request_schema["properties"]["features"]["examples"][0]

    assert isinstance(example, list)
    assert len(example) == 78
