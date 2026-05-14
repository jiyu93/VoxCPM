import base64

from fastapi.testclient import TestClient

from voxcpm.api_server import (
    ServerSettings,
    SynthesisRequest,
    VoxCPMService,
    create_app,
)


def test_api_key_required_for_model_metadata(tmp_path):
    app = create_app(ServerSettings(output_dir=tmp_path, api_key="secret", lazy_load=True))

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/v1/model").status_code == 401

        response = client.get("/v1/model", headers={"Authorization": "Bearer secret"})

    assert response.status_code == 200
    assert response.json()["capabilities"]["voice_design"] is True


def test_reference_base64_upload_is_deduplicated(tmp_path):
    app = create_app(ServerSettings(output_dir=tmp_path, lazy_load=True))
    payload = {
        "audio_base64": base64.b64encode(b"fake wav data").decode("ascii"),
        "filename": "reference.wav",
        "content_type": "audio/wav",
    }

    with TestClient(app) as client:
        first = client.post("/v1/references/json", json=payload)
        second = client.post("/v1/references/json", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["reference_id"] == second.json()["reference_id"]


def test_prompt_text_requires_prompt_audio(tmp_path):
    app = create_app(ServerSettings(output_dir=tmp_path, lazy_load=True))

    with TestClient(app) as client:
        response = client.post(
            "/v1/synthesis/jobs",
            json={
                "text": "hello",
                "voice": {
                    "prompt_text": "reference transcript",
                },
            },
        )

    assert response.status_code == 400
    assert "prompt audio is required" in response.json()["detail"]


def test_voxcpm_native_capabilities_and_lora_status_are_exposed(tmp_path):
    app = create_app(ServerSettings(output_dir=tmp_path, lazy_load=True, lora_weights_path="/tmp/demo-lora"))

    with TestClient(app) as client:
        model = client.get("/v1/model")
        lora = client.get("/v1/model/lora")

    assert model.status_code == 200
    capabilities = model.json()["capabilities"]
    assert capabilities["voice_design"] is True
    assert capabilities["reference_clone"] is True
    assert capabilities["prompt_continuation"] is True
    assert capabilities["hybrid_clone"] is True
    assert capabilities["streaming"] is True
    assert capabilities["lora"] is True
    assert lora.status_code == 200
    assert lora.json()["configured"] is True


def test_synthesis_mode_inference_is_voxcpm_native(tmp_path):
    service = VoxCPMService(ServerSettings(output_dir=tmp_path, lazy_load=True))

    design = service.resolve_synthesis(
        SynthesisRequest.model_validate(
            {
                "text": "hello",
                "voice": {"instruction": "warm voice"},
            }
        )
    )
    plain = service.resolve_synthesis(SynthesisRequest.model_validate({"text": "hello"}))

    assert design.mode == "design"
    assert plain.mode == "plain"
