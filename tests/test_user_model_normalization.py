from app.config import ModelConfig, Settings, model_config_for_user


def test_google_model_id_is_normalized():
    config = model_config_for_user("google", "gemini/gemini-3.5-flash-lite")
    assert config.model_id == "gemini-3.5-flash-lite"


def test_openrouter_model_slug_is_preserved():
    config = model_config_for_user("openrouter", "openai/gpt-oss-20b:free")
    assert config.model_id == "openai/gpt-oss-20b:free"


def test_openai_legacy_prefix_is_removed():
    config = model_config_for_user("openai", "openai/gpt-4o-mini")
    assert config.model_id == "gpt-4o-mini"


def test_legacy_model_registry_key_is_still_accepted(monkeypatch):
    monkeypatch.setenv(
        "MODEL_REGISTRY_JSON",
        '[{"id":"legacy-google","provider":"google",'
        '"litellm_model":"gemini/gemini-2.5-flash"}]',
    )
    settings = Settings()
    assert settings.models["legacy-google"].model_id == "gemini-2.5-flash"


def test_new_model_registry_key_is_accepted(monkeypatch):
    monkeypatch.setenv(
        "MODEL_REGISTRY_JSON",
        '[{"id":"new-google","provider":"google",'
        '"model_id":"gemini-2.5-flash"}]',
    )
    settings = Settings()
    assert settings.models["new-google"].model_id == "gemini-2.5-flash"
