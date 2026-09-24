from app.core.config import LLMSettings


def test_legacy_single_underscore_llm_env_names(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_BASE_URL=https://llm.example/v1\n"
        "LLM_OPENAI_PROXY_URL=http://proxy.example:8080\n"
        "LLM__DEFAULT_MODEL=gpt-4o-mini\n",
        encoding="utf-8",
    )

    settings = LLMSettings(_env_file=env_file)

    assert settings.base_url == "https://llm.example/v1"
    assert settings.openai_proxy_url == "http://proxy.example:8080"
    assert settings.default_model == "gpt-4o-mini"
