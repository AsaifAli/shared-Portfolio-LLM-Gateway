from app.config import model_config_for_user

def test_google():
    c=model_config_for_user('google','gemini-3.5-flash-lite')
    assert c.litellm_model=='gemini/gemini-3.5-flash-lite'

def test_openrouter():
    c=model_config_for_user('openrouter','openai/gpt-oss-20b:free')
    assert c.litellm_model=='openrouter/openai/gpt-oss-20b:free'
