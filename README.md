# Portfolio LLM Gateway

A shared LLM inference gateway that lets multiple AI applications consume OpenAI, Gemini, Anthropic, and OpenRouter models through a unified OpenAI-compatible interface, with temporary BYOK sessions and centralized controls.

## Design

- OpenAI-compatible `POST /v1/chat/completions` endpoint.
- Multi-provider routing through LiteLLM.
- User/provider credentials are encrypted before temporary storage.
- Upstash Redis stores session state with TTL.
- JWT contains session claims only; it never contains the provider API key.
- Per-session/IP rate limiting.
- Server-side model allowlist.
- Optional controlled fallback using the portfolio owner's own provider key.

LiteLLM provides the common input/output format across 100+ providers and can be used as a centralized gateway layer. Upstash Redis is suitable for stateless/HTTP deployments and session storage. See the linked official docs cited in the implementation notes.

## Production environment

Required:

- `ENVIRONMENT=production`
- `GATEWAY_JWT_SECRET`
- `GATEWAY_ENCRYPTION_KEY`
- `UPSTASH_REDIS_REST_URL`
- `UPSTASH_REDIS_REST_TOKEN`
- `MODEL_REGISTRY_JSON`

Recommended defaults:

- `SESSION_TTL_SECONDS=1800`
- `MAX_REQUESTS_PER_MINUTE=12`
- `MAX_OUTPUT_TOKENS=4096`

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Generate secrets:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

## API flow

1. Frontend calls `POST /v1/demo/session` with provider, approved model id, and the user's provider API key.
2. Gateway encrypts the credential, stores the session in Redis with a TTL, and returns a short-lived JWT.
3. Project backend calls `POST /v1/chat/completions` with `Authorization: Bearer <session JWT>`.
4. Gateway resolves the session, decrypts the provider credential in memory, and calls LiteLLM.
5. Session expires automatically; the credential is deleted from Redis.

## Important integration note

The five projects still need a tiny request-scoped integration step for true BYOK sessions: the user's JWT must reach the project's LLM client for each request. Merely changing a static `LLM_API_KEY` environment variable is not enough for multiple simultaneous interviewers. The gateway itself is ready for that integration, but project-by-project propagation should be done deliberately and tested before exposing the interactive demos.

## Security rules

Never log provider API keys. Never put them in JWTs, URLs, Wix CMS, browser localStorage, or ordinary Postgres columns. Use HTTPS in production. Keep the gateway model list server-side and allowlist models explicitly.
