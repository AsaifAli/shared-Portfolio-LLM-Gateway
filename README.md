# Portfolio LLM Gateway

Shared provider-agnostic inference gateway for the portfolio's interactive AI projects.

The gateway provides a single OpenAI-compatible endpoint while keeping each user's provider credentials outside the client application and outside the project repositories.

## Architecture

```text
Portfolio
   │
   │  create BYOK session
   ▼
Gateway
   │
   ├── encrypt provider credential
   ├── store short-lived session in Redis
   └── issue temporary JWT
   │
   ▼
Interactive Project
   │
   │ Authorization: Bearer <session JWT>
   ▼
POST /v1/chat/completions
   │
   ▼
Gateway resolves session
   │
   ▼
LiteLLM / provider adapter
   │
   ▼
User-selected provider/model
```

The JWT contains session claims only. The provider API key is never placed in the JWT, URL, browser storage, or project source code.

## Projects using the gateway

The portfolio currently routes BYOK sessions to:

- QuoteSense — quotation analysis
- WebQA — web application QA intelligence
- FlowPilot — AI automation workflows
- EvidenceFlow — verified RAG / research
- LegacyLens — software modernization

## Core capabilities

- OpenAI-compatible `POST /v1/chat/completions`
- Multi-provider routing through LiteLLM
- Encrypted provider credentials
- Upstash Redis session storage with TTL
- Short-lived JWT authentication
- Per-session / IP rate limiting
- Server-side model allowlist
- Optional controlled fallback using the portfolio owner's own provider key

## Production configuration

Required:

- `ENVIRONMENT=production`
- `GATEWAY_JWT_SECRET`
- `GATEWAY_ENCRYPTION_KEY`
- `UPSTASH_REDIS_REST_URL`
- `UPSTASH_REDIS_REST_TOKEN`
- `MODEL_REGISTRY_JSON`

Recommended defaults:

- `SESSION_TTL_SECONDS=1800`
- `MAX_REQUESTS_PER_MINUTE=60`
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

1. The portfolio calls the session endpoint with the selected provider, allowed model, and user's provider API key.
2. The gateway encrypts the credential and stores the temporary session in Redis.
3. The gateway returns a short-lived JWT.
4. The selected project forwards the JWT to this gateway for each inference request.
5. The gateway resolves the session, decrypts the provider credential in memory, and calls the configured provider through LiteLLM.
6. Expiration or revocation removes access to the provider credential.

## Security rules

Never:

- log provider API keys
- place provider API keys in JWTs
- place provider keys in URLs
- store provider keys in browser localStorage
- expose provider credentials to project frontends
- commit `.env` or production secrets

Use HTTPS in production and keep the model allowlist server-side.

## Health and observability

The gateway exposes application health/readiness endpoints through FastAPI. Deployment and runtime logs should never contain raw provider credentials.

## Testing

Run the repository's configured lint, test, and compile checks before deployment. Integration testing should cover session creation, expiration, authorization failure, provider routing, and request-scoped project handoff.

## Portfolio boundary

This service is the shared infrastructure layer for the portfolio demos. It is designed to demonstrate secure request-scoped BYOK routing and provider abstraction for a portfolio environment; multi-tenant enterprise identity, centralized billing, and enterprise policy management remain outside the current scope.

## License

MIT License.
