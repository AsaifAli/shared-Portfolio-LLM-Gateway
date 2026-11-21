# Portfolio LLM Gateway

Shared provider-agnostic inference gateway for the portfolio's interactive AI projects.

The gateway provides a single OpenAI-compatible inference interface while keeping each user's provider credentials outside client applications and project repositories.

## What changed in the portfolio-ready release

### BYOK session architecture

```text
Portfolio
   │
   │ create BYOK session
   ▼
Gateway
   ├── encrypt provider credential
   ├── store short-lived session in Redis
   └── issue temporary JWT
   │
   ▼
Interactive Project
   │ Authorization: Bearer <session JWT>
   ▼
POST /v1/chat/completions
   │
   ▼
Gateway resolves session
   │
   ▼
Direct provider adapters (official SDKs)
   │
   ▼
User-selected provider/model
```

Provider API keys are not placed in JWT claims, browser storage, URLs, or project source code. The website retains the short-lived bearer token in session storage; only a random, project-bound, single-use code is passed to a demo application's URL. The receiving app removes that code immediately and exchanges it server-side. Handoff codes expire after 60 seconds and are atomically consumed once.

### Demo stability

The hosted portfolio gateway is tuned for multi-step agent workflows rather than only single chat requests:

- Higher default per-session request budget (`MAX_REQUESTS_PER_MINUTE=60`)
- `Retry-After` guidance on gateway throttling responses
- Provider rate-limit errors are normalized as explicit `429` responses rather than being mislabeled as generic upstream failures
- The gateway remains compatible with request-scoped project sessions

A portfolio demo can still be limited by an upstream provider's own quota/rate limit; the gateway does not bypass provider policy.

## Projects using the gateway

- QuoteSense — procurement quotation analysis
- WebQA — web application QA intelligence
- FlowPilot / AI Automation — asynchronous agent workflows
- EvidenceFlow — verified RAG / research
- LegacyLens — software modernization

## Core capabilities

- OpenAI-compatible `POST /v1/chat/completions`
- Provider-specific direct SDK adapters for OpenAI, OpenRouter, Gemini and Anthropic
- Encrypted provider credentials
- Redis-backed session storage with TTL
- Short-lived JWT authentication
- Per-session / IP rate limiting
- Server-side model allowlist
- Optional controlled fallback
- Health / readiness endpoints

## Production configuration

Recommended demo-safe defaults include:

```env
SESSION_TTL_SECONDS=1800
MAX_REQUESTS_PER_MINUTE=60
MAX_OUTPUT_TOKENS=4096
```

Required production configuration still includes the JWT secret, encryption key, Redis credentials, and model registry.

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

1. Portfolio creates a session with the selected provider/model and user credential.
2. Gateway encrypts the credential and stores the temporary session in Redis.
3. Gateway returns a short-lived JWT.
4. The project forwards the JWT for inference requests.
5. Gateway resolves the session and calls the selected provider through its direct SDK adapter.
6. Session expiration/revocation removes access to the provider credential.

## Security rules

Never:

- log provider API keys
- place provider keys in JWTs
- place provider keys in URLs
- store provider keys in browser localStorage
- expose provider credentials to project frontends
- commit `.env` or production secrets

## Testing

Run the repository's configured lint, test, and compile checks before deployment. Integration coverage should include session creation, expiration, authorization failure, provider routing, project handoff, and rate-limit behavior.

## Portfolio positioning

**Shared inference infrastructure** — secure BYOK sessions, provider abstraction, request-scoped routing, and predictable rate-limit/error behavior for multiple AI portfolio applications.
