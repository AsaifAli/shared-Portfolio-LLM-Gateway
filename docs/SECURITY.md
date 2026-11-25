# Security Review — Portfolio LLM Gateway

This gateway handles user-provided provider API credentials and bearer session JWTs. Treat it as a high-value credential boundary. This review is based on visible source/configuration and is not a production penetration test.

## Observed controls
- JWTs are signed with HS256 and require standard identity/expiry claims in `app/security.py`; runtime configuration requires at least 32 characters.
- Provider keys are encrypted before session persistence using Fernet; production requires an explicit encryption key and Redis.
- Optional authenticated-owner binding is implemented: when `SUPABASE_URL` and `SUPABASE_PUBLISHABLE_KEY` are configured, the Gateway verifies the supplied Supabase access token with `/auth/v1/user` (redirects/proxy environment disabled) and derives a stable opaque owner hash. Unauthenticated BYOK sessions remain supported but receive a separate per-session scope.
- The owner scope is returned from Gateway status and carried through one-time project handoff; apps must compare it server-side, never accept a scope supplied in workflow payloads.
- Redis-backed session store fails closed when configured Redis is unavailable.
- Session and per-IP throttling are implemented in `app/security.py`.
- Handoff uses a random code, stores a SHA-256 digest, bounds TTL, validates project audience, and uses Redis GETDEL for single-use redemption (`app/handoff.py`, `app/security.py`).
- CORS allows explicit configured origins, disallows credentials, and production rejects wildcard/localhost/regex origins (`app/config.py`, `app/main.py`).
- Request body limits are enforced for inference POST routes.
- Provider exception logs record exception class/status metadata rather than raw provider messages (`app/llm.py`).
- SQL, file uploads, password auth, RLS and webhook receiver are not present in the gateway service.
- API keys are supplied via server-side environment variables; provider keys are not included in JWT claims or gateway-generated browser URLs.

## BYOK compatibility and identity activation
The Gateway's existing encrypted provider-key store, short-lived bearer JWT, `sessionStorage` token handoff and one-time exchange semantics are preserved. No HttpOnly-cookie migration was attempted because it requires a coordinated same-site/BFF change across all apps. The studio website's Supabase identity attachment is opt-in via the server-only `GATEWAY_SUPABASE_IDENTITY_ENABLED=true` runtime binding; configure both Gateway Supabase variables before enabling the flag. With the flag off, the current guest BYOK flow is unchanged.

## Important architectural note
The Gateway returns a short-lived bearer JWT to its trusted calling service. The studio website's BFF stores that upstream token in a path-scoped HttpOnly cookie and never returns it to browser JavaScript. Other project applications must use a server-side proxy or an equally narrow, short-lived session boundary; do not place Gateway or provider credentials in localStorage. Keep short TTLs, project-scoped audiences, no-store responses and one-time server-side handoff.

## Remaining / recommended work
1. Add strict validation for inbound message/tool JSON structures and enforce aggregate nested payload limits, not only top-level request bytes.
2. Consider key rotation with key IDs and overlapping JWT verification keys; current JWT signing uses one configured secret.
3. Add a distributed edge/WAF rate limit in addition to Redis application limits.
4. Add request ID propagation and redact all authorization/API-key values from proxy/APM logs.
5. Verify the website session token storage and CSP in the actual frontend; HttpOnly cookie migration must be cross-application.
6. Scan resolved Python dependencies and container base image for high/critical vulnerabilities.
7. Configure GitHub/hosting/provider MFA and secret rotation outside the repository.

## 20-control assessment
| # | Control | Result |
|---|---|---|
|01|SQLi|N/A — no SQL persistence in gateway|
|02|XSS|PARTIAL — API JSON only; caller apps must safely render model/document output|
|03|CSRF|N/A/PARTIAL — bearer-header API, no cookie auth; revisit if cookie sessions are introduced|
|04|File upload|N/A — no upload endpoints|
|05|SSRF|PASS for provider routing — provider destinations are hard-coded SDK endpoints; Gemini remote image URLs are rejected|
|06|BOLA/IDOR|IMPROVED/PARTIAL — signed sessions contain a server-derived owner scope that is checked against the Redis session record and survives handoff; downstream apps must enforce that scope on their own data|
|07|Rate limiting|PASS in app — Redis session/IP counters; add edge protection|
|08|Password hashing|N/A — no passwords|
|09|MFA|EXTERNAL — enable for provider, GitHub and hosting admins|
|10|Server-side permissions|PASS for inference — requires live session and validates project-bound handoff|
|11|RLS|N/A — no application database|
|12|JWT secret|PARTIAL — 256-bit length validation and expiry; no built-in multi-key rotation|
|13|Server-side API keys|PASS — provider keys encrypted and server-side|
|14|No localStorage tokens|PARTIAL — studio website uses a path-scoped HttpOnly BFF cookie; other consuming apps still require individual review|
|15|No default credentials|PASS in production config — missing secrets fail closed; verify deployment values|
|16|Strict CORS|PASS — explicit origins, no credentials, production validation|
|17|Webhook signatures|N/A — no webhook receiver|
|18|Source maps|N/A — no frontend bundle|
|19|Clean logs|PASS in provider adapter — raw upstream messages/keys are not logged; verify hosting proxy logs|
|20|Dependency patching|UNVERIFIED — run resolver-aware audit and container scan|

## AI audit prompt
Act as a Principal Application Security Engineer. Audit all 20 controls and cite exact file/line evidence. Mark PASS, PARTIAL, FAIL or N/A without assuming absence means pass. For each finding state exploit preconditions, impact, minimal safe patch, and regression test. Separate code controls from deployment controls. Do not print credentials, tokens or secret values.