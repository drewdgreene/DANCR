"""``dancr gateway``: serve the MCP tools over HTTP, policy-gated (roadmap B1 transport).

This is the one place DANCR opens a network socket, and it is opt-in and
fail-closed:

- **Loopback only by default.** The server binds ``127.0.0.1``; a non-loopback host
  is refused unless ``--allow-remote`` is given (and then TLS belongs in front).
- **Bearer tokens from the policy file.** A request must carry
  ``Authorization: Bearer <token>``; the token is compared in constant time and
  resolved to a principal. With no policy, or no principal token, the gateway
  refuses to start — it can never come up open.
- **Every tool call is policy-gated and audited** (allow / deny / approve), and an
  ``approve`` verdict is satisfied by a person's earlier decision in the approval
  queue or queued for one.
- **DNS-rebinding protection** is on, with the bind host/port as the only allowed
  hosts/origins.

The MCP tool contracts are unchanged: the gateway serves the same tools the stdio
server does, using the MCP SDK's streamable-HTTP transport (Starlette/uvicorn ship
with ``mcp``; no new dependency). See ``docs/GATEWAY.md`` and
``docs/adr/0006-gateway-transport.md``.
"""
from __future__ import annotations

import ipaddress
import json
import logging
from pathlib import Path
from typing import Any

from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from . import headless as hl
from .core.gateway import Policy, category_for

log = logging.getLogger("dancr.gateway")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_PATH = "/mcp"


def is_loopback(host: str) -> bool:
    """Whether a bind address is loopback (so it never leaves the machine). An empty or unspecified address is
    *not* loopback — treating it as safe could bind every interface."""
    h = str(host or "").strip()
    if h == "localhost":
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


_PROJECT_ARGS = ("path", "file_path", "file", "root")   # argument names that name what a tool acts on


def _project_from_args(args: Any) -> str | None:
    """The project a tool call names, from whichever path-like argument it uses (a pipeline `path`, a file, a
    repository `root`). ``None`` when the call names nothing."""
    if not isinstance(args, dict):
        return None
    for key in _PROJECT_ARGS:
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _jsonrpc_error(status: int, rid: Any, message: str, data: str = "") -> JSONResponse:
    error: dict[str, Any] = {"code": status, "message": message}
    if data:
        error["data"] = data
    return JSONResponse({"jsonrpc": "2.0", "id": rid, "error": error}, status_code=status)


class GatewayMiddleware(BaseHTTPMiddleware):
    """Authenticate and authorise every request to the MCP path, and audit each tool call."""

    def __init__(self, app: Any, *, root: Path, path: str, policy: Policy) -> None:
        super().__init__(app)
        self.root = Path(root)
        self.path = path
        self.policy = policy

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        if request.url.path != self.path:
            return await call_next(request)
        auth = request.headers.get("authorization", "")
        token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
        principal = (await run_in_threadpool(hl.principal_for_request, self.root, token))[0]
        if not principal:
            return _jsonrpc_error(401, None, "unauthorized", "A valid bearer token is required.")
        request.state.gateway_principal = principal

        body = await request.body()
        message: Any = {}
        if body:
            try:
                message = json.loads(body)
            except ValueError:
                message = {}
        if isinstance(message, dict) and message.get("method") == "tools/call":
            params = message.get("params") or {}
            name = str(params.get("name") or "")
            args = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
            project = _project_from_args(args)
            # A principal scoped to particular projects must name one; otherwise a tool keyed on `root` or
            # `file_path` (or nothing) would pass project=None and slip past the scope.
            who = self.policy.resolve(principal)
            scoped = who is not None and who.projects and "*" not in who.projects
            if scoped and not project:
                return _jsonrpc_error(403, message.get("id"), "denied",
                                      "This principal is limited to particular projects; the call names none.")
            # policy + audit touch the filesystem: keep them off the event loop
            decision = await run_in_threadpool(
                hl.authorize, self.root, principal, name,
                category=category_for(name), project=project, args=args)
            if decision["verdict"] == "deny":
                return _jsonrpc_error(403, message.get("id"), "denied", decision.get("reason", ""))
            if decision["verdict"] == "approve":
                return _jsonrpc_error(403, message.get("id"), "approval_required", decision.get("reason", ""))
        return await call_next(request)


def build_app(root: str | Path, *, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
              path: str = DEFAULT_PATH) -> Any:
    """The gateway's ASGI app: the MCP streamable-HTTP app wrapped in auth + policy middleware. Raises when the
    repository has no policy, or no principal with a token — the gateway is fail-closed."""
    from . import mcp_server
    from mcp.server.transport_security import TransportSecuritySettings
    root = Path(root).expanduser().resolve()
    policy = hl.load_policy(root)
    if policy is None or not any(p.tokens for p in policy.principals.values()):
        raise ValueError(
            "The gateway needs a policy with at least one principal token, or it would be open to anyone. "
            f"Write {hl.policy_path(root)} with principals that have a 'tokens' list, then start it again.")
    mcp_server.configure_root(root)
    allowed = sorted({host, "127.0.0.1", "localhost", f"{host}:{port}", f"127.0.0.1:{port}", f"localhost:{port}"})
    app = mcp_server.mcp.streamable_http_app(
        streamable_http_path=path, host=host,
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=True,
                                                     allowed_hosts=allowed, allowed_origins=allowed))
    app.add_middleware(GatewayMiddleware, root=root, path=path, policy=policy)
    return app


def main(root: str | Path, *, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, path: str = DEFAULT_PATH,
         allow_remote: bool = False) -> None:
    """Run the gateway. Refuses a non-loopback bind unless ``allow_remote`` (put TLS in front for remote use)."""
    import uvicorn
    if not allow_remote and not is_loopback(host):
        raise ValueError(f"{host} is not a loopback address. Bind {DEFAULT_HOST}, or pass --allow-remote to "
                         "serve on the network (and terminate TLS in front of it).")
    app = build_app(root, host=host, port=port, path=path)
    log.warning("DANCR gateway on http://%s:%s%s (policy-gated; Ctrl+C to stop)", host, port, path)
    print(f"DANCR gateway on http://{host}:{port}{path} (policy-gated; Ctrl+C to stop)")
    uvicorn.run(app, host=host, port=port, log_level="warning")
