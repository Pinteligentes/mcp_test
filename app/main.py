# main.py
from __future__ import annotations

import asyncio
import json
import os
import uuid
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

# -----------------------------
# App & CORS (wide open)
# -----------------------------
app = FastAPI(title="AB-Compat MCP (FastAPI)", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # AB requiere * durante pruebas
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,
)

# -----------------------------
# In-memory state (debug/SSE)
# -----------------------------
_clients: Dict[str, asyncio.Queue[str]] = {}
_client_lock = asyncio.Lock()

_last_debug: Dict[str, Any] = {
    "last_request": None,
    "last_response": None,
    "user_agent": None,
}

HEARTBEAT_INTERVAL_SEC = 15

# -----------------------------
# Utilities
# -----------------------------
def json_result(req_id: Any, result: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}

def json_error(req_id: Any, code: int, message: str, data: Optional[Any] = None) -> Dict[str, Any]:
    err = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": req_id, "error": err}

def reduce_to_single_digit(n: int) -> int:
    n = abs(int(n))
    while n >= 10:
        s = 0
        while n > 0:
            s += n % 10
            n //= 10
        n = s
    return n

async def _register_client(client_id: str) -> asyncio.Queue[str]:
    async with _client_lock:
        q = _clients.get(client_id)
        if q is None:
            q = asyncio.Queue()
            _clients[client_id] = q
        return q

async def _unregister_client(client_id: str) -> None:
    async with _client_lock:
        _clients.pop(client_id, None)

async def _emit_sse(client_id: str, payload: Dict[str, Any]) -> None:
    async with _client_lock:
        q = _clients.get(client_id)
    if q is not None:
        await q.put(json.dumps(payload))

# -----------------------------
# Health & Debug
# -----------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/debug/last")
async def debug_last(request: Request):
    _last_debug["user_agent"] = request.headers.get("user-agent")
    return _last_debug

# -----------------------------
# CORS preflight & HEAD helpers
# -----------------------------
@app.options("/{full_path:path}")
async def any_options(full_path: str):
    return Response(
        status_code=204,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Methods": "GET,POST,OPTIONS,HEAD",
            "Access-Control-Max-Age": "86400",
        },
    )

@app.head("/sse")
async def sse_head():
    # Reporta que /sse existe (AB a veces hace HEAD)
    return Response(status_code=200, headers={"Access-Control-Allow-Origin": "*"})

# -----------------------------
# SSE endpoint (GET /sse)
# -----------------------------
async def _sse_event_generator(client_id: str) -> AsyncGenerator[bytes, None]:
    queue = await _register_client(client_id)

    async def heartbeat():
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL_SEC)
            try:
                await queue.put(json.dumps({"type": "heartbeat", "client_id": client_id}))
            except RuntimeError:
                break

    hb_task = asyncio.create_task(heartbeat())
    try:
        while True:
            data = await queue.get()
            yield b"event: message\n"
            yield f"data: {data}\n\n".encode("utf-8")
    except asyncio.CancelledError:
        pass
    finally:
        hb_task.cancel()
        await _unregister_client(client_id)

@app.get("/sse")
async def sse_get(client_id: Optional[str] = None) -> StreamingResponse:
    if not client_id:
        client_id = uuid.uuid4().hex
    generator = _sse_event_generator(client_id)
    return StreamingResponse(
        generator,
        media_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        },
    )

# -----------------------------
# JSON-RPC (POST /sse)  + alias (POST /)
# -----------------------------
@app.post("/sse")
async def sse_post(request: Request) -> Response:
    resp = await handle_jsonrpc_envelope(request)
    # Asegurar headers para AB
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Content-Type"] = "application/json; charset=utf-8"
    return resp

@app.post("/")
async def root_post(request: Request) -> Response:
    resp = await handle_jsonrpc_envelope(request)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Content-Type"] = "application/json; charset=utf-8"
    return resp

async def handle_jsonrpc_envelope(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            json_error(None, -32700, "Parse error"),
            status_code=400,
        )

    if isinstance(body, list):
        results = [await handle_jsonrpc_one(entry) for entry in body]
        return JSONResponse(results)
    elif isinstance(body, dict):
        result = await handle_jsonrpc_one(body)
        return JSONResponse(result)
    else:
        return JSONResponse(json_error(None, -32600, "Invalid Request: must be object or array"), status_code=400)

async def handle_jsonrpc_one(payload: Dict[str, Any]) -> Dict[str, Any]:
    req_id = payload.get("id")
    jsonrpc_ver = payload.get("jsonrpc") or "2.0"
    method = payload.get("method")
    params = payload.get("params") or {}

    if jsonrpc_ver != "2.0":
        return json_error(req_id, -32600, "Invalid Request: jsonrpc must be '2.0'")

    try:
        # ---- MCP handshake ----
        if method == "initialize":
            _last_debug["last_request"] = payload
            result = {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {"listChanged": True},
                },
                "serverInfo": {"name": "ab-compat-railway-mcp", "version": app.version or "0.3.0"},
            }
            _last_debug["last_response"] = result
            return json_result(req_id, result)

        if method == "notifications/initialized":
            _last_debug["last_request"] = payload
            result = {"ok": True}
            _last_debug["last_response"] = result
            return json_result(req_id, result)

        # ---- Tools list (acepta dos estilos) ----
        if method in ("tools/list", "tools.list"):
            _last_debug["last_request"] = payload
            result = tools_list()
            _last_debug["last_response"] = result
            return json_result(req_id, result)

        # ---- Tools call (acepta dos estilos) ----
        if method in ("tools/call", "tools.call"):
            _last_debug["last_request"] = payload
            result = await tools_call(params)
            _last_debug["last_response"] = result
            return json_result(req_id, result)

        return json_error(req_id, -32601, "Method not found")
    except HTTPException as e:
        return json_error(req_id, -32000, f"HTTP error: {e.detail}")
    except Exception as e:
        return json_error(req_id, -32000, f"Server error: {e}")

def tools_list() -> Dict[str, Any]:
    return {
        "tools": [
            {
                "name": "digits",
                "description": "Reduce un entero a un solo dígito sumando sus dígitos repetidamente.",
                "annotations": None,
                "input_schema": {
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "type": "object",
                    "properties": {
                        "number": {"type": "integer", "description": "Número a reducir"}
                    },
                    "required": ["number"],
                    "additionalProperties": False
                },
            },
            {
                "name": "echo",
                "description": "Devuelve el texto tal cual.",
                "annotations": None,
                "input_schema": {
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Texto a devolver"}
                    },
                    "required": ["text"],
                    "additionalProperties": False
                },
            },
        ]
    }

async def tools_call(params: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(params, dict):
        raise HTTPException(status_code=400, detail="params must be an object")

    name = params.get("name")
    args = params.get("arguments") or {}
    client_id = params.get("client_id")  # opcional: notificar por SSE

    if name == "digits":
        if "number" not in args:
            raise HTTPException(status_code=400, detail="'arguments.number' is required")
        try:
            n = int(args["number"])
        except Exception:
            raise HTTPException(status_code=400, detail="'number' must be integer")
        value = reduce_to_single_digit(n)
        result = {"content": [{"type": "text", "text": str(value)}]}

    elif name == "echo":
        if "text" not in args:
            raise HTTPException(status_code=400, detail="'arguments.text' is required")
        result = {"content": [{"type": "text", "text": str(args['text'])}]}

    else:
        raise HTTPException(status_code=400, detail=f"Unknown tool: {name}")

    # Notificación por SSE (opcional)
    if client_id:
        await _emit_sse(client_id, {"type": "tool_result", "tool": name, "result": result, "client_id": client_id})

    return result

# -----------------------------
# Local runner (Railway usa PORT)
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
