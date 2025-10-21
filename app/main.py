from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="MCP over FastAPI (SSE)", version="0.1.0")

# Allow cross-origin apps to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# In-memory SSE client registry
_clients: Dict[str, asyncio.Queue[str]] = {}
_client_lock = asyncio.Lock()

HEARTBEAT_INTERVAL_SEC = 15


def reduce_to_single_digit(n: int) -> int:
    # Handle negatives by absolute value; 0 remains 0
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
            yield f"event: message\n".encode("utf-8")
            # Each message is a JSON string already
            payload = f"data: {data}\n\n".encode("utf-8")
            yield payload
    except asyncio.CancelledError:
        pass
    finally:
        hb_task.cancel()
        await _unregister_client(client_id)


@app.get("/sse")
async def sse(client_id: Optional[str] = None) -> StreamingResponse:
    # Make client_id optional to be compatible with generic MCP SSE clients
    if not client_id:
        client_id = uuid.uuid4().hex

    generator = _sse_event_generator(client_id)
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# Minimal JSON-RPC 2.0 handler with MCP-like methods for tools
@app.post("/message")
async def message(request: Request) -> Response:
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # Support single-object and batch (array) requests for compatibility
    if isinstance(body, list):
        results = []
        for entry in body:
            res = await _handle_jsonrpc(entry)
            results.append(res.body)
        return JSONResponse(results)
    elif isinstance(body, dict):
        res = await _handle_jsonrpc(body)
        return res
    else:
        raise HTTPException(status_code=400, detail="Request body must be an object or array")


# Some MCP clients POST to /sse for messages; support it by delegating to the same handler
@app.post("/sse")
async def sse_post(request: Request) -> Response:
    return await message(request)


def _jsonrpc_result(req_id: Any, result: Any) -> JSONResponse:
    return JSONResponse({
        "jsonrpc": "2.0",
        "id": req_id,
        "result": result,
    })


def _jsonrpc_error(req_id: Any, code: int, message: str, data: Optional[Any] = None) -> JSONResponse:
    err = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return JSONResponse({
        "jsonrpc": "2.0",
        "id": req_id,
        "error": err,
    })


async def _handle_jsonrpc(payload: Dict[str, Any]) -> JSONResponse:
    # Handle a single JSON-RPC request object, returning a JSONResponse
    if not isinstance(payload, dict):
        return _jsonrpc_error(None, code=-32600, message="Invalid Request: expected object")

    # Be compatible with clients that omit 'jsonrpc' or 'id'
    req_id = payload.get("id", 0)
    jsonrpc = payload.get("jsonrpc") or "2.0"
    method = payload.get("method")
    params = payload.get("params") or {}

    if jsonrpc != "2.0":
        return _jsonrpc_error(req_id, code=-32600, message="Invalid Request: jsonrpc must be '2.0'")

    try:
        # Accept method aliases used by some MCP clients
        if method in ("tools/list", "tools.list"):
            # Debug log for diagnostics
            print("[MCP] tools/list request:", json.dumps(payload))
            result = _tools_list()
            print("[MCP] tools/list response:", json.dumps(result))
            return _jsonrpc_result(req_id, result)
        elif method in ("tools/call", "tools.call"):
            print("[MCP] tools/call request:", json.dumps(payload))
            result = await _tools_call(params)
            print("[MCP] tools/call result:", json.dumps(result))
            return _jsonrpc_result(req_id, result)
        else:
            return _jsonrpc_error(req_id, code=-32601, message="Method not found")
    except HTTPException:
        raise
    except Exception as e:
        return _jsonrpc_error(req_id, code=-32000, message=f"Server error: {e}")


def _tools_list() -> Dict[str, Any]:
    # MCP-like schema for tools listing, with explicit JSON Schema metadata
    return {
        "tools": [
            {
                "name": "reduce_digits",
                "description": "Reduce an integer by summing its digits repeatedly until a single digit remains.",
                "input_schema": {
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "type": "object",
                    "properties": {
                        "number": {
                            "type": "integer",
                            "description": "The number to reduce to a single digit"
                        }
                    },
                    "required": ["number"],
                    "additionalProperties": False
                }
            }
        ]
    }


async def _tools_call(params: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(params, dict):
        raise HTTPException(status_code=400, detail="params must be an object")

    name = params.get("name")
    args = params.get("arguments") or {}
    client_id = params.get("client_id")  # optional for SSE notifications

    if name != "reduce_digits":
        raise HTTPException(status_code=400, detail="Unknown tool name")

    if not isinstance(args, dict) or "number" not in args:
        raise HTTPException(status_code=400, detail="'arguments.number' is required")

    number = args["number"]
    if not isinstance(number, int):
        # Allow numeric strings that can be cast
        try:
            number = int(number)
        except Exception:
            raise HTTPException(status_code=400, detail="'number' must be an integer")

    value = reduce_to_single_digit(number)

    # MCP-like tool result structure
    result = {
        "content": [
            {"type": "text", "text": str(value)}
        ]
    }

    # If client_id provided, push an SSE event
    if client_id:
        payload = {
            "type": "tool_result",
            "tool": name,
            "arguments": {"number": number},
            "result": value,
            "client_id": client_id,
        }
        await _emit_sse(client_id, payload)

    return result


async def _emit_sse(client_id: str, payload: Dict[str, Any]) -> None:
    async with _client_lock:
        q = _clients.get(client_id)
    if q is not None:
        await q.put(json.dumps(payload))


# Health check
@app.get("/health")
async def health():
    return {"status": "ok"}


# For local run: uvicorn app.main:app --reload
