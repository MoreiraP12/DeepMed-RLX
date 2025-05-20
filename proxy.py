from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import httpx
import json
import logging

app = FastAPI()
OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
logging.basicConfig(level=logging.INFO)

@app.post("/v1/chat/completions")
async def proxy_chat(request: Request):
    body = await request.json()
    model = body.get("model", "")
    is_stream = body.get("stream", False)

    logging.info(f"🔹 Proxying to Ollama (model: {model})\n{json.dumps(body, indent=2)}")

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            request_obj = client.build_request("POST", OLLAMA_URL, json=body)
            if is_stream:
                # Manual send with stream enabled
                response = await client.send(request_obj, stream=True)

                async def stream_generator():
                    async for line in response.aiter_lines():
                        if line.strip():
                            yield f"data: {line}\n\n"

                return StreamingResponse(stream_generator(), media_type="text/event-stream")
            else:
                response = await client.send(request_obj)
                response.raise_for_status()
                data = response.json()
                logging.info(f"🔸 Response:\n{json.dumps(data, indent=2)}")
                return JSONResponse(content=data)

    except httpx.ReadTimeout:
        return JSONResponse(content={"error": f"Ollama timed out on model '{model}'"}, status_code=504)
    except Exception as e:
        logging.exception("🔥 Proxy failed")
        return JSONResponse(content={"error": str(e)}, status_code=500)
