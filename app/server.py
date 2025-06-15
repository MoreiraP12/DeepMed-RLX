from fastapi import WebSocket

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    doctor_agent = get_doctor_agent()
    try:
        while True:
            user_message = await websocket.receive_text()
            # Use asyncio.create_task to run the agent's stream method
            async for event in doctor_agent.astream(
                {"messages": ("user", user_message)}
            ):
                if "messages" in event:
                    for message in event["messages"]: 