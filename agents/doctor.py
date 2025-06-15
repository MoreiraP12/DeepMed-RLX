from typing import TypedDict, Annotated, Any, Literal
from operator import add

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add]
    sender: str
    is_final: bool 

def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return "__end__"

# ... graph definition ...
workflow.add_conditional_edges(
    "doctor",
    should_continue,
)
workflow.add_edge("tools", "doctor") 