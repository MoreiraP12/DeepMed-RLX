#!/usr/bin/env python3
"""
LangGraph Real-time Visualization Demo

This script demonstrates how to use LangGraph's built-in methods to visualize 
graphs in real-time as they're being built and show how the state is changing.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage

# Configure logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DemoState(MessagesState):
    """Enhanced state for demonstration with visualization tracking"""
    step_counter: int = 0
    execution_log: list[str] = []
    node_status: Dict[str, str] = {}
    
    
def demo_node_1(state: DemoState) -> Dict[str, Any]:
    """First demo node that processes initial input"""
    logger.info("🟢 Executing Node 1: Input Processing")
    
    # Simulate some processing
    import time
    time.sleep(1)
    
    execution_log = state.get("execution_log", [])
    execution_log.append(f"Node 1 executed at {datetime.now().isoformat()}")
    
    node_status = state.get("node_status", {})
    node_status["node_1"] = "completed"
    
    return {
        "messages": [AIMessage(content="Node 1: Processed initial input successfully")],
        "step_counter": state.get("step_counter", 0) + 1,
        "execution_log": execution_log,
        "node_status": node_status
    }


def demo_node_2(state: DemoState) -> Dict[str, Any]:
    """Second demo node that does analysis"""
    logger.info("🟡 Executing Node 2: Analysis Phase")
    
    import time
    time.sleep(1.5)
    
    execution_log = state.get("execution_log", [])
    execution_log.append(f"Node 2 executed at {datetime.now().isoformat()}")
    
    node_status = state.get("node_status", {})
    node_status["node_2"] = "completed"
    
    return {
        "messages": [AIMessage(content="Node 2: Completed analysis of the input")],
        "step_counter": state.get("step_counter", 0) + 1,
        "execution_log": execution_log,
        "node_status": node_status
    }


def demo_node_3(state: DemoState) -> Dict[str, Any]:
    """Third demo node that generates final output"""
    logger.info("🔵 Executing Node 3: Output Generation")
    
    import time
    time.sleep(1)
    
    execution_log = state.get("execution_log", [])
    execution_log.append(f"Node 3 executed at {datetime.now().isoformat()}")
    
    node_status = state.get("node_status", {})
    node_status["node_3"] = "completed"
    
    return {
        "messages": [AIMessage(content="Node 3: Generated final output based on analysis")],
        "step_counter": state.get("step_counter", 0) + 1,
        "execution_log": execution_log,
        "node_status": node_status
    }


def conditional_node(state: DemoState) -> str:
    """Conditional logic to demonstrate branching"""
    step_count = state.get("step_counter", 0)
    
    # Simple condition: if we have processed more than 2 steps, go to final node
    if step_count >= 2:
        return "node_3"
    else:
        return "node_2"


def build_demo_graph() -> StateGraph:
    """Build a demonstration graph with visualization capabilities"""
    
    # Create the state graph
    builder = StateGraph(DemoState)
    
    # Add nodes
    builder.add_node("node_1", demo_node_1)
    builder.add_node("node_2", demo_node_2) 
    builder.add_node("node_3", demo_node_3)
    
    # Add edges
    builder.add_edge(START, "node_1")
    builder.add_edge("node_1", "node_2")
    
    # Add conditional edge for demonstration
    builder.add_conditional_edges(
        "node_2",
        conditional_node,
        {
            "node_2": "node_2",  # Loop back if condition not met
            "node_3": "node_3"   # Move to final node
        }
    )
    
    builder.add_edge("node_3", END)
    
    return builder


async def visualize_graph_realtime(graph: StateGraph, initial_state: Dict[str, Any]):
    """
    Demonstrate real-time visualization of graph execution with state tracking
    """
    print("\n" + "="*60)
    print("🚀 STARTING REAL-TIME GRAPH VISUALIZATION")
    print("="*60)
    
    # Compile the graph with memory for state persistence
    memory = MemorySaver()
    compiled_graph = graph.compile(checkpointer=memory)
    
    # Print initial graph structure
    print("\n📊 GRAPH STRUCTURE (Mermaid):")
    print("-" * 40)
    mermaid_diagram = compiled_graph.get_graph().draw_mermaid()
    print(mermaid_diagram)
    
    # Save mermaid diagram to file for external viewing
    with open("graph_structure.mmd", "w") as f:
        f.write(mermaid_diagram)
    print("💾 Graph structure saved to 'graph_structure.mmd'")
    
    # Print graph as ASCII art (if available)
    try:
        print("\n🎨 GRAPH STRUCTURE (ASCII):")
        print("-" * 40)
        ascii_diagram = compiled_graph.get_graph().draw_ascii()
        print(ascii_diagram)
    except Exception as e:
        print(f"ASCII diagram not available: {e}")
    
    print("\n🔄 STARTING GRAPH EXECUTION...")
    print("=" * 60)
    
    config = {"configurable": {"thread_id": "demo_thread"}}
    
    # Stream the execution in real-time
    step_number = 0
    async for state_snapshot in compiled_graph.astream(
        initial_state, 
        config=config,
        stream_mode="values"  # Stream state values
    ):
        step_number += 1
        
        print(f"\n📍 STEP {step_number} - State Update:")
        print("-" * 30)
        
        # Display current state information
        if "step_counter" in state_snapshot:
            print(f"   Step Counter: {state_snapshot['step_counter']}")
        
        if "node_status" in state_snapshot and state_snapshot["node_status"]:
            print(f"   Node Status: {state_snapshot['node_status']}")
        
        if "execution_log" in state_snapshot and state_snapshot["execution_log"]:
            print(f"   Latest Log: {state_snapshot['execution_log'][-1]}")
        
        # Display messages
        if "messages" in state_snapshot and state_snapshot["messages"]:
            latest_message = state_snapshot["messages"][-1]
            if hasattr(latest_message, 'content'):
                print(f"   Message: {latest_message.content}")
        
        # Add a small delay for better visualization
        await asyncio.sleep(0.5)
    
    print("\n✅ GRAPH EXECUTION COMPLETED!")
    print("=" * 60)


async def demonstrate_state_debugging(graph: StateGraph, initial_state: Dict[str, Any]):
    """
    Demonstrate debugging capabilities with state inspection
    """
    print("\n" + "="*60)
    print("🔍 ADVANCED STATE DEBUGGING DEMONSTRATION")
    print("="*60)
    
    memory = MemorySaver()
    compiled_graph = graph.compile(checkpointer=memory)
    
    config = {"configurable": {"thread_id": "debug_thread"}}
    
    # Execute with debug mode streaming
    print("\n🐛 Debug Mode - Streaming Updates and Debug Info:")
    print("-" * 50)
    
    async for event in compiled_graph.astream(
        initial_state,
        config=config,
        stream_mode="debug"  # Debug mode for detailed info
    ):
        print(f"\n🔸 Debug Event: {event['type']}")
        if event['type'] == 'task':
            task_info = event['payload']
            print(f"   Task ID: {task_info.get('id', 'N/A')}")
            print(f"   Task Name: {task_info.get('name', 'N/A')}")
            if 'input' in task_info:
                print(f"   Input Keys: {list(task_info['input'].keys())}")
        elif event['type'] == 'task_result':
            print(f"   Result: {event['payload'].get('result', 'N/A')}")
    
    print("\n📊 Final State Inspection:")
    print("-" * 30)
    
    # Get final state
    final_state = await compiled_graph.ainvoke(initial_state, config=config)
    
    # Pretty print final state
    for key, value in final_state.items():
        if key == "messages":
            print(f"   {key}: {len(value)} messages")
            for i, msg in enumerate(value):
                if hasattr(msg, 'content'):
                    print(f"      Message {i}: {msg.content[:100]}...")
        else:
            print(f"   {key}: {value}")


async def demonstrate_graph_updates():
    """
    Demonstrate how to visualize graph structure updates
    """
    print("\n" + "="*60)
    print("🔄 DYNAMIC GRAPH STRUCTURE UPDATES")
    print("="*60)
    
    # Build initial graph
    builder = build_demo_graph()
    
    print("\n📋 Initial Graph Nodes:")
    compiled_graph = builder.compile()
    initial_nodes = compiled_graph.get_graph().nodes
    for node_id, node_data in initial_nodes.items():
        print(f"   - {node_id}: {node_data}")
    
    print(f"\n🔗 Initial Graph Edges:")
    initial_edges = compiled_graph.get_graph().edges
    for edge in initial_edges:
        print(f"   - {edge.source} → {edge.target}")
    
    print("\n💾 Saving graph variations...")
    
    # Save different representations
    representations = {
        "mermaid": compiled_graph.get_graph().draw_mermaid(),
        "mermaid_png": "Use: compiled_graph.get_graph().draw_mermaid_png() for PNG",
        "ascii": "Use: compiled_graph.get_graph().draw_ascii() for ASCII art"
    }
    
    for format_name, content in representations.items():
        filename = f"graph_{format_name}.txt"
        with open(filename, "w") as f:
            f.write(str(content))
        print(f"   ✅ Saved {filename}")


def create_interactive_visualization():
    """
    Create an interactive visualization example
    """
    print("\n" + "="*60)
    print("🎮 INTERACTIVE VISUALIZATION SETUP")
    print("="*60)
    
    # Build graph
    builder = build_demo_graph()
    compiled_graph = builder.compile()
    
    # Generate visualization files for external tools
    visualizations = {
        "mermaid": compiled_graph.get_graph().draw_mermaid(),
        "json": json.dumps(compiled_graph.get_graph().to_json(), indent=2)
    }
    
    print("\n📁 Generated Visualization Files:")
    for format_name, content in visualizations.items():
        filename = f"interactive_graph.{format_name}"
        with open(filename, "w") as f:
            f.write(content)
        print(f"   ✅ {filename}")
    
    print("\n🌐 To view interactively:")
    print("   1. Mermaid: Copy content from 'interactive_graph.mermaid' to https://mermaid.live/")
    print("   2. JSON: Use the JSON with graph visualization libraries")
    print("   3. LangSmith: Use LangSmith for advanced tracing (requires API key)")


async def main():
    """
    Main demonstration function showcasing all LangGraph visualization features
    """
    print("🎯 LangGraph Real-time Visualization Demo")
    print("=" * 60)
    
    # Initial state for demonstration
    initial_state = {
        "messages": [HumanMessage(content="Start the demonstration workflow")],
        "step_counter": 0,
        "execution_log": [],
        "node_status": {}
    }
    
    # Build the demo graph
    builder = build_demo_graph()
    
    # 1. Real-time execution visualization
    await visualize_graph_realtime(builder, initial_state)
    
    # 2. Advanced debugging
    await demonstrate_state_debugging(builder, initial_state)
    
    # 3. Graph structure updates
    await demonstrate_graph_updates()
    
    # 4. Interactive visualization setup
    create_interactive_visualization()
    
    print("\n🎉 DEMONSTRATION COMPLETED!")
    print("\nFiles generated for external visualization:")
    print("  - graph_structure.mmd (Mermaid diagram)")
    print("  - interactive_graph.mermaid (Interactive Mermaid)")
    print("  - interactive_graph.json (JSON representation)")
    
    print("\n💡 Pro Tips:")
    print("  1. Use stream_mode='values' for state updates")
    print("  2. Use stream_mode='debug' for detailed execution info")
    print("  3. Use MemorySaver for state persistence")
    print("  4. Use get_graph().draw_mermaid() for visual representation")
    print("  5. Use LangSmith for production-level tracing and visualization")


if __name__ == "__main__":
    asyncio.run(main()) 