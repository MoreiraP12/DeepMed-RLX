#!/usr/bin/env python3
"""
Enhanced Workflow Visualization for DeepMed-RLX

This script enhances the existing workflow with real-time visualization capabilities,
showing graph execution progress and state changes as they happen.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import time

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage

# Import your existing components
from src.graph import build_graph
from src.graph.types import State

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class VisualizationState(State):
    """Enhanced state with visualization tracking"""
    visualization_log: List[Dict[str, Any]] = []
    node_execution_times: Dict[str, float] = {}
    current_active_node: Optional[str] = None
    execution_start_time: Optional[datetime] = None


class GraphVisualizer:
    """Real-time graph visualization manager"""
    
    def __init__(self, graph):
        self.graph = graph
        self.compiled_graph = graph if hasattr(graph, 'get_graph') else None
        self.execution_log = []
        self.node_timings = {}
        
    def compile_with_visualization(self, checkpointer=None):
        """Compile graph with visualization enhancements"""
        if checkpointer is None:
            checkpointer = MemorySaver()
        
        self.compiled_graph = self.graph.compile(checkpointer=checkpointer)
        return self.compiled_graph
    
    def display_graph_structure(self, save_files: bool = True):
        """Display and optionally save graph structure in multiple formats"""
        if not self.compiled_graph:
            raise ValueError("Graph must be compiled first")
        
        print("\n" + "="*80)
        print("📊 GRAPH STRUCTURE VISUALIZATION")
        print("="*80)
        
        # Display Mermaid diagram
        print("\n🔸 Mermaid Diagram:")
        print("-" * 50)
        mermaid_content = self.compiled_graph.get_graph().draw_mermaid()
        print(mermaid_content)
        
        if save_files:
            # Save Mermaid diagram
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            mermaid_file = f"graph_structure_{timestamp}.mmd"
            with open(mermaid_file, "w") as f:
                f.write(mermaid_content)
            print(f"💾 Saved to: {mermaid_file}")
            
            # Save JSON representation
            json_content = json.dumps(
                self.compiled_graph.get_graph().to_json(), 
                indent=2
            )
            json_file = f"graph_structure_{timestamp}.json"
            with open(json_file, "w") as f:
                f.write(json_content)
            print(f"💾 Saved to: {json_file}")
        
        # Display graph statistics
        graph_info = self.compiled_graph.get_graph()
        print(f"\n📈 Graph Statistics:")
        print(f"   Nodes: {len(graph_info.nodes)}")
        print(f"   Edges: {len(graph_info.edges)}")
        
        print(f"\n🔗 Node List:")
        for node_id in graph_info.nodes:
            print(f"   - {node_id}")
        
        print(f"\n🔀 Edge List:")
        for edge in graph_info.edges:
            print(f"   - {edge.source} → {edge.target}")
    
    async def stream_execution_with_visualization(
        self, 
        initial_state: Dict[str, Any], 
        config: Dict[str, Any],
        stream_mode: str = "values"
    ):
        """Stream graph execution with real-time visualization"""
        
        if not self.compiled_graph:
            raise ValueError("Graph must be compiled first")
        
        print("\n" + "="*80)
        print("🚀 REAL-TIME EXECUTION VISUALIZATION")
        print("="*80)
        
        execution_start = datetime.now()
        step_counter = 0
        node_execution_start = {}
        
        # Enhanced initial state with visualization tracking
        enhanced_state = {
            **initial_state,
            "visualization_log": [],
            "node_execution_times": {},
            "execution_start_time": execution_start,
        }
        
        print(f"\n⏰ Execution started at: {execution_start.isoformat()}")
        print(f"🎯 Stream mode: {stream_mode}")
        print("\n" + "-"*80)
        
        try:
            async for state_update in self.compiled_graph.astream(
                enhanced_state,
                config=config,
                stream_mode=stream_mode
            ):
                step_counter += 1
                current_time = datetime.now()
                
                print(f"\n📍 STEP {step_counter} | {current_time.strftime('%H:%M:%S.%f')[:-3]}")
                print("-" * 40)
                
                # Log the state update
                self.execution_log.append({
                    "step": step_counter,
                    "timestamp": current_time.isoformat(),
                    "state_keys": list(state_update.keys()) if isinstance(state_update, dict) else str(type(state_update)),
                })
                
                # Display current state information
                if isinstance(state_update, dict):
                    self._display_state_info(state_update, step_counter)
                else:
                    print(f"   Update: {state_update}")
                
                # Add visual separator and delay for better readability
                print("   " + "·" * 30)
                await asyncio.sleep(0.3)  # Adjustable delay for visualization
                
        except Exception as e:
            logger.error(f"Error during execution: {e}")
            print(f"❌ Execution error: {e}")
        
        execution_end = datetime.now()
        total_duration = execution_end - execution_start
        
        print(f"\n✅ EXECUTION COMPLETED!")
        print(f"⏱️  Total duration: {total_duration.total_seconds():.2f} seconds")
        print(f"📊 Total steps: {step_counter}")
        print("="*80)
        
        return self.execution_log
    
    def _display_state_info(self, state: Dict[str, Any], step_number: int):
        """Display formatted state information"""
        
        # Track important state changes
        important_keys = [
            "messages", "current_plan", "plan_iterations", 
            "final_report", "observations", "node_status",
            "visualization_log", "locale"
        ]
        
        for key in important_keys:
            if key in state:
                value = state[key]
                
                if key == "messages" and value:
                    print(f"   💬 Messages: {len(value)} total")
                    if value:
                        latest_msg = value[-1]
                        if hasattr(latest_msg, 'content'):
                            content_preview = latest_msg.content[:100]
                            print(f"      Latest: {content_preview}...")
                        
                elif key == "current_plan" and value:
                    if hasattr(value, 'steps'):
                        completed_steps = sum(1 for step in value.steps if step.execution_res)
                        total_steps = len(value.steps)
                        print(f"   📋 Plan Progress: {completed_steps}/{total_steps} steps")
                    else:
                        print(f"   📋 Current Plan: {str(value)[:50]}...")
                        
                elif key == "observations" and value:
                    print(f"   🔍 Observations: {len(value)} items")
                    
                elif key == "final_report" and value:
                    print(f"   📄 Final Report: {len(value)} characters")
                    
                elif key == "plan_iterations":
                    print(f"   🔄 Plan Iterations: {value}")
                    
                elif key == "locale":
                    print(f"   🌐 Locale: {value}")
                    
                elif key == "node_status" and value:
                    print(f"   🎯 Node Status: {value}")
    
    async def debug_execution(
        self, 
        initial_state: Dict[str, Any], 
        config: Dict[str, Any]
    ):
        """Run execution with detailed debugging information"""
        
        print("\n" + "="*80)
        print("🔍 DEBUG MODE EXECUTION")
        print("="*80)
        
        if not self.compiled_graph:
            raise ValueError("Graph must be compiled first")
        
        debug_events = []
        
        try:
            async for event in self.compiled_graph.astream(
                initial_state,
                config=config,
                stream_mode="debug"
            ):
                debug_events.append(event)
                
                print(f"\n🔸 Debug Event: {event.get('type', 'unknown')}")
                
                if event.get('type') == 'task':
                    payload = event.get('payload', {})
                    print(f"   Task: {payload.get('name', 'unnamed')}")
                    print(f"   ID: {payload.get('id', 'no-id')}")
                    
                elif event.get('type') == 'task_result':
                    payload = event.get('payload', {})
                    print(f"   Result available")
                    
                elif event.get('type') == 'error':
                    payload = event.get('payload', {})
                    print(f"   ❌ Error: {payload}")
                
                await asyncio.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Debug execution error: {e}")
            print(f"❌ Debug error: {e}")
        
        print(f"\n📊 Debug Summary: {len(debug_events)} events captured")
        return debug_events
    
    def save_execution_report(self, filename: Optional[str] = None):
        """Save detailed execution report"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"execution_report_{timestamp}.json"
        
        report = {
            "execution_log": self.execution_log,
            "node_timings": self.node_timings,
            "graph_info": {
                "nodes": list(self.compiled_graph.get_graph().nodes.keys()),
                "edges": [
                    {"source": edge.source, "target": edge.target} 
                    for edge in self.compiled_graph.get_graph().edges
                ]
            },
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filename, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"💾 Execution report saved to: {filename}")
        return filename


async def run_enhanced_workflow_with_visualization(
    user_input: str,
    debug: bool = False,
    max_plan_iterations: int = 1,
    max_step_num: int = 3,
    enable_background_investigation: bool = True,
    save_visualization_files: bool = True
):
    """
    Run the workflow with enhanced real-time visualization
    
    Args:
        user_input: The user's query or request
        debug: Enable debug mode for detailed execution info
        max_plan_iterations: Maximum number of plan iterations
        max_step_num: Maximum number of steps in a plan
        enable_background_investigation: Enable web search before planning
        save_visualization_files: Save visualization files to disk
    """
    
    if not user_input:
        raise ValueError("Input cannot be empty")
    
    print("🎯 Enhanced DeepMed-RLX Workflow with Real-time Visualization")
    print("="*80)
    
    # Build the graph using existing infrastructure (already compiled)
    compiled_graph = build_graph()
    
    # Create visualizer with the compiled graph
    visualizer = GraphVisualizer(compiled_graph)
    visualizer.compiled_graph = compiled_graph  # Set directly since it's already compiled
    
    # Display graph structure
    visualizer.display_graph_structure(save_files=save_visualization_files)
    
    # Prepare initial state
    initial_state = {
        "messages": [{"role": "user", "content": user_input}],
        "locale": "en-US",
        "auto_accepted_plan": True,
        "enable_background_investigation": enable_background_investigation,
    }
    
    # Configuration
    config = {
        "configurable": {
            "thread_id": f"enhanced_viz_{int(time.time())}",
            "max_plan_iterations": max_plan_iterations,
            "max_step_num": max_step_num,
        },
        "recursion_limit": 100,
    }
    
    try:
        # Main execution with visualization
        execution_log = await visualizer.stream_execution_with_visualization(
            initial_state, 
            config,
            stream_mode="values"
        )
        
        # If debug mode is enabled, run debug execution
        if debug:
            print("\n" + "🔍 Starting debug execution...")
            debug_events = await visualizer.debug_execution(initial_state, config)
        
        # Save execution report
        if save_visualization_files:
            visualizer.save_execution_report()
        
        print("\n🎉 Enhanced workflow completed successfully!")
        
        return execution_log
        
    except Exception as e:
        logger.error(f"Enhanced workflow error: {e}")
        print(f"❌ Workflow error: {e}")
        raise


# Integration with existing workflow
async def main():
    """Main function demonstrating enhanced visualization"""
    
    sample_queries = [
        "Research the latest trends in AI-powered medical diagnosis",
        "Create a comprehensive analysis of renewable energy adoption",
        "Investigate the impact of remote work on productivity",
    ]
    
    print("🚀 LangGraph Enhanced Visualization Demo")
    print("Choose a sample query or enter your own:")
    
    for i, query in enumerate(sample_queries, 1):
        print(f"  {i}. {query}")
    
    choice = input(f"\nEnter choice (1-{len(sample_queries)}) or custom query: ").strip()
    
    if choice.isdigit() and 1 <= int(choice) <= len(sample_queries):
        user_input = sample_queries[int(choice) - 1]
    else:
        user_input = choice
    
    if not user_input:
        user_input = sample_queries[0]  # Default
    
    print(f"\n🎯 Selected query: {user_input}")
    
    # Run enhanced workflow
    try:
        await run_enhanced_workflow_with_visualization(
            user_input=user_input,
            debug=True,  # Enable debug mode for demonstration
            max_plan_iterations=1,
            max_step_num=3,
            enable_background_investigation=True,
            save_visualization_files=True
        )
    except Exception as e:
        print(f"❌ Error running enhanced workflow: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 