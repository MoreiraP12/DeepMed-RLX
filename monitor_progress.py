#!/usr/bin/env python3
"""
Monitor script to check the progress of MedBrowse dataset processing in real-time.
"""

import time
import json
import sys
from pathlib import Path
from datetime import datetime
import argparse


def find_latest_progress_files(output_dir: str = "outputs"):
    """Find the latest progress files."""
    output_path = Path(output_dir)
    
    # Find all progress files
    progress_files = list(output_path.glob("medbrowse_progress_*.txt"))
    current_files = list(output_path.glob("medbrowse_current_*.json"))
    answers_files = list(output_path.glob("medbrowse_current_answers_*.txt"))
    
    if not progress_files:
        return None, None, None
    
    # Get the latest files
    latest_progress = max(progress_files, key=lambda x: x.stat().st_mtime)
    latest_current = max(current_files, key=lambda x: x.stat().st_mtime) if current_files else None
    latest_answers = max(answers_files, key=lambda x: x.stat().st_mtime) if answers_files else None
    
    return latest_progress, latest_current, latest_answers


def show_progress(output_dir: str = "outputs"):
    """Show current progress."""
    progress_file, current_file, answers_file = find_latest_progress_files(output_dir)
    
    if not progress_file:
        print("❌ No progress files found. Make sure the processing has started.")
        return
    
    print(f"📊 Latest Progress Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Show progress summary
    if progress_file.exists():
        with open(progress_file, 'r', encoding='utf-8') as f:
            content = f.read()
            print(content)
    
    # Show file info
    print("\n📁 File Information:")
    print(f"   Progress file: {progress_file.name}")
    print(f"   Last updated: {datetime.fromtimestamp(progress_file.stat().st_mtime).strftime('%H:%M:%S')}")
    
    if current_file and current_file.exists():
        print(f"   Current results: {current_file.name} ({current_file.stat().st_size / 1024:.1f} KB)")
    
    if answers_file and answers_file.exists():
        print(f"   Current answers: {answers_file.name} ({answers_file.stat().st_size / 1024:.1f} KB)")


def show_latest_results(output_dir: str = "outputs", num_results: int = 3):
    """Show the latest processed results."""
    _, current_file, _ = find_latest_progress_files(output_dir)
    
    if not current_file or not current_file.exists():
        print("❌ No current results file found.")
        return
    
    try:
        with open(current_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        if not results:
            print("📭 No results processed yet.")
            return
        
        print(f"\n🔍 Latest {min(num_results, len(results))} Results:")
        print("-" * 60)
        
        for result in results[-num_results:]:
            q_num = result["question_index"] + 1
            status = result["status"]
            method = result.get("processing_method", "unknown")
            time_taken = result.get("processing_time_seconds", 0)
            
            status_emoji = "✅" if status == "success" else "❌"
            print(f"{status_emoji} Question {q_num}: {status} ({method}) - {time_taken:.2f}s")
            
            # Show question preview
            question_preview = result["question_text"][:80].replace('\n', ' ')
            print(f"   Q: {question_preview}...")
            
            # Show answer or error
            if status == "success" and result.get("final_answer"):
                answer_preview = result["final_answer"][:80].replace('\n', ' ')
                print(f"   A: {answer_preview}...")
            elif status == "error":
                error_msg = result.get("primary_error", "Unknown error")[:80]
                print(f"   Error: {error_msg}")
            
            print()
            
    except Exception as e:
        print(f"❌ Error reading results file: {e}")


def show_answers(output_dir: str = "outputs"):
    """Show successful answers."""
    _, _, answers_file = find_latest_progress_files(output_dir)
    
    if not answers_file or not answers_file.exists():
        print("❌ No answers file found.")
        return
    
    try:
        with open(answers_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print("\n📝 Current Successful Answers:")
        print("=" * 60)
        print(content)
        
    except Exception as e:
        print(f"❌ Error reading answers file: {e}")


def monitor_continuous(output_dir: str = "outputs", interval: int = 30):
    """Monitor progress continuously."""
    print(f"🔄 Starting continuous monitoring (updates every {interval} seconds)")
    print("Press Ctrl+C to stop monitoring")
    print("=" * 60)
    
    try:
        while True:
            # Clear screen (works on most terminals)
            print("\033[2J\033[H", end="")
            
            show_progress(output_dir)
            show_latest_results(output_dir, num_results=2)
            
            print(f"\n⏰ Next update in {interval} seconds... (Ctrl+C to stop)")
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped.")


def main():
    parser = argparse.ArgumentParser(description="Monitor MedBrowse processing progress")
    parser.add_argument(
        "--output-dir", 
        default="outputs",
        help="Output directory to monitor (default: outputs)"
    )
    parser.add_argument(
        "--mode", 
        choices=["progress", "results", "answers", "continuous"],
        default="progress",
        help="What to show: progress, results, answers, or continuous monitoring"
    )
    parser.add_argument(
        "--interval", 
        type=int, 
        default=30,
        help="Update interval for continuous monitoring (seconds)"
    )
    parser.add_argument(
        "--num-results", 
        type=int, 
        default=5,
        help="Number of latest results to show"
    )
    
    args = parser.parse_args()
    
    if args.mode == "progress":
        show_progress(args.output_dir)
    elif args.mode == "results":
        show_latest_results(args.output_dir, args.num_results)
    elif args.mode == "answers":
        show_answers(args.output_dir)
    elif args.mode == "continuous":
        monitor_continuous(args.output_dir, args.interval)


if __name__ == "__main__":
    main() 