#!/usr/bin/env python3
"""
Script to process questions from the MedBrowseComp dataset through the multi-agent pipeline.
"""

import asyncio
import json
import csv
import logging
import sys
import io
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
import re

# Third-party imports
import pandas as pd
from datasets import load_dataset

# Local imports
from src.workflow import run_agent_workflow_async

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class OutputCapture:
    """Utility class to capture stdout/stderr and workflow outputs."""
    
    def __init__(self):
        self.stdout_capture = io.StringIO()
        self.stderr_capture = io.StringIO()
        self.captured_output = ""
        self.captured_errors = ""
    
    def __enter__(self):
        self.stdout_backup = sys.stdout
        self.stderr_backup = sys.stderr
        sys.stdout = self.stdout_capture
        sys.stderr = self.stderr_capture
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.stdout_backup
        sys.stderr = self.stderr_backup
        self.captured_output = self.stdout_capture.getvalue()
        self.captured_errors = self.stderr_capture.getvalue()
    
    def get_output(self) -> str:
        return self.captured_output
    
    def get_errors(self) -> str:
        return self.captured_errors


class MedBrowseProcessor:
    """Processor for MedBrowseComp dataset questions."""
    
    def __init__(
        self,
        output_dir: str = "outputs",
        max_plan_iterations: int = 2,  # Set to 2 as requested
        max_step_num: int = 3,         # Always set to 3 as requested
        enable_background_investigation: bool = False,
        debug: bool = False,
        enable_fallback: bool = True,
        fallback_timeout: int = 300,   # 5 minutes timeout for fallback
        max_retries: int = 5           # Add retry limit for parsing errors (increased to 5)
    ):
        """Initialize the processor.
        
        Args:
            output_dir: Directory to store results
            max_plan_iterations: Maximum number of plan iterations for each question
            max_step_num: Maximum number of steps in a plan
            enable_background_investigation: Whether to enable background investigation
            debug: Whether to enable debug logging
            enable_fallback: Whether to enable fallback mechanisms
            fallback_timeout: Timeout in seconds for fallback processing
            max_retries: Maximum number of retries for parsing errors
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.max_plan_iterations = max_plan_iterations
        self.max_step_num = max_step_num
        self.enable_background_investigation = enable_background_investigation
        self.debug = debug
        self.enable_fallback = enable_fallback
        self.fallback_timeout = fallback_timeout
        self.max_retries = max_retries
        
        # Generate timestamp for unique output files
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = []
        
    def load_dataset(self, dataset_name: str = "AIM-Harvard/MedBrowseComp", 
                    split: str = "MedBrowseComp_50") -> pd.DataFrame:
        """Load the MedBrowseComp dataset from HuggingFace.
        
        Args:
            dataset_name: Name of the dataset on HuggingFace
            split: Which split to load (default: MedBrowseComp_50)
            
        Returns:
            DataFrame containing the dataset
        """
        try:
            logger.info(f"Loading dataset {dataset_name}, split: {split}")
            dataset = load_dataset(dataset_name, split=split)
            df = dataset.to_pandas()
            logger.info(f"Successfully loaded {len(df)} questions from the dataset")
            return df
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    async def process_single_question(self, question_data: Dict[str, Any], 
                                    question_idx: int) -> Dict[str, Any]:
        """Process a single question through the multi-agent pipeline.
        
        Args:
            question_data: Dictionary containing question data
            question_idx: Index of the question for tracking
            
        Returns:
            Dictionary containing the question, answer, and metadata
        """
        # Extract the question text - adjust field name based on actual dataset structure
        question_text = self._extract_question_text(question_data)
        
        logger.info(f"Processing question {question_idx + 1}: {question_text[:100]}...")
        
        start_time = datetime.now()
        
        # Try primary approach with retries
        primary_errors = []
        
        for retry_attempt in range(self.max_retries):
            try:
                if retry_attempt > 0:
                    logger.info(f"Retry attempt {retry_attempt + 1}/{self.max_retries} for question {question_idx + 1}")
                
                result = await self._run_workflow_with_capture(question_text, is_fallback=False, retry_attempt=retry_attempt)
                result.update({
                    "question_index": question_idx,
                    "question_text": question_text,
                    "original_data": question_data,
                    "processing_time_seconds": (datetime.now() - start_time).total_seconds(),
                    "timestamp": start_time.isoformat(),
                    "status": "success",
                    "processing_method": "primary",
                    "retry_attempt": retry_attempt,
                    # Store all captured messages for both success and error analysis
                    "workflow_output": result.get("workflow_output", ""),
                    "captured_output": result.get("captured_output", ""),
                    "captured_errors": result.get("captured_errors"),
                    "output_length": result.get("output_length", 0),
                    "has_errors": result.get("has_errors", False),
                    "final_answer": result.get("final_answer")
                })
                
                logger.info(f"Successfully processed question {question_idx + 1} in {result['processing_time_seconds']:.2f}s (attempt {retry_attempt + 1})")
                return result
                
            except Exception as error:
                primary_errors.append(str(error))
                
                # Check if it's a parsing error that we should retry
                error_msg = str(error).lower()
                is_parsing_error = any(keyword in error_msg for keyword in [
                    "failed to parse", "parse error", "json", "completion", 
                    "invalid", "decode", "format", "validation error", "pydantic",
                    "field required", "missing", "output_parsing_failure"
                ])
                
                # Also check for specific Plan parsing errors
                is_plan_error = "failed to parse plan" in error_msg or "validation errors for plan" in error_msg
                
                if (is_parsing_error or is_plan_error) and retry_attempt < self.max_retries - 1:
                    wait_time = (2 ** retry_attempt) + 1  # 2s, 3s, 5s delays
                    logger.warning(f"Parser error (attempt {retry_attempt + 1}/{self.max_retries}) for question {question_idx + 1}: {str(error)[:200]}...")
                    logger.info(f"Will retry in {wait_time}s... ({retry_attempt + 2}/{self.max_retries})")
                    # Add a delay before retry
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    # If not a parsing error or we've exhausted retries
                    if retry_attempt == self.max_retries - 1:
                        if is_parsing_error or is_plan_error:
                            logger.error(f"All {self.max_retries} retry attempts failed with parsing errors for question {question_idx + 1}")
                        else:
                            logger.error(f"Non-parsing error encountered for question {question_idx + 1}: {str(error)[:200]}...")
                    break
        
        # All primary attempts failed
        primary_error = primary_errors[-1] if primary_errors else "Unknown error"
        logger.warning(f"Primary processing failed for question {question_idx + 1}: {primary_error}")
        
        # Try fallback approach if enabled
        if self.enable_fallback:
            try:
                logger.info(f"Attempting fallback processing for question {question_idx + 1}")
                result = await self._run_workflow_with_fallback(question_text)
                result.update({
                    "question_index": question_idx,
                    "question_text": question_text,
                    "original_data": question_data,
                    "processing_time_seconds": (datetime.now() - start_time).total_seconds(),
                    "timestamp": start_time.isoformat(),
                    "status": "success",
                    "processing_method": "fallback",
                    "primary_error": str(primary_error),
                    # Store all captured messages for both success and error analysis
                    "workflow_output": result.get("workflow_output", ""),
                    "captured_output": result.get("captured_output", ""),
                    "captured_errors": result.get("captured_errors"),
                    "output_length": result.get("output_length", 0),
                    "has_errors": result.get("has_errors", False),
                    "final_answer": result.get("final_answer"),
                    "fallback_params": result.get("fallback_params", {})
                })
                
                logger.info(f"Successfully processed question {question_idx + 1} via fallback in {result['processing_time_seconds']:.2f}s")
                return result
                
            except Exception as fallback_error:
                logger.error(f"Both primary and fallback processing failed for question {question_idx + 1}")
                return self._create_error_result(
                    question_idx, question_text, question_data, start_time,
                    primary_error, fallback_error
                )
        else:
            logger.error(f"Processing failed for question {question_idx + 1}, fallback disabled")
            return self._create_error_result(
                question_idx, question_text, question_data, start_time,
                primary_error, None
            )
    
    def _create_error_result(self, question_idx: int, question_text: str, 
                           question_data: Dict[str, Any], start_time: datetime,
                           primary_error: Exception, fallback_error: Optional[Exception]) -> Dict[str, Any]:
        """Create an error result dictionary."""
        result = {
            "question_index": question_idx,
            "question_text": question_text,
            "original_data": question_data,
            "workflow_output": f"Error: {str(primary_error)}",
            "captured_output": "",  # Initialize as empty string instead of None
            "captured_errors": str(primary_error),  # Store error as captured error
            "final_answer": None,
            "output_length": 0,
            "has_errors": True,
            "processing_time_seconds": (datetime.now() - start_time).total_seconds(),
            "timestamp": start_time.isoformat(),
            "status": "error",
            "processing_method": "failed",
            "primary_error": str(primary_error),
            "error_details": {
                "primary_error_type": type(primary_error).__name__,
                "primary_error_message": str(primary_error)
            }
        }
        
        if fallback_error:
            result["fallback_error"] = str(fallback_error)
            result["error_details"]["fallback_error_type"] = type(fallback_error).__name__
            result["error_details"]["fallback_error_message"] = str(fallback_error)
        
        return result
    
    def _extract_question_text(self, question_data: Dict[str, Any]) -> str:
        """Extract question text from the question data.
        
        This method handles different possible field names in the dataset.
        """
        # Common field names that might contain the question
        possible_fields = [
            "question", "query", "text", "prompt", "input", 
            "Question", "Query", "Text", "Prompt", "Input"
        ]
        
        for field in possible_fields:
            if field in question_data and question_data[field]:
                return str(question_data[field])
        
        # If no standard field found, use the first string value
        for key, value in question_data.items():
            if isinstance(value, str) and len(value) > 10:  # Assume questions are at least 10 chars
                logger.warning(f"Using field '{key}' as question text")
                return value
        
        # Fallback: return JSON representation
        logger.warning("Could not identify question field, using full data as question")
        return json.dumps(question_data)
    
    async def _run_workflow_with_capture(self, question: str, is_fallback: bool = False, retry_attempt: int = 0) -> Dict[str, Any]:
        """Run the workflow and capture its output comprehensively.
        
        Args:
            question: The question to process
            is_fallback: Whether this is a fallback attempt
            retry_attempt: Which retry attempt this is (0 = first attempt)
            
        Returns:
            Dictionary containing workflow results and captured output
        """
        with OutputCapture() as capture:
            try:
                logger.info(f"Running workflow (attempt {retry_attempt + 1}) with params: max_plan_iterations={self.max_plan_iterations}, max_step_num={self.max_step_num}")
                
                # Run the workflow with output capture
                workflow_result = await run_agent_workflow_async(
                    user_input=question,
                    debug=self.debug,
                    max_plan_iterations=self.max_plan_iterations,
                    max_step_num=self.max_step_num,
                    enable_background_investigation=self.enable_background_investigation,
                )
                
                captured_output = capture.get_output()
                captured_errors = capture.get_errors()
                
                logger.info(f"Workflow completed. Output length: {len(captured_output)} chars, Errors: {len(captured_errors) if captured_errors else 0} chars")
                
                # Extract meaningful answer from captured output
                final_answer = self._extract_final_answer(captured_output)
                
                if final_answer:
                    logger.info(f"Successfully extracted answer: {final_answer[:100]}...")
                else:
                    logger.warning("No answer could be extracted from workflow output")
                
                return {
                    "workflow_output": str(workflow_result) if workflow_result else "Workflow completed successfully",
                    "final_answer": final_answer,
                    "captured_output": captured_output,
                    "captured_errors": captured_errors if captured_errors else None,
                    "output_length": len(captured_output),
                    "has_errors": bool(captured_errors)
                }
                
            except Exception as e:
                captured_output = capture.get_output()
                captured_errors = capture.get_errors()
                
                # Even if workflow failed, try to extract any partial answer
                partial_answer = self._extract_final_answer(captured_output) if captured_output else None
                
                error_info = {
                    "workflow_output": f"Error: {str(e)}",
                    "final_answer": partial_answer,
                    "captured_output": captured_output if captured_output else None,
                    "captured_errors": captured_errors if captured_errors else None,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
                
                logger.error(f"Workflow execution failed: {e}")
                raise
    
    async def _run_workflow_with_fallback(self, question: str) -> Dict[str, Any]:
        """Run workflow with simplified parameters as fallback.
        
        Args:
            question: The question to process
            
        Returns:
            Dictionary containing fallback workflow results
        """
        logger.info("Using fallback workflow with simplified parameters")
        
        # Use more conservative settings for fallback
        fallback_params = {
            "max_plan_iterations": 1,
            "max_step_num": 2,  # Increased from 1 to 2 to allow research + answer
            "enable_background_investigation": False
        }
        
        logger.info(f"Fallback using: {fallback_params}")
        
        with OutputCapture() as capture:
            try:
                # Add timeout for fallback
                workflow_result = await asyncio.wait_for(
                    run_agent_workflow_async(
                        user_input=question,
                        debug=True,  # Enable debug to get more output for fallback
                        **fallback_params
                    ),
                    timeout=self.fallback_timeout
                )
                
                captured_output = capture.get_output()
                captured_errors = capture.get_errors()
                
                final_answer = self._extract_final_answer(captured_output)
                
                return {
                    "workflow_output": str(workflow_result) if workflow_result else "Fallback workflow completed",
                    "final_answer": final_answer,
                    "captured_output": captured_output,
                    "captured_errors": captured_errors if captured_errors else None,
                    "fallback_params": fallback_params,
                    "output_length": len(captured_output),
                    "has_errors": bool(captured_errors)
                }
                
            except asyncio.TimeoutError:
                logger.error(f"Fallback workflow timed out after {self.fallback_timeout} seconds")
                raise Exception(f"Fallback workflow timed out after {self.fallback_timeout} seconds")
            except Exception as e:
                captured_output = capture.get_output()
                captured_errors = capture.get_errors()
                
                partial_answer = self._extract_final_answer(captured_output) if captured_output else None
                
                logger.error(f"Fallback workflow execution failed: {e}")
                raise Exception(f"Fallback failed: {str(e)}")
    
    def _extract_final_answer(self, captured_output: str) -> Optional[str]:
        """Extract the final answer from captured output.
        
        Args:
            captured_output: The captured stdout from workflow execution
            
        Returns:
            Extracted final answer or None
        """
        if not captured_output:
            return None
        
        # First, look for the specific INGREDIENT: format the questions expect
        ingredient_pattern = r'INGREDIENT:\s*([A-Z\s]+(?:[A-Z\s]*[A-Z]+)*)'
        ingredient_matches = re.findall(ingredient_pattern, captured_output, re.IGNORECASE)
        if ingredient_matches:
            # Return the last (most recent) ingredient match
            ingredient = ingredient_matches[-1].strip().upper()
            logger.info(f"Found ingredient pattern: INGREDIENT: {ingredient}")
            return f"INGREDIENT: {ingredient}"
        
        # Look for standalone ingredient names (all caps words)
        caps_pattern = r'\b([A-Z]{3,}(?:\s+[A-Z]{3,})*)\b'
        caps_matches = re.findall(caps_pattern, captured_output)
        if caps_matches:
            # Filter out common non-ingredient words
            excluded_words = {'HTTP', 'POST', 'GET', 'INFO', 'ERROR', 'DEBUG', 'WARNING', 'TRUE', 'FALSE', 'NULL', 'JSON', 'API'}
            filtered_matches = [match for match in caps_matches if not any(word in excluded_words for word in match.split())]
            if filtered_matches:
                ingredient = filtered_matches[-1].strip()
                logger.info(f"Found caps ingredient: {ingredient}")
                return f"INGREDIENT: {ingredient}"
        
        # Look for common patterns in output that indicate final answers
        lines = captured_output.split('\n')
        
        # Try to find the last substantial content block
        answer_lines = []
        current_block = []
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_block:
                    answer_lines.extend(current_block)
                    current_block = []
                continue
            
            # Skip log lines and technical output
            if any(skip_pattern in line.lower() for skip_pattern in [
                'info -', 'error -', 'warning -', 'debug -',
                'http request:', 'loading', 'installing',
                'node is', 'coordinator', 'planner', 'researcher',
                'executing step:', 'starting', 'completed'
            ]):
                continue
            
            current_block.append(line)
        
        # Add final block
        if current_block:
            answer_lines.extend(current_block)
        
        if answer_lines:
            # Join the last substantial block as the answer
            final_answer = '\n'.join(answer_lines[-20:])  # Last 20 lines max
            if len(final_answer.strip()) > 10:
                logger.info(f"Extracted general answer from output: {final_answer[:100]}...")
                return final_answer
        
        logger.warning("No meaningful answer found in captured output")
        return None
    
    async def process_all_questions(self, df: pd.DataFrame, 
                                  start_idx: int = 0, 
                                  end_idx: int = None) -> List[Dict[str, Any]]:
        """Process all questions in the dataset.
        
        Args:
            df: DataFrame containing the questions
            start_idx: Starting index for processing (for resuming interrupted runs)
            end_idx: Ending index for processing (None means process all)
            
        Returns:
            List of results for all processed questions
        """
        if end_idx is None:
            end_idx = len(df)
        
        logger.info(f"Processing questions {start_idx} to {end_idx - 1} ({end_idx - start_idx} total)")
        
        for idx in range(start_idx, end_idx):
            question_data = df.iloc[idx].to_dict()
            
            logger.info(f"🔄 Starting question {idx + 1}/{end_idx}")
            result = await self.process_single_question(question_data, idx)
            self.results.append(result)
            
            # Log immediate result
            status_emoji = "✅" if result["status"] == "success" else "❌"
            method = result.get("processing_method", "unknown")
            time_taken = result.get("processing_time_seconds", 0)
            
            logger.info(f"{status_emoji} Question {idx + 1} completed - Status: {result['status']}, Method: {method}, Time: {time_taken:.2f}s")
            
            if result["status"] == "success" and result.get("final_answer"):
                answer_preview = result["final_answer"][:100].replace('\n', ' ')
                logger.info(f"📝 Answer preview: {answer_preview}...")
            elif result["status"] == "error":
                error_msg = result.get("primary_error", "Unknown error")[:100]
                logger.info(f"❌ Error: {error_msg}")
            
            # Save results after EVERY question (not just every 5)
            self._save_iterative_results(idx + 1, end_idx)
            
            # Also save intermediate results every 5 questions for backup
            if (idx + 1) % 5 == 0:
                self._save_intermediate_results()
                self._generate_progress_summary(idx + 1, end_idx)
        
        return self.results
    
    def _save_intermediate_results(self):
        """Save intermediate results to avoid losing progress."""
        intermediate_file = self.output_dir / f"medbrowse_results_intermediate_{self.timestamp}.json"
        with open(intermediate_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 Saved intermediate results to {intermediate_file}")
    
    def _save_iterative_results(self, current_idx: int, total_questions: int):
        """Save results after each question for real-time monitoring."""
        # Save the latest results
        iterative_file = self.output_dir / f"medbrowse_current_{self.timestamp}.json"
        with open(iterative_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        # Save a progress file with just the summary
        progress_file = self.output_dir / f"medbrowse_progress_{self.timestamp}.txt"
        
        successful = len([r for r in self.results if r["status"] == "success"])
        failed = len(self.results) - successful
        success_rate = (successful / len(self.results)) * 100 if self.results else 0
        
        with open(progress_file, 'w', encoding='utf-8') as f:
            f.write(f"MedBrowse Processing Progress - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n")
            f.write(f"Progress: {current_idx}/{total_questions} questions ({(current_idx/total_questions)*100:.1f}%)\n")
            f.write(f"Successful: {successful}\n")
            f.write(f"Failed: {failed}\n")
            f.write(f"Success Rate: {success_rate:.1f}%\n")
            f.write("-" * 60 + "\n")
            
            # Show last few results
            f.write("Last 5 Results:\n")
            for result in self.results[-5:]:
                status = result["status"]
                method = result.get("processing_method", "unknown")
                time_taken = result.get("processing_time_seconds", 0)
                q_num = result["question_index"] + 1
                f.write(f"  Q{q_num}: {status} ({method}) - {time_taken:.2f}s\n")
        
        # Also save successful answers incrementally
        self._save_current_answers()
    
    def _save_current_answers(self):
        """Save successful answers incrementally."""
        successful_results = [r for r in self.results if r["status"] == "success" and r.get("final_answer")]
        
        if not successful_results:
            return
        
        current_answers_file = self.output_dir / f"medbrowse_current_answers_{self.timestamp}.txt"
        with open(current_answers_file, 'w', encoding='utf-8') as f:
            f.write(f"MedBrowse Current Answers - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Successful Answers: {len(successful_results)}\n")
            f.write("=" * 80 + "\n\n")
            
            for result in successful_results:
                f.write(f"Question {result['question_index'] + 1}:\n")
                f.write(f"Q: {result['question_text']}\n\n")
                f.write(f"A: {result.get('final_answer', 'No answer extracted')}\n\n")
                f.write(f"Method: {result.get('processing_method', 'unknown')}, Time: {result.get('processing_time_seconds', 0):.2f}s\n")
                f.write("-" * 80 + "\n\n")
    
    def _generate_progress_summary(self, current_idx: int, total_questions: int):
        """Generate and log a progress summary."""
        successful = len([r for r in self.results if r["status"] == "success"])
        failed = len(self.results) - successful
        success_rate = (successful / len(self.results)) * 100 if self.results else 0
        
        primary_success = len([r for r in self.results if r.get("processing_method") == "primary"])
        fallback_success = len([r for r in self.results if r.get("processing_method") == "fallback"])
        with_answers = len([r for r in self.results if r.get("final_answer")])
        
        # Calculate retry statistics
        retry_counts = {}
        for r in self.results:
            if r["status"] == "success" and "retry_attempt" in r:
                retry_attempt = r["retry_attempt"]
                retry_counts[retry_attempt] = retry_counts.get(retry_attempt, 0) + 1
        
        # Calculate average processing time
        successful_times = [r["processing_time_seconds"] for r in self.results 
                          if r["status"] == "success" and r.get("processing_time_seconds")]
        avg_time = sum(successful_times) / len(successful_times) if successful_times else 0
        
        logger.info("📊 PROGRESS SUMMARY:")
        logger.info(f"   Progress: {current_idx}/{total_questions} ({(current_idx/total_questions)*100:.1f}%)")
        logger.info(f"   ✅ Successful: {successful} ({success_rate:.1f}%)")
        logger.info(f"   ❌ Failed: {failed}")
        logger.info(f"   🎯 Primary method: {primary_success}, 🔄 Fallback: {fallback_success}")
        logger.info(f"   📝 With answers: {with_answers}")
        logger.info(f"   ⏱️  Avg time: {avg_time:.2f}s")
        if retry_counts:
            retry_summary = ", ".join([f"Attempt {k+1}: {v}" for k, v in sorted(retry_counts.items())])
            logger.info(f"   🔁 Retry distribution: {retry_summary}")
        logger.info(f"Summary saved to {summary_file}")
    
    def save_results(self, results: List[Dict[str, Any]] = None):
        """Save processing results to files with comprehensive data storage like MedXpert.
        
        Args:
            results: Results to save (if None, uses self.results)
        """
        if results is None:
            results = self.results
        
        if not results:
            logger.warning("No results to save")
            return
        
        # Generate comprehensive results
        timestamp = self.timestamp
        
        # Save detailed JSON results
        json_file = self.output_dir / f"medbrowse_results_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # Save comprehensive CSV summary (like MedXpert)
        csv_file = self.output_dir / f"medbrowse_results_{timestamp}.csv"
        csv_data = []
        
        for result in results:
            csv_data.append({
                'question_index': result.get('question_index', ''),
                'question_text': str(result.get('question_text', ''))[:500],  # Truncate long text
                'status': result.get('status', ''),
                'processing_method': result.get('processing_method', ''),
                'processing_time_seconds': result.get('processing_time_seconds', 0),
                'retry_attempt': result.get('retry_attempt', 0),
                'has_final_answer': bool(result.get('final_answer')),
                'final_answer': str(result.get('final_answer', ''))[:200] if result.get('final_answer') else '',
                'workflow_output': str(result.get('workflow_output', ''))[:200],  # Truncate for CSV
                'captured_output_length': result.get('output_length', 0),
                'has_errors': result.get('has_errors', False),
                'captured_errors': str(result.get('captured_errors', ''))[:200] if result.get('captured_errors') else '',
                'primary_error': str(result.get('primary_error', ''))[:200] if result.get('primary_error') else '',
                'fallback_error': str(result.get('fallback_error', ''))[:200] if result.get('fallback_error') else '',
                'timestamp': result.get('timestamp', '')
            })
        
        pd.DataFrame(csv_data).to_csv(csv_file, index=False)
        
        # Save successful answers separately
        self._save_successful_answers(results)
        
        # Generate summary report
        self._generate_summary_report(results)
        
        logger.info(f"Results saved:")
        logger.info(f"  📋 JSON: {json_file}")
        logger.info(f"  📊 CSV: {csv_file}")
    
    def _save_successful_answers(self, results: List[Dict[str, Any]]):
        """Save successful answers in a readable format."""
        successful_results = [r for r in results if r["status"] == "success" and r.get("final_answer")]
        
        if not successful_results:
            logger.warning("No successful answers to save")
            return
        
        answers_file = self.output_dir / f"medbrowse_answers_{self.timestamp}.txt"
        with open(answers_file, 'w', encoding='utf-8') as f:
            f.write(f"MedBrowse Dataset Processing Results\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Total Successful Answers: {len(successful_results)}\n")
            f.write("=" * 80 + "\n\n")
            
            for result in successful_results:
                f.write(f"Question {result['question_index'] + 1}:\n")
                f.write(f"Q: {result['question_text']}\n\n")
                f.write(f"A: {result.get('final_answer', 'No answer extracted')}\n\n")
                f.write(f"Processing Method: {result.get('processing_method', 'unknown')}\n")
                f.write(f"Processing Time: {result.get('processing_time_seconds', 0):.2f}s\n")
                f.write("-" * 80 + "\n\n")
        
        logger.info(f"Saved {len(successful_results)} successful answers to {answers_file}")
    
    def _generate_summary_report(self, results: List[Dict[str, Any]]):
        """Generate a comprehensive summary report like MedXpert."""
        total_questions = len(results)
        successful_results = [r for r in results if r.get('status') == 'success']
        
        success_rate = len(successful_results) / total_questions if total_questions > 0 else 0
        avg_time = sum(r.get('processing_time_seconds', 0) for r in results) / total_questions if total_questions > 0 else 0
        
        # Calculate processing method stats
        primary_success = len([r for r in results if r.get("processing_method") == "primary"])
        fallback_success = len([r for r in results if r.get("processing_method") == "fallback"])
        
        # Calculate retry statistics
        retry_counts = {}
        for r in results:
            if "retry_attempt" in r:
                retry_attempt = r["retry_attempt"]
                retry_counts[retry_attempt] = retry_counts.get(retry_attempt, 0) + 1
        
        # Count questions with extracted answers
        with_answers = len([r for r in results if r.get("final_answer")])
        
        summary = {
            'processing_summary': {
                'total_questions': total_questions,
                'successful_questions': len(successful_results),
                'failed_questions': total_questions - len(successful_results),
                'success_rate': success_rate,
                'questions_with_extracted_answers': with_answers,
                'answer_extraction_rate': with_answers / total_questions if total_questions > 0 else 0,
                'avg_processing_time_seconds': avg_time,
                'timestamp': datetime.now().isoformat()
            },
            'detailed_metrics': {
                'by_processing_method': self._group_by_field(results, 'processing_method'),
                'by_status': self._group_by_field(results, 'status'),
                'retry_statistics': retry_counts,
                'processing_stats': {
                    'primary_method_success': primary_success,
                    'fallback_method_success': fallback_success,
                    'total_failed': total_questions - (primary_success + fallback_success)
                },
                'processing_settings': {
                    'max_plan_iterations': self.max_plan_iterations,
                    'max_step_num': self.max_step_num,
                    'enable_background_investigation': self.enable_background_investigation,
                    'enable_fallback': self.enable_fallback,
                    'fallback_timeout': self.fallback_timeout,
                    'max_retries': self.max_retries
                }
            }
        }
        
        # Save summary
        summary_file = self.output_dir / f"medbrowse_summary_{self.timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"📄 Summary: {summary_file}")
        logger.info(f"   Success Rate: {success_rate:.2%}")
        logger.info(f"   Answer Extraction Rate: {with_answers / total_questions:.2%}" if total_questions > 0 else "   Answer Extraction Rate: 0%")
        logger.info(f"   Primary Method: {primary_success}, Fallback Method: {fallback_success}")
        logger.info(f"   Average Processing Time: {avg_time:.2f}s")

    def _group_by_field(self, results: List[Dict[str, Any]], field: str) -> Dict[str, Any]:
        """Group results by a specific field and calculate metrics (like MedXpert)."""
        groups = {}
        
        for result in results:
            field_value = result.get(field, 'Unknown')
            if field_value not in groups:
                groups[field_value] = {
                    'total': 0,
                    'successful': 0,
                    'with_answers': 0,
                    'success_rate': 0,
                    'answer_extraction_rate': 0
                }
            
            groups[field_value]['total'] += 1
            if result.get('status') == 'success':
                groups[field_value]['successful'] += 1
            if result.get('final_answer'):
                groups[field_value]['with_answers'] += 1
        
        # Calculate rates
        for group_data in groups.values():
            total = group_data['total']
            group_data['success_rate'] = group_data['successful'] / total if total > 0 else 0
            group_data['answer_extraction_rate'] = group_data['with_answers'] / total if total > 0 else 0
        
        return groups


async def main():
    """Main function to run the MedBrowse dataset processing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process MedBrowseComp dataset questions")
    parser.add_argument(
        "--dataset", 
        default="AIM-Harvard/MedBrowseComp",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--split", 
        default="MedBrowseComp_50",
        help="Dataset split to process"
    )
    parser.add_argument(
        "--output-dir", 
        default="outputs",
        help="Output directory for results"
    )
    parser.add_argument(
        "--start-idx", 
        type=int, 
        default=0,
        help="Starting index for processing"
    )
    parser.add_argument(
        "--end-idx", 
        type=int,
        help="Ending index for processing (None = all)"
    )
    parser.add_argument(
        "--max-plan-iterations", 
        type=int, 
        default=2,  # Changed default to 2
        help="Maximum plan iterations per question"
    )
    parser.add_argument(
        "--max-step-num", 
        type=int, 
        default=3,  # Always set to 3
        help="Maximum steps per plan"
    )
    parser.add_argument(
        "--disable-background-investigation", 
        action="store_true",
        help="Disable background investigation"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--disable-fallback",
        action="store_true",
        help="Disable fallback processing"
    )
    parser.add_argument(
        "--fallback-timeout",
        type=int,
        default=300,
        help="Timeout for fallback processing in seconds (default: 300)"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum number of retries for parsing errors (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Create processor
    processor = MedBrowseProcessor(
        output_dir=args.output_dir,
        max_plan_iterations=args.max_plan_iterations,
        max_step_num=args.max_step_num,
        enable_background_investigation=not args.disable_background_investigation,
        debug=args.debug,
        enable_fallback=not args.disable_fallback,
        fallback_timeout=args.fallback_timeout,
        max_retries=args.max_retries
    )
    
    try:
        # Load dataset
        df = processor.load_dataset(args.dataset, args.split)
        
        # Process questions
        results = await processor.process_all_questions(
            df, 
            start_idx=args.start_idx, 
            end_idx=args.end_idx
        )
        
        # Save results
        processor.save_results(results)
        
        logger.info("Processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise


if __name__ == "__main__":
    # Install required package if not available
    try:
        import datasets
    except ImportError:
        logger.error("The 'datasets' package is required. Install it with: pip install datasets>=2.0.0")
        exit(1)
    
    asyncio.run(main()) 