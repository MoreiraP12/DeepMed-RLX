#!/usr/bin/env python3
"""
Ablation Study Script for MedXpertQA Text-Only Dataset Processing.

This script runs the multi-agent pipeline on the MedXpertQA text-only dataset with different
max_plan_iterations values (1 to 5) to study how the model improves with more
steps of research planning.

The MedXpertQA dataset contains medical multiple-choice questions with options A-E.

Usage:
    python ablation_study_medxpert.py [options]
    
Examples:
    # Run on first 10 questions (sample mode)
    python ablation_study_medxpert.py --sample-mode
    
    # Run on first 50 questions
    python ablation_study_medxpert.py --end-idx 50
    
    # Run with debug mode
    python ablation_study_medxpert.py --debug --sample-mode
"""

import asyncio
import json
import logging
import sys
import time
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add the current directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent))

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


class MedXpertProcessor:
    """Processor for MedXpertQA text-only dataset questions."""
    
    def __init__(
        self,
        output_dir: str = "outputs",
        max_plan_iterations: int = 2,
        max_step_num: int = 3,
        enable_background_investigation: bool = True,
        debug: bool = False,
        enable_fallback: bool = True,
        fallback_timeout: int = 300,
        max_retries: int = 5
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
    
    def load_dataset(self, dataset_name: str = "TsinghuaC3I/MedXpertQA", 
                    subset: str = "Text", split: str = "test") -> pd.DataFrame:
        """Load the MedXpertQA dataset.
        
        Args:
            dataset_name: HuggingFace dataset name
            subset: Dataset subset (Text for text-only questions)
            split: Dataset split to process
            
        Returns:
            DataFrame containing the questions
        """
        try:
            logger.info(f"Loading dataset {dataset_name}, subset: {subset}, split: {split}")
            
            # Load dataset from HuggingFace
            dataset = load_dataset(dataset_name, subset, split=split)
            
            # Convert to pandas DataFrame
            df = pd.DataFrame(dataset)
            
            logger.info(f"Successfully loaded {len(df)} questions from the dataset")
            logger.info(f"Dataset columns: {list(df.columns)}")
            
            # Show sample data structure
            if len(df) > 0:
                sample_data = df.iloc[0].to_dict()
                logger.info(f"Sample question structure:")
                for key, value in sample_data.items():
                    if isinstance(value, str):
                        logger.info(f"  {key}: {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")
                    else:
                        logger.info(f"  {key}: {value}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def format_question_for_pipeline(self, question_data: Dict[str, Any]) -> str:
        """Format a MedXpertQA question for the pipeline.
        
        Args:
            question_data: Raw question data from dataset
            
        Returns:
            Formatted question string for the pipeline
        """
        question_text = question_data.get('question', '')
        options = question_data.get('options', {})
        
        # Format as multiple choice question
        formatted_question = f"{question_text}\n\nAnswer Choices:\n"
        
        # Add options A-E
        for key in sorted(options.keys()):
            formatted_question += f"({key}) {options[key]}\n"
        
        formatted_question += "\nPlease provide your answer as a single letter (A, B, C, D, or E) followed by a brief explanation."
        
        return formatted_question
    
    async def process_single_question(self, question_data: Dict[str, Any], 
                                    question_idx: int) -> Dict[str, Any]:
        """Process a single question through the multi-agent pipeline.
        
        Args:
            question_data: Dictionary containing question data
            question_idx: Index of the question for tracking
            
        Returns:
            Dictionary containing the question, answer, and metadata
        """
        # Format the question for the pipeline
        formatted_question = self.format_question_for_pipeline(question_data)
        
        logger.info(f"Processing question {question_idx + 1}: {question_data.get('question', '')[:100]}...")
        
        start_time = datetime.now()
        
        # Try primary approach with retries
        primary_errors = []
        
        for retry_attempt in range(self.max_retries):
            try:
                if retry_attempt > 0:
                    logger.info(f"Retry attempt {retry_attempt + 1}/{self.max_retries} for question {question_idx + 1}")
                
                result = await self._run_workflow_with_capture(formatted_question, retry_attempt=retry_attempt)
                
                # Extract answer choice from pipeline output
                extracted_answer = self._extract_answer_choice(result.get('captured_output', ''))
                correct_answer = question_data.get('label', '')
                
                # Determine if answer is correct
                is_correct = extracted_answer.upper() == correct_answer.upper() if extracted_answer and correct_answer else False
                
                result.update({
                    "question_index": question_idx,
                    "question_id": question_data.get('id', f'q_{question_idx}'),
                    "question_text": question_data.get('question', ''),
                    "options": question_data.get('options', {}),
                    "correct_answer": correct_answer,
                    "extracted_answer": extracted_answer,
                    "is_correct": is_correct,
                    "formatted_question": formatted_question,
                    "original_data": question_data,
                    "processing_time_seconds": (datetime.now() - start_time).total_seconds(),
                    "timestamp": start_time.isoformat(),
                    "status": "success",
                    "processing_method": "primary",
                    "retry_attempt": retry_attempt,
                    "medical_task": question_data.get('medical_task', ''),
                    "body_system": question_data.get('body_system', ''),
                    "question_type": question_data.get('question_type', '')
                })
                
                logger.info(f"Successfully processed question {question_idx + 1} in {result['processing_time_seconds']:.2f}s")
                logger.info(f"   Answer: {extracted_answer} | Correct: {correct_answer} | Match: {is_correct}")
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
                
                if is_parsing_error and retry_attempt < self.max_retries - 1:
                    wait_time = (2 ** retry_attempt) + 1
                    logger.warning(f"Parser error (attempt {retry_attempt + 1}/{self.max_retries}) for question {question_idx + 1}")
                    logger.info(f"Will retry in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    break
        
        # All primary attempts failed - try fallback if enabled
        primary_error = primary_errors[-1] if primary_errors else "Unknown error"
        
        if self.enable_fallback:
            try:
                logger.info(f"Attempting fallback processing for question {question_idx + 1}")
                result = await self._run_workflow_with_fallback(formatted_question)
                
                extracted_answer = self._extract_answer_choice(result.get('captured_output', ''))
                correct_answer = question_data.get('label', '')
                is_correct = extracted_answer.upper() == correct_answer.upper() if extracted_answer and correct_answer else False
                
                result.update({
                    "question_index": question_idx,
                    "question_id": question_data.get('id', f'q_{question_idx}'),
                    "question_text": question_data.get('question', ''),
                    "options": question_data.get('options', {}),
                    "correct_answer": correct_answer,
                    "extracted_answer": extracted_answer,
                    "is_correct": is_correct,
                    "formatted_question": formatted_question,
                    "original_data": question_data,
                    "processing_time_seconds": (datetime.now() - start_time).total_seconds(),
                    "timestamp": start_time.isoformat(),
                    "status": "success",
                    "processing_method": "fallback",
                    "primary_error": str(primary_error),
                    "medical_task": question_data.get('medical_task', ''),
                    "body_system": question_data.get('body_system', ''),
                    "question_type": question_data.get('question_type', '')
                })
                
                logger.info(f"Successfully processed question {question_idx + 1} via fallback")
                return result
                
            except Exception as fallback_error:
                logger.error(f"Both primary and fallback processing failed for question {question_idx + 1}")
                return self._create_error_result(
                    question_idx, question_data, start_time, primary_error, fallback_error
                )
        else:
            return self._create_error_result(
                question_idx, question_data, start_time, primary_error, None
            )
    
    def _extract_answer_choice(self, captured_output: str) -> Optional[str]:
        """Extract the answer choice (A, B, C, D, E) from captured output.
        
        Args:
            captured_output: The captured stdout from workflow execution
            
        Returns:
            Extracted answer choice or None
        """
        if not captured_output:
            return None
        
        # Look for patterns like "Answer: A", "(A)", "The answer is A", etc.
        patterns = [
            r'\b(?:answer|choice|option)(?:\s*is)?\s*:?\s*\(?([A-E])\)?',
            r'\b([A-E])\)\s',  # Pattern like "A) "
            r'^\s*([A-E])\s*[:\.]',  # Pattern like "A:" or "A."
            r'\b(?:the\s+)?(?:correct\s+)?(?:answer|choice|option)\s+(?:is\s+)?(?:\(?([A-E])\)?)',
            r'\(([A-E])\)',  # Simple (A) pattern
            r'\b([A-E])\b(?=\s*(?:is|would|because|since))'  # Letter followed by explanation
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, captured_output, re.IGNORECASE | re.MULTILINE)
            if matches:
                # Return the last match (most likely the final answer)
                return matches[-1].upper()
        
        # Look for standalone capital letters A-E in the last part of output
        lines = captured_output.split('\n')
        for line in reversed(lines[-10:]):  # Check last 10 lines
            line = line.strip()
            if len(line) == 1 and line.upper() in 'ABCDE':
                return line.upper()
        
        logger.warning("Could not extract answer choice from output")
        return None
    
    async def _run_workflow_with_capture(self, question: str, retry_attempt: int = 0) -> Dict[str, Any]:
        """Run the workflow and capture its output."""
        import io
        import sys
        from contextlib import redirect_stdout, redirect_stderr
        
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                workflow_result = await run_agent_workflow_async(
                    user_input=question,
                    debug=self.debug,
                    max_plan_iterations=self.max_plan_iterations,
                    max_step_num=self.max_step_num,
                    enable_background_investigation=self.enable_background_investigation,
                )
            
            captured_output = stdout_capture.getvalue()
            captured_errors = stderr_capture.getvalue()
            
            return {
                "workflow_output": str(workflow_result) if workflow_result else "Workflow completed",
                "captured_output": captured_output,
                "captured_errors": captured_errors if captured_errors else None,
                "output_length": len(captured_output),
                "has_errors": bool(captured_errors)
            }
            
        except Exception as e:
            captured_output = stdout_capture.getvalue()
            captured_errors = stderr_capture.getvalue()
            
            logger.error(f"Workflow execution failed: {e}")
            raise e
    
    async def _run_workflow_with_fallback(self, question: str) -> Dict[str, Any]:
        """Run workflow with simplified parameters as fallback."""
        logger.info("Using fallback workflow with simplified parameters")
        
        # Store original parameters
        original_max_plan_iterations = self.max_plan_iterations
        original_max_step_num = self.max_step_num
        original_background_investigation = self.enable_background_investigation
        
        # Use fallback parameters
        self.max_plan_iterations = 1
        self.max_step_num = 2
        self.enable_background_investigation = False
        
        try:
            result = await self._run_workflow_with_capture(question, retry_attempt=0)
            return result
        finally:
            # Restore original parameters
            self.max_plan_iterations = original_max_plan_iterations
            self.max_step_num = original_max_step_num
            self.enable_background_investigation = original_background_investigation
    
    def _create_error_result(self, question_idx: int, question_data: Dict[str, Any], 
                           start_time: datetime, primary_error: Exception, 
                           fallback_error: Optional[Exception]) -> Dict[str, Any]:
        """Create an error result dictionary."""
        result = {
            "question_index": question_idx,
            "question_id": question_data.get('id', f'q_{question_idx}'),
            "question_text": question_data.get('question', ''),
            "options": question_data.get('options', {}),
            "correct_answer": question_data.get('label', ''),
            "extracted_answer": None,
            "is_correct": False,
            "formatted_question": self.format_question_for_pipeline(question_data),
            "original_data": question_data,
            "processing_time_seconds": (datetime.now() - start_time).total_seconds(),
            "timestamp": start_time.isoformat(),
            "status": "error",
            "processing_method": "failed",
            "primary_error": str(primary_error),
            "medical_task": question_data.get('medical_task', ''),
            "body_system": question_data.get('body_system', ''),
            "question_type": question_data.get('question_type', ''),
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
    
    async def process_all_questions(self, df: pd.DataFrame, 
                                  start_idx: int = 0, 
                                  end_idx: int = None) -> List[Dict[str, Any]]:
        """Process all questions in the dataset."""
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
            is_correct = result.get("is_correct", False)
            
            logger.info(f"{status_emoji} Question {idx + 1} completed - Status: {result['status']}, "
                       f"Method: {method}, Time: {time_taken:.2f}s, Correct: {is_correct}")
        
        return self.results
    
    def save_results(self, results: List[Dict[str, Any]] = None):
        """Save processing results to files."""
        if results is None:
            results = self.results
        
        # Generate comprehensive results
        timestamp = self.timestamp
        
        # Save detailed JSON results
        json_file = self.output_dir / f"medxpert_results_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # Save CSV summary
        csv_file = self.output_dir / f"medxpert_results_{timestamp}.csv"
        csv_data = []
        
        for result in results:
            csv_data.append({
                'question_id': result.get('question_id', ''),
                'question_index': result.get('question_index', ''),
                'status': result.get('status', ''),
                'is_correct': result.get('is_correct', False),
                'extracted_answer': result.get('extracted_answer', ''),
                'correct_answer': result.get('correct_answer', ''),
                'processing_time_seconds': result.get('processing_time_seconds', 0),
                'processing_method': result.get('processing_method', ''),
                'medical_task': result.get('medical_task', ''),
                'body_system': result.get('body_system', ''),
                'question_type': result.get('question_type', '')
            })
        
        pd.DataFrame(csv_data).to_csv(csv_file, index=False)
        
        # Generate summary report
        self._generate_summary_report(results)
        
        logger.info(f"Results saved:")
        logger.info(f"  📋 JSON: {json_file}")
        logger.info(f"  📊 CSV: {csv_file}")
    
    def _generate_summary_report(self, results: List[Dict[str, Any]]):
        """Generate a summary report of the processing results."""
        total_questions = len(results)
        successful_results = [r for r in results if r.get('status') == 'success']
        correct_results = [r for r in results if r.get('is_correct', False)]
        
        success_rate = len(successful_results) / total_questions if total_questions > 0 else 0
        accuracy_rate = len(correct_results) / total_questions if total_questions > 0 else 0
        
        # Calculate accuracy among successful results only
        accuracy_among_successful = len(correct_results) / len(successful_results) if successful_results else 0
        
        avg_time = sum(r.get('processing_time_seconds', 0) for r in results) / total_questions if total_questions > 0 else 0
        
        summary = {
            'processing_summary': {
                'total_questions': total_questions,
                'successful_questions': len(successful_results),
                'failed_questions': total_questions - len(successful_results),
                'success_rate': success_rate,
                'correct_answers': len(correct_results),
                'accuracy_rate': accuracy_rate,
                'accuracy_among_successful': accuracy_among_successful,
                'avg_processing_time_seconds': avg_time,
                'timestamp': datetime.now().isoformat()
            },
            'detailed_metrics': {
                'by_medical_task': self._group_by_field(results, 'medical_task'),
                'by_body_system': self._group_by_field(results, 'body_system'),
                'by_question_type': self._group_by_field(results, 'question_type'),
                'by_processing_method': self._group_by_field(results, 'processing_method')
            }
        }
        
        # Save summary
        summary_file = self.output_dir / f"medxpert_summary_{self.timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"📄 Summary: {summary_file}")
        logger.info(f"   Success Rate: {success_rate:.2%}")
        logger.info(f"   Overall Accuracy: {accuracy_rate:.2%}")
        logger.info(f"   Accuracy Among Successful: {accuracy_among_successful:.2%}")
    
    def _group_by_field(self, results: List[Dict[str, Any]], field: str) -> Dict[str, Any]:
        """Group results by a specific field and calculate metrics."""
        groups = {}
        
        for result in results:
            field_value = result.get(field, 'Unknown')
            if field_value not in groups:
                groups[field_value] = {
                    'total': 0,
                    'successful': 0,
                    'correct': 0,
                    'success_rate': 0,
                    'accuracy_rate': 0
                }
            
            groups[field_value]['total'] += 1
            if result.get('status') == 'success':
                groups[field_value]['successful'] += 1
            if result.get('is_correct', False):
                groups[field_value]['correct'] += 1
        
        # Calculate rates
        for group_data in groups.values():
            total = group_data['total']
            group_data['success_rate'] = group_data['successful'] / total if total > 0 else 0
            group_data['accuracy_rate'] = group_data['correct'] / total if total > 0 else 0
        
        return groups


class MedXpertAblationStudyRunner:
    """Runner for the ablation study with different max_plan_iterations values on MedXpertQA."""
    
    def __init__(
        self,
        dataset_name: str = "TsinghuaC3I/MedXpertQA",
        dataset_subset: str = "Text",
        dataset_split: str = "test",
        base_output_dir: str = "outputs/ablation_study_medxpert",
        start_idx: int = 0,
        end_idx: int = None,
        max_step_num: int = 3,
        enable_background_investigation: bool = True,
        debug: bool = False
    ):
        self.dataset_name = dataset_name
        self.dataset_subset = dataset_subset
        self.dataset_split = dataset_split
        self.base_output_dir = Path(base_output_dir)
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.max_step_num = max_step_num
        self.enable_background_investigation = enable_background_investigation
        self.debug = debug
        
        # Create base output directory
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp for this ablation study
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Store results for comparison
        self.ablation_results = {}
        
        # Range of max_plan_iterations to test (1 to 5)
        self.max_plan_iterations_range = list(range(1, 6))
    
    def create_processor(self, max_plan_iterations: int) -> MedXpertProcessor:
        """Create a MedXpertProcessor with specific max_plan_iterations."""
        output_dir = self.base_output_dir / f"max_plan_iterations_{max_plan_iterations}"
        
        return MedXpertProcessor(
            output_dir=str(output_dir),
            max_plan_iterations=max_plan_iterations,
            max_step_num=self.max_step_num,
            enable_background_investigation=self.enable_background_investigation,
            debug=self.debug,
            enable_fallback=True,
            fallback_timeout=300,
            max_retries=5
        )
    
    async def run_single_experiment(self, max_plan_iterations: int) -> Dict[str, Any]:
        """Run the experiment with a specific max_plan_iterations value."""
        print(f"\n{'='*60}")
        print(f"🧪 Running MedXpertQA Experiment: max_plan_iterations = {max_plan_iterations}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Create processor for this experiment
        processor = self.create_processor(max_plan_iterations)
        
        try:
            # Load dataset
            print(f"📥 Loading dataset: {self.dataset_name} ({self.dataset_subset}/{self.dataset_split})")
            df = processor.load_dataset(self.dataset_name, self.dataset_subset, self.dataset_split)
            
            # Determine question range
            total_questions = len(df)
            end_idx = self.end_idx if self.end_idx is not None else total_questions
            actual_end_idx = min(end_idx, total_questions)
            num_questions = actual_end_idx - self.start_idx
            
            print(f"✅ Dataset loaded: {total_questions} total questions")
            print(f"📋 Processing questions {self.start_idx} to {actual_end_idx-1} ({num_questions} questions)")
            
            # Process questions
            print(f"🔄 Processing with max_plan_iterations={max_plan_iterations}...")
            results = await processor.process_all_questions(
                df, 
                start_idx=self.start_idx, 
                end_idx=actual_end_idx
            )
            
            # Save results
            print("💾 Saving experiment results...")
            processor.save_results(results)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Calculate metrics
            successful_results = [r for r in results if r.get('status') == 'success']
            correct_results = [r for r in results if r.get('is_correct', False)]
            error_results = [r for r in results if r.get('status') == 'error']
            
            success_rate = len(successful_results) / len(results) if results else 0
            accuracy_rate = len(correct_results) / len(results) if results else 0
            accuracy_among_successful = len(correct_results) / len(successful_results) if successful_results else 0
            avg_processing_time = sum(r.get('processing_time_seconds', 0) for r in results) / len(results) if results else 0
            
            experiment_summary = {
                'max_plan_iterations': max_plan_iterations,
                'total_questions': num_questions,
                'successful_questions': len(successful_results),
                'failed_questions': len(error_results),
                'correct_answers': len(correct_results),
                'success_rate': success_rate,
                'accuracy_rate': accuracy_rate,
                'accuracy_among_successful': accuracy_among_successful,
                'total_experiment_time': total_time,
                'avg_processing_time_per_question': avg_processing_time,
                'output_directory': str(processor.output_dir),
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"✅ Experiment completed!")
            print(f"   📊 Success Rate: {success_rate:.2%} ({len(successful_results)}/{num_questions})")
            print(f"   🎯 Overall Accuracy: {accuracy_rate:.2%} ({len(correct_results)}/{num_questions})")
            print(f"   🎯 Accuracy Among Successful: {accuracy_among_successful:.2%} ({len(correct_results)}/{len(successful_results) if successful_results else 0})")
            print(f"   ⏱️  Total Time: {total_time:.1f}s")
            print(f"   ⏱️  Avg Time/Question: {avg_processing_time:.1f}s")
            print(f"   📁 Results saved in: {processor.output_dir}")
            
            return experiment_summary
            
        except Exception as e:
            end_time = time.time()
            total_time = end_time - start_time
            
            print(f"❌ Experiment failed: {e}")
            
            error_summary = {
                'max_plan_iterations': max_plan_iterations,
                'total_questions': 0,
                'successful_questions': 0,
                'failed_questions': 0,
                'correct_answers': 0,
                'success_rate': 0.0,
                'accuracy_rate': 0.0,
                'accuracy_among_successful': 0.0,
                'total_experiment_time': total_time,
                'avg_processing_time_per_question': 0.0,
                'output_directory': str(processor.output_dir),
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
            
            return error_summary
    
    async def run_full_ablation_study(self) -> Dict[str, Any]:
        """Run the complete ablation study with all max_plan_iterations values."""
        print(f"\n🎯 Starting MedXpertQA Ablation Study")
        print(f"📊 Testing max_plan_iterations values: {self.max_plan_iterations_range}")
        print(f"📁 Base output directory: {self.base_output_dir}")
        print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        study_start_time = time.time()
        
        # Run experiments for each max_plan_iterations value
        for max_plan_iterations in self.max_plan_iterations_range:
            experiment_result = await self.run_single_experiment(max_plan_iterations)
            self.ablation_results[max_plan_iterations] = experiment_result
        
        study_end_time = time.time()
        total_study_time = study_end_time - study_start_time
        
        # Create final study summary
        study_summary = {
            'study_metadata': {
                'timestamp': self.timestamp,
                'total_study_time': total_study_time,
                'dataset_name': self.dataset_name,
                'dataset_subset': self.dataset_subset,
                'dataset_split': self.dataset_split,
                'question_range': f"{self.start_idx}-{self.end_idx or 'end'}",
                'max_step_num': self.max_step_num,
                'max_plan_iterations_tested': self.max_plan_iterations_range,
                'enable_background_investigation': self.enable_background_investigation
            },
            'experiment_results': self.ablation_results
        }
        
        # Save comprehensive study results
        self.save_study_summary(study_summary)
        
        # Print final summary
        self.print_final_summary(study_summary)
        
        return study_summary
    
    def save_study_summary(self, study_summary: Dict[str, Any]):
        """Save the comprehensive study summary to files."""
        # Save JSON summary
        summary_file = self.base_output_dir / f"medxpert_ablation_study_summary_{self.timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(study_summary, f, indent=2, default=str)
        
        # Save CSV summary for easy analysis
        csv_file = self.base_output_dir / f"medxpert_ablation_study_comparison_{self.timestamp}.csv"
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write("max_plan_iterations,success_rate,accuracy_rate,accuracy_among_successful,avg_processing_time,total_experiment_time,successful_questions,correct_answers,failed_questions\n")
            
            for max_plan_iter, result in self.ablation_results.items():
                f.write(f"{max_plan_iter},{result.get('success_rate', 0):.4f},"
                       f"{result.get('accuracy_rate', 0):.4f},"
                       f"{result.get('accuracy_among_successful', 0):.4f},"
                       f"{result.get('avg_processing_time_per_question', 0):.2f},"
                       f"{result.get('total_experiment_time', 0):.2f},"
                       f"{result.get('successful_questions', 0)},"
                       f"{result.get('correct_answers', 0)},"
                       f"{result.get('failed_questions', 0)}\n")
        
        print(f"\n📄 Study summary saved:")
        print(f"   📋 JSON: {summary_file}")
        print(f"   📊 CSV:  {csv_file}")
    
    def print_final_summary(self, study_summary: Dict[str, Any]):
        """Print a formatted final summary of the ablation study."""
        print(f"\n{'='*80}")
        print(f"🎉 MEDXPERTQA ABLATION STUDY COMPLETED")
        print(f"{'='*80}")
        
        metadata = study_summary['study_metadata']
        
        print(f"\n📊 STUDY OVERVIEW")
        print(f"   Dataset: {metadata['dataset_name']} ({metadata['dataset_subset']}/{metadata['dataset_split']})")
        print(f"   Questions: {metadata['question_range']}")
        print(f"   Total Time: {metadata['total_study_time']:.1f}s")
        print(f"   Tested Values: {metadata['max_plan_iterations_tested']}")
        
        print(f"\n📈 RESULTS BY MAX_PLAN_ITERATIONS")
        print(f"   {'Value':<6} {'Success':<8} {'Accuracy':<9} {'Acc/Succ':<9} {'Time/Q':<8} {'Total Time':<10}")
        print(f"   {'-'*6} {'-'*8} {'-'*9} {'-'*9} {'-'*8} {'-'*10}")
        
        best_accuracy = 0
        best_accuracy_config = None
        
        for max_plan_iter in sorted(self.ablation_results.keys()):
            result = self.ablation_results[max_plan_iter]
            success_rate = result.get('success_rate', 0)
            accuracy_rate = result.get('accuracy_rate', 0)
            accuracy_among_successful = result.get('accuracy_among_successful', 0)
            avg_time = result.get('avg_processing_time_per_question', 0)
            total_time = result.get('total_experiment_time', 0)
            
            if accuracy_rate > best_accuracy:
                best_accuracy = accuracy_rate
                best_accuracy_config = max_plan_iter
            
            print(f"   {max_plan_iter:<6} {success_rate:<8.2%} {accuracy_rate:<9.2%} "
                  f"{accuracy_among_successful:<9.2%} {avg_time:<8.1f}s {total_time:<10.1f}s")
        
        print(f"\n🎯 BEST CONFIGURATION")
        if best_accuracy_config is not None:
            print(f"   Highest Overall Accuracy: max_plan_iterations = {best_accuracy_config} ({best_accuracy:.2%})")
        
        print(f"\n📁 All results saved in: {self.base_output_dir}")
        print(f"{'='*80}")


async def main():
    """Main function to run the MedXpertQA ablation study."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run ablation study on MedXpertQA text-only dataset with different max_plan_iterations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --sample-mode                    # Run on first 10 questions only
  %(prog)s --end-idx 50                     # Run on first 50 questions
  %(prog)s --debug --sample-mode            # Run in debug mode on sample
  %(prog)s --disable-background-investigation  # Disable background search
        """
    )
    
    parser.add_argument(
        "--dataset", 
        default="TsinghuaC3I/MedXpertQA",
        help="HuggingFace dataset name (default: TsinghuaC3I/MedXpertQA)"
    )
    parser.add_argument(
        "--subset", 
        default="Text",
        help="Dataset subset to process (default: Text for text-only questions)"
    )
    parser.add_argument(
        "--split", 
        default="test",
        help="Dataset split to process (default: test)"
    )
    parser.add_argument(
        "--output-dir", 
        default="outputs/ablation_study_medxpert",
        help="Base output directory for ablation study results"
    )
    parser.add_argument(
        "--start-idx", 
        type=int, 
        default=0,
        help="Starting index for processing questions (default: 0)"
    )
    parser.add_argument(
        "--end-idx", 
        type=int,
        help="Ending index for processing questions (default: None = all questions)"
    )
    parser.add_argument(
        "--max-step-num", 
        type=int, 
        default=3,
        help="Maximum steps per plan (kept constant across experiments, default: 3)"
    )
    parser.add_argument(
        "--disable-background-investigation", 
        action="store_true",
        help="Disable background investigation (default: enabled)"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug logging (default: disabled)"
    )
    parser.add_argument(
        "--sample-mode",
        action="store_true",
        help="Run on first 10 questions only for testing (default: disabled)"
    )
    
    args = parser.parse_args()
    
    # Adjust for sample mode
    if args.sample_mode:
        args.end_idx = 10
        print("🧪 Sample mode enabled: Processing first 10 questions only")
    elif args.end_idx is None:
        args.end_idx = 50  # Default to first 50 questions as requested
        print("📊 Processing first 50 questions (default)")
    
    # Create and run ablation study
    runner = MedXpertAblationStudyRunner(
        dataset_name=args.dataset,
        dataset_subset=args.subset,
        dataset_split=args.split,
        base_output_dir=args.output_dir,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        max_step_num=args.max_step_num,
        enable_background_investigation=not args.disable_background_investigation,
        debug=args.debug
    )
    
    try:
        study_results = await runner.run_full_ablation_study()
        print("\n✅ MedXpertQA ablation study completed successfully!")
        return study_results
        
    except Exception as e:
        logger.error(f"MedXpertQA ablation study failed: {e}")
        print(f"\n❌ MedXpertQA ablation study failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 