#!/usr/bin/env python3
"""
Ablation Study Script for MedBrowse Dataset Processing.

This script runs the multi-agent pipeline on the MedBrowse dataset with different
max_plan_iterations values (1 to 5) to study how the model improves with more
steps of research planning.

Usage:
    python ablation_study_medbrowse.py [options]
    
Examples:
    # Run on first 5 questions (sample mode)
    python ablation_study_medbrowse.py --sample-mode
    
    # Run on all 50 questions
    python ablation_study_medbrowse.py
    
    # Run on questions 0-10 only
    python ablation_study_medbrowse.py --start-idx 0 --end-idx 10
    
    # Run with debug mode
    python ablation_study_medbrowse.py --debug --sample-mode
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Add the current directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent))

from process_medbrowse_questions import MedBrowseProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class AblationStudyRunner:
    """Runner for the ablation study with different max_plan_iterations values."""
    
    def __init__(
        self,
        dataset_name: str = "AIM-Harvard/MedBrowseComp",
        dataset_split: str = "MedBrowseComp_50",
        base_output_dir: str = "outputs/ablation_study",
        start_idx: int = 0,
        end_idx: int = None,
        max_step_num: int = 3,
        enable_background_investigation: bool = True,
        debug: bool = False,
        enable_fallback: bool = True,
        fallback_timeout: int = 300,
        max_retries: int = 5
    ):
        """Initialize the ablation study runner.
        
        Args:
            dataset_name: HuggingFace dataset name
            dataset_split: Dataset split to process
            base_output_dir: Base directory for all ablation study outputs
            start_idx: Starting index for processing questions
            end_idx: Ending index for processing questions (None = all)
            max_step_num: Maximum number of steps in a plan (kept constant)
            enable_background_investigation: Whether to enable background investigation
            debug: Whether to enable debug logging
            enable_fallback: Whether to enable fallback mechanisms
            fallback_timeout: Timeout in seconds for fallback processing
            max_retries: Maximum number of retries for parsing errors
        """
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.base_output_dir = Path(base_output_dir)
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.max_step_num = max_step_num
        self.enable_background_investigation = enable_background_investigation
        self.debug = debug
        self.enable_fallback = enable_fallback
        self.fallback_timeout = fallback_timeout
        self.max_retries = max_retries
        
        # Create base output directory
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp for this ablation study
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Store results for comparison
        self.ablation_results = {}
        
        # Range of max_plan_iterations to test (1 to 5)
        self.max_plan_iterations_range = list(range(1, 6))
        
    def create_processor(self, max_plan_iterations: int) -> MedBrowseProcessor:
        """Create a MedBrowseProcessor with specific max_plan_iterations.
        
        Args:
            max_plan_iterations: Number of planning iterations to use
            
        Returns:
            Configured MedBrowseProcessor instance
        """
        output_dir = self.base_output_dir / f"max_plan_iterations_{max_plan_iterations}"
        
        return MedBrowseProcessor(
            output_dir=str(output_dir),
            max_plan_iterations=max_plan_iterations,
            max_step_num=self.max_step_num,
            enable_background_investigation=self.enable_background_investigation,
            debug=self.debug,
            enable_fallback=self.enable_fallback,
            fallback_timeout=self.fallback_timeout,
            max_retries=self.max_retries
        )
    
    async def run_single_experiment(self, max_plan_iterations: int) -> Dict[str, Any]:
        """Run the experiment with a specific max_plan_iterations value.
        
        Args:
            max_plan_iterations: Number of planning iterations to use
            
        Returns:
            Dictionary containing experiment results and metadata
        """
        print(f"\n{'='*60}")
        print(f"🧪 Running Experiment: max_plan_iterations = {max_plan_iterations}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Create processor for this experiment
        processor = self.create_processor(max_plan_iterations)
        
        try:
            # Load dataset (only once, reuse for all experiments)
            print(f"📥 Loading dataset: {self.dataset_name} ({self.dataset_split})")
            df = processor.load_dataset(self.dataset_name, self.dataset_split)
            
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
            
            # Calculate success metrics
            successful_results = [r for r in results if r.get('status') == 'success']
            error_results = [r for r in results if r.get('status') == 'error']
            
            success_rate = len(successful_results) / len(results) if results else 0
            avg_processing_time = sum(r.get('processing_time', 0) for r in results) / len(results) if results else 0
            
            experiment_summary = {
                'max_plan_iterations': max_plan_iterations,
                'total_questions': num_questions,
                'successful_questions': len(successful_results),
                'failed_questions': len(error_results),
                'success_rate': success_rate,
                'total_experiment_time': total_time,
                'avg_processing_time_per_question': avg_processing_time,
                'output_directory': str(processor.output_dir),
                'timestamp': datetime.now().isoformat(),
                'results_summary': {
                    'successful_answers': [r.get('final_answer', '') for r in successful_results],
                    'error_messages': [r.get('error', '') for r in error_results]
                }
            }
            
            print(f"✅ Experiment completed!")
            print(f"   📊 Success Rate: {success_rate:.2%} ({len(successful_results)}/{num_questions})")
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
                'success_rate': 0.0,
                'total_experiment_time': total_time,
                'avg_processing_time_per_question': 0.0,
                'output_directory': str(processor.output_dir),
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'results_summary': {'successful_answers': [], 'error_messages': [str(e)]}
            }
            
            return error_summary
    
    async def run_full_ablation_study(self) -> Dict[str, Any]:
        """Run the complete ablation study with all max_plan_iterations values.
        
        Returns:
            Dictionary containing all experiment results and comparative analysis
        """
        print(f"\n🎯 Starting Ablation Study: MedBrowse Dataset")
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
        
        # Generate comparative analysis
        comparative_analysis = self.generate_comparative_analysis()
        
        # Create final study summary
        study_summary = {
            'study_metadata': {
                'timestamp': self.timestamp,
                'total_study_time': total_study_time,
                'dataset_name': self.dataset_name,
                'dataset_split': self.dataset_split,
                'question_range': f"{self.start_idx}-{self.end_idx or 'end'}",
                'max_step_num': self.max_step_num,
                'max_plan_iterations_tested': self.max_plan_iterations_range,
                'enable_background_investigation': self.enable_background_investigation
            },
            'experiment_results': self.ablation_results,
            'comparative_analysis': comparative_analysis
        }
        
        # Save comprehensive study results
        self.save_study_summary(study_summary)
        
        # Print final summary
        self.print_final_summary(study_summary)
        
        return study_summary
    
    def generate_comparative_analysis(self) -> Dict[str, Any]:
        """Generate comparative analysis of results across different max_plan_iterations.
        
        Returns:
            Dictionary containing comparative metrics and insights
        """
        if not self.ablation_results:
            return {}
        
        # Extract key metrics for comparison
        success_rates = {}
        processing_times = {}
        total_times = {}
        
        for max_plan_iter, result in self.ablation_results.items():
            success_rates[max_plan_iter] = result.get('success_rate', 0)
            processing_times[max_plan_iter] = result.get('avg_processing_time_per_question', 0)
            total_times[max_plan_iter] = result.get('total_experiment_time', 0)
        
        # Find best performing configurations
        best_success_rate = max(success_rates.items(), key=lambda x: x[1]) if success_rates else (None, 0)
        fastest_processing = min(processing_times.items(), key=lambda x: x[1]) if processing_times else (None, 0)
        
        # Calculate trends
        success_rate_trend = self.calculate_trend(success_rates)
        time_trend = self.calculate_trend(processing_times)
        
        return {
            'success_rates_by_max_plan_iterations': success_rates,
            'processing_times_by_max_plan_iterations': processing_times,
            'total_times_by_max_plan_iterations': total_times,
            'best_success_rate': {
                'max_plan_iterations': best_success_rate[0],
                'success_rate': best_success_rate[1]
            },
            'fastest_processing': {
                'max_plan_iterations': fastest_processing[0],
                'avg_time_per_question': fastest_processing[1]
            },
            'trends': {
                'success_rate_trend': success_rate_trend,
                'processing_time_trend': time_trend
            },
            'recommendations': self.generate_recommendations(success_rates, processing_times)
        }
    
    def calculate_trend(self, metrics: Dict[int, float]) -> str:
        """Calculate the trend direction for a metric across max_plan_iterations values.
        
        Args:
            metrics: Dictionary mapping max_plan_iterations to metric values
            
        Returns:
            String describing the trend ('increasing', 'decreasing', 'mixed', 'stable')
        """
        if len(metrics) < 2:
            return 'insufficient_data'
        
        sorted_items = sorted(metrics.items())
        values = [item[1] for item in sorted_items]
        
        # Simple trend analysis
        increases = sum(1 for i in range(1, len(values)) if values[i] > values[i-1])
        decreases = sum(1 for i in range(1, len(values)) if values[i] < values[i-1])
        
        if increases > decreases:
            return 'increasing'
        elif decreases > increases:
            return 'decreasing'
        elif increases == decreases and increases > 0:
            return 'mixed'
        else:
            return 'stable'
    
    def generate_recommendations(self, success_rates: Dict[int, float], processing_times: Dict[int, float]) -> List[str]:
        """Generate recommendations based on the ablation study results.
        
        Args:
            success_rates: Success rates by max_plan_iterations
            processing_times: Processing times by max_plan_iterations
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if not success_rates or not processing_times:
            return ["Insufficient data for recommendations"]
        
        # Find optimal configurations
        best_success = max(success_rates.items(), key=lambda x: x[1])
        fastest_time = min(processing_times.items(), key=lambda x: x[1])
        
        # Calculate efficiency scores (success rate / normalized processing time)
        max_time = max(processing_times.values()) if processing_times.values() else 1
        efficiency_scores = {}
        for max_plan_iter in success_rates.keys():
            if max_plan_iter in processing_times:
                normalized_time = processing_times[max_plan_iter] / max_time if max_time > 0 else 1
                efficiency_scores[max_plan_iter] = success_rates[max_plan_iter] / (normalized_time + 0.1)
        
        best_efficiency = max(efficiency_scores.items(), key=lambda x: x[1]) if efficiency_scores else (None, 0)
        
        # Generate specific recommendations
        recommendations.append(f"For highest success rate ({best_success[1]:.2%}): Use max_plan_iterations = {best_success[0]}")
        recommendations.append(f"For fastest processing ({fastest_time[1]:.1f}s avg): Use max_plan_iterations = {fastest_time[0]}")
        
        if best_efficiency[0] is not None:
            recommendations.append(f"For best efficiency balance: Use max_plan_iterations = {best_efficiency[0]}")
        
        # Trend-based recommendations
        success_trend = self.calculate_trend(success_rates)
        time_trend = self.calculate_trend(processing_times)
        
        if success_trend == 'increasing':
            recommendations.append("Success rate improves with more planning iterations - consider higher values for better quality")
        elif success_trend == 'decreasing':
            recommendations.append("Success rate decreases with more planning iterations - lower values may be more reliable")
        
        if time_trend == 'increasing':
            recommendations.append("Processing time increases with more planning iterations - balance quality vs speed needs")
        
        return recommendations
    
    def save_study_summary(self, study_summary: Dict[str, Any]):
        """Save the comprehensive study summary to files.
        
        Args:
            study_summary: Complete study results and analysis
        """
        # Save JSON summary
        summary_file = self.base_output_dir / f"ablation_study_summary_{self.timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(study_summary, f, indent=2, default=str)
        
        # Save CSV summary for easy analysis
        csv_file = self.base_output_dir / f"ablation_study_comparison_{self.timestamp}.csv"
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write("max_plan_iterations,success_rate,avg_processing_time,total_experiment_time,successful_questions,failed_questions\n")
            
            for max_plan_iter, result in self.ablation_results.items():
                f.write(f"{max_plan_iter},{result.get('success_rate', 0):.4f},"
                       f"{result.get('avg_processing_time_per_question', 0):.2f},"
                       f"{result.get('total_experiment_time', 0):.2f},"
                       f"{result.get('successful_questions', 0)},"
                       f"{result.get('failed_questions', 0)}\n")
        
        print(f"\n📄 Study summary saved:")
        print(f"   📋 JSON: {summary_file}")
        print(f"   📊 CSV:  {csv_file}")
    
    def print_final_summary(self, study_summary: Dict[str, Any]):
        """Print a formatted final summary of the ablation study.
        
        Args:
            study_summary: Complete study results and analysis
        """
        print(f"\n{'='*80}")
        print(f"🎉 ABLATION STUDY COMPLETED")
        print(f"{'='*80}")
        
        metadata = study_summary['study_metadata']
        analysis = study_summary['comparative_analysis']
        
        print(f"\n📊 STUDY OVERVIEW")
        print(f"   Dataset: {metadata['dataset_name']} ({metadata['dataset_split']})")
        print(f"   Questions: {metadata['question_range']}")
        print(f"   Total Time: {metadata['total_study_time']:.1f}s")
        print(f"   Tested Values: {metadata['max_plan_iterations_tested']}")
        
        print(f"\n📈 RESULTS BY MAX_PLAN_ITERATIONS")
        print(f"   {'Value':<6} {'Success Rate':<12} {'Avg Time/Q':<11} {'Total Time':<11}")
        print(f"   {'-'*6} {'-'*12} {'-'*11} {'-'*11}")
        
        for max_plan_iter in sorted(self.ablation_results.keys()):
            result = self.ablation_results[max_plan_iter]
            success_rate = result.get('success_rate', 0)
            avg_time = result.get('avg_processing_time_per_question', 0)
            total_time = result.get('total_experiment_time', 0)
            
            print(f"   {max_plan_iter:<6} {success_rate:<12.2%} {avg_time:<11.1f}s {total_time:<11.1f}s")
        
        if analysis.get('recommendations'):
            print(f"\n💡 RECOMMENDATIONS")
            for i, rec in enumerate(analysis['recommendations'], 1):
                print(f"   {i}. {rec}")
        
        print(f"\n🎯 BEST CONFIGURATIONS")
        if 'best_success_rate' in analysis:
            best_success = analysis['best_success_rate']
            print(f"   Highest Success Rate: max_plan_iterations = {best_success['max_plan_iterations']} ({best_success['success_rate']:.2%})")
        
        if 'fastest_processing' in analysis:
            fastest = analysis['fastest_processing']
            print(f"   Fastest Processing:   max_plan_iterations = {fastest['max_plan_iterations']} ({fastest['avg_time_per_question']:.1f}s avg)")
        
        print(f"\n📁 All results saved in: {self.base_output_dir}")
        print(f"{'='*80}")


async def main():
    """Main function to run the ablation study."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run ablation study on MedBrowse dataset with different max_plan_iterations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --sample-mode                    # Run on first 5 questions only
  %(prog)s --start-idx 0 --end-idx 10       # Run on questions 0-9
  %(prog)s --debug --sample-mode            # Run in debug mode on sample
  %(prog)s --disable-background-investigation  # Disable background search
        """
    )
    
    parser.add_argument(
        "--dataset", 
        default="AIM-Harvard/MedBrowseComp",
        help="HuggingFace dataset name (default: AIM-Harvard/MedBrowseComp)"
    )
    parser.add_argument(
        "--split", 
        default="MedBrowseComp_50",
        help="Dataset split to process (default: MedBrowseComp_50)"
    )
    parser.add_argument(
        "--output-dir", 
        default="outputs/ablation_study",
        help="Base output directory for ablation study results (default: outputs/ablation_study)"
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
        "--disable-fallback",
        action="store_true",
        help="Disable fallback processing (default: enabled)"
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
    parser.add_argument(
        "--sample-mode",
        action="store_true",
        help="Run on first 5 questions only for testing (default: disabled)"
    )
    
    args = parser.parse_args()
    
    # Adjust for sample mode
    if args.sample_mode:
        args.end_idx = 5
        print("🧪 Sample mode enabled: Processing first 5 questions only")
    
    # Create and run ablation study
    runner = AblationStudyRunner(
        dataset_name=args.dataset,
        dataset_split=args.split,
        base_output_dir=args.output_dir,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        max_step_num=args.max_step_num,
        enable_background_investigation=not args.disable_background_investigation,
        debug=args.debug,
        enable_fallback=not args.disable_fallback,
        fallback_timeout=args.fallback_timeout,
        max_retries=args.max_retries
    )
    
    try:
        study_results = await runner.run_full_ablation_study()
        print("\n✅ Ablation study completed successfully!")
        return study_results
        
    except Exception as e:
        logger.error(f"Ablation study failed: {e}")
        print(f"\n❌ Ablation study failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 