#!/usr/bin/env python3
"""
Example script demonstrating how to run the MedBrowse dataset processor.
"""

import asyncio
import sys
from pathlib import Path

# Add the current directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent))

from process_medbrowse_questions import MedBrowseProcessor


async def run_sample_processing():
    """Run a sample processing of the first 3 questions from the dataset."""
    
    print("🚀 Starting MedBrowse Dataset Processing Example")
    print("=" * 50)
    
    # Create processor with sample configuration and enhanced error handling
    processor = MedBrowseProcessor(
        output_dir="outputs/medbrowse_sample",
        max_plan_iterations=2,  # Set to 2 as requested  
        max_step_num=3,         # Always set to 3 as requested
        enable_background_investigation=False,  # Disabled for faster processing
        debug=False,
        enable_fallback=True,   # Enable fallback for robustness
        fallback_timeout=180,   # 3 minute timeout for fallback
        max_retries=5           # Try up to 5 times for parsing errors (increased)
    )
    
    try:
        # Load dataset
        print("📥 Loading MedBrowseComp dataset...")
        df = processor.load_dataset("AIM-Harvard/MedBrowseComp", "MedBrowseComp_50")
        
        print(f"✅ Successfully loaded {len(df)} questions")
        print(f"📋 Dataset columns: {list(df.columns)}")
        
        if len(df) > 0:
            print(f"📄 Sample question data structure:")
            sample_data = df.iloc[0].to_dict()
            for key, value in sample_data.items():
                if isinstance(value, str):
                    print(f"  {key}: {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")
                else:
                    print(f"  {key}: {value}")
        
        # Process only the first 3 questions as an example
        print(f"\n🔄 Processing first 3 questions as example...")
        results = await processor.process_all_questions(df, start_idx=0, end_idx=3)
        
        # Save results
        print("💾 Saving results...")
        processor.save_results(results)
        
        print("✅ Example processing completed successfully!")
        print(f"📁 Results saved in: {processor.output_dir}")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        raise


async def run_full_processing():
    """Run processing of all questions in the dataset."""
    
    print("🚀 Starting Full MedBrowse Dataset Processing")
    print("=" * 50)
    
    # Create processor for full processing
    processor = MedBrowseProcessor(
        output_dir="outputs/medbrowse_full",
        max_plan_iterations=1,
        max_step_num=3,
        enable_background_investigation=True,
        debug=False
    )
    
    try:
        # Load dataset
        print("📥 Loading MedBrowseComp dataset...")
        df = processor.load_dataset("AIM-Harvard/MedBrowseComp", "MedBrowseComp_50")
        
        print(f"✅ Successfully loaded {len(df)} questions")
        
        # Process all questions
        print(f"🔄 Processing all {len(df)} questions...")
        print("⚠️  This may take a while depending on your pipeline performance...")
        
        results = await processor.process_all_questions(df)
        
        # Save results
        print("💾 Saving final results...")
        processor.save_results(results)
        
        print("✅ Full processing completed successfully!")
        print(f"📁 Results saved in: {processor.output_dir}")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        raise


async def main():
    """Main function with options for different processing modes."""
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        print("Usage:")
        print("  python run_medbrowse_example.py sample   # Process first 3 questions")
        print("  python run_medbrowse_example.py full     # Process all questions")
        print("")
        print("Which mode would you like to run?")
        print("1. Sample (first 3 questions)")
        print("2. Full processing (all questions)")
        
        choice = input("Enter choice (1 or 2): ").strip()
        mode = "sample" if choice == "1" else "full"
    
    if mode == "sample":
        await run_sample_processing()
    elif mode == "full":
        await run_full_processing()
    else:
        print(f"Unknown mode: {mode}")
        print("Use 'sample' or 'full'")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())