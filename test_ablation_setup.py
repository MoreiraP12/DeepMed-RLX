#!/usr/bin/env python3
"""
Test script to verify the ablation study setup is working correctly.
This runs a minimal test before running the full study.
"""

import asyncio
import sys
from pathlib import Path

# Add the current directory to the path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test that all required modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        from process_medbrowse_questions import MedBrowseProcessor
        print("✅ MedBrowseProcessor imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import MedBrowseProcessor: {e}")
        return False
    
    try:
        from ablation_study_medbrowse import AblationStudyRunner
        print("✅ AblationStudyRunner imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import AblationStudyRunner: {e}")
        return False
    
    return True

async def test_dataset_loading():
    """Test that the dataset can be loaded."""
    print("\n📥 Testing dataset loading...")
    
    try:
        from process_medbrowse_questions import MedBrowseProcessor
        
        processor = MedBrowseProcessor(
            output_dir="test_outputs",
            max_plan_iterations=1,
            max_step_num=3,
            debug=False
        )
        
        # Try to load dataset
        df = processor.load_dataset("AIM-Harvard/MedBrowseComp", "MedBrowseComp_50")
        print(f"✅ Dataset loaded successfully: {len(df)} questions")
        
        # Show sample question
        if len(df) > 0:
            sample_question = df.iloc[0]
            print(f"📄 Sample question available")
            
        return True
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return False

async def test_workflow_basic():
    """Test that the basic workflow can run."""
    print("\n🔄 Testing basic workflow...")
    
    try:
        from src.workflow import run_agent_workflow_async
        
        # Test with a very simple question and minimal settings
        test_question = "What is aspirin?"
        
        print(f"🧪 Testing with question: {test_question}")
        
        result = await run_agent_workflow_async(
            user_input=test_question,
            debug=False,
            max_plan_iterations=1,
            max_step_num=1,
            enable_background_investigation=False
        )
        
        print("✅ Basic workflow test completed")
        return True
        
    except Exception as e:
        print(f"❌ Basic workflow test failed: {e}")
        print(f"💡 This might be due to missing API keys or configuration issues")
        return False

async def test_ablation_runner_creation():
    """Test that the ablation runner can be created."""
    print("\n🏗️  Testing ablation runner creation...")
    
    try:
        from ablation_study_medbrowse import AblationStudyRunner
        
        runner = AblationStudyRunner(
            base_output_dir="test_outputs/ablation_test",
            start_idx=0,
            end_idx=2,  # Just 2 questions for testing
            debug=False
        )
        
        print("✅ AblationStudyRunner created successfully")
        print(f"   📁 Output directory: {runner.base_output_dir}")
        print(f"   📊 Testing range: {runner.max_plan_iterations_range}")
        
        return True
        
    except Exception as e:
        print(f"❌ AblationStudyRunner creation failed: {e}")
        return False

def clean_test_outputs():
    """Clean up test output directories."""
    print("\n🧹 Cleaning up test outputs...")
    
    import shutil
    
    test_dirs = ["test_outputs"]
    
    for test_dir in test_dirs:
        test_path = Path(test_dir)
        if test_path.exists():
            shutil.rmtree(test_path)
            print(f"✅ Cleaned up {test_dir}")

async def main():
    """Run all tests."""
    print("🧪 ABLATION STUDY SETUP TEST")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Dataset Loading Test", test_dataset_loading),
        ("Basic Workflow Test", test_workflow_basic),
        ("Ablation Runner Test", test_ablation_runner_creation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            if asyncio.iscoroutinefunction(test_func):
                success = await test_func()
            else:
                success = test_func()
            
            if success:
                passed += 1
            
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    # Clean up
    clean_test_outputs()
    
    print(f"\n{'=' * 50}")
    print(f"🎯 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! You're ready to run the ablation study.")
        print("\n🚀 Next steps:")
        print("   1. Run sample test: python ablation_study_medbrowse.py --sample-mode")
        print("   2. Run full study: python ablation_study_medbrowse.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\n🔧 Common fixes:")
        print("   1. Install missing dependencies: pip install -r requirements_medbrowse.txt")
        print("   2. Check API keys in your environment")
        print("   3. Verify your workflow configuration")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 