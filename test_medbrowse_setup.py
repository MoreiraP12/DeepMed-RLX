#!/usr/bin/env python3
"""
Test script to validate MedBrowse setup and dependencies.
"""

import sys
import asyncio
from pathlib import Path

def test_imports():
    """Test if all required imports are available."""
    print("🧪 Testing imports...")
    
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
    except ImportError as e:
        print(f"❌ pandas import failed: {e}")
        return False
    
    try:
        from datasets import load_dataset
        print("✅ datasets imported successfully")
    except ImportError as e:
        print(f"❌ datasets import failed: {e}")
        print("💡 Install with: pip install datasets>=2.0.0")
        return False
    
    try:
        import huggingface_hub
        print("✅ huggingface_hub imported successfully")
    except ImportError as e:
        print(f"❌ huggingface_hub import failed: {e}")
        print("💡 Install with: pip install huggingface_hub>=0.20.0")
        return False
    
    try:
        from src.workflow import run_agent_workflow_async
        print("✅ workflow imported successfully")
    except ImportError as e:
        print(f"❌ workflow import failed: {e}")
        print("💡 Make sure you're in the correct directory and src/ is available")
        return False
    
    return True


def test_dataset_access():
    """Test if we can access the MedBrowseComp dataset."""
    print("\n🔍 Testing dataset access...")
    
    try:
        from datasets import load_dataset
        
        print("📥 Attempting to load MedBrowseComp dataset...")
        dataset = load_dataset("AIM-Harvard/MedBrowseComp", split="MedBrowseComp_50")
        df = dataset.to_pandas()
        
        print(f"✅ Successfully loaded dataset with {len(df)} rows")
        print(f"📋 Dataset columns: {list(df.columns)}")
        
        if len(df) > 0:
            print(f"📄 Sample data from first row:")
            sample = df.iloc[0].to_dict()
            for key, value in sample.items():
                if isinstance(value, str):
                    preview = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                    print(f"  {key}: {preview}")
                else:
                    print(f"  {key}: {value}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        print("💡 Check your internet connection and HuggingFace access")
        return False


async def test_pipeline():
    """Test if the pipeline works with a simple question."""
    print("\n🚀 Testing pipeline with simple question...")
    
    try:
        from src.workflow import run_agent_workflow_async
        
        test_question = "What is aspirin used for?"
        print(f"🔄 Testing with question: {test_question}")
        
        # Try to run the workflow
        result = await run_agent_workflow_async(
            user_input=test_question,
            debug=False,
            max_plan_iterations=1,
            max_step_num=1,  # Minimal steps for testing
            enable_background_investigation=False  # Faster for testing
        )
        
        print("✅ Pipeline test completed successfully")
        return True
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        print("💡 Check your pipeline configuration and API keys")
        return False


def test_file_structure():
    """Test if all required files are present."""
    print("\n📁 Testing file structure...")
    
    required_files = [
        "process_medbrowse_questions.py",
        "run_medbrowse_example.py", 
        "requirements_medbrowse.txt",
        "MEDBROWSE_README.md"
    ]
    
    all_present = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path} found")
        else:
            print(f"❌ {file_path} missing")
            all_present = False
    
    # Check if outputs directory can be created
    try:
        outputs_dir = Path("outputs")
        outputs_dir.mkdir(exist_ok=True)
        print("✅ outputs directory available")
    except Exception as e:
        print(f"❌ Cannot create outputs directory: {e}")
        all_present = False
    
    return all_present


async def main():
    """Run all tests."""
    print("🧪 MedBrowse Setup Validation")
    print("=" * 40)
    
    tests = [
        ("Import Test", test_imports),
        ("File Structure Test", test_file_structure),
        ("Dataset Access Test", test_dataset_access),
        ("Pipeline Test", test_pipeline)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*20} SUMMARY {'='*20}")
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed! You're ready to process the MedBrowse dataset.")
        print("\n🚀 Try running:")
        print("   python run_medbrowse_example.py sample")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed. Please fix the issues above before proceeding.")
        print("\n💡 Common fixes:")
        print("   - pip install -r requirements_medbrowse.txt")
        print("   - Check your API keys and configuration")
        print("   - Ensure you're in the correct directory")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 