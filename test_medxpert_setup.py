#!/usr/bin/env python3
"""
Test script to verify MedXpertQA ablation study setup.

This script tests:
1. Dataset loading from HuggingFace
2. Question formatting
3. Answer extraction patterns
4. Basic ablation runner creation

Usage:
    python test_medxpert_setup.py
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test that all required imports work."""
    print("🔄 Testing imports...")
    
    try:
        import pandas as pd
        from datasets import load_dataset
        import asyncio
        import json
        import re
        from datetime import datetime
        from typing import Dict, List, Any, Optional
        print("✅ Standard library imports successful")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    try:
        from src.workflow import run_agent_workflow_async
        print("✅ Local workflow import successful")
    except ImportError as e:
        print(f"⚠️  Warning: Could not import workflow module: {e}")
        print("   This is expected if running setup test before full environment is ready")
    
    return True


def test_dataset_loading():
    """Test loading a small sample from MedXpertQA dataset."""
    print("\n🔄 Testing dataset loading...")
    
    try:
        from datasets import load_dataset
        import pandas as pd
        
        # Load just a few samples for testing
        print("   Loading MedXpertQA Text subset...")
        dataset = load_dataset("TsinghuaC3I/MedXpertQA", "Text", split="test")
        
        # Convert to DataFrame
        df = pd.DataFrame(dataset)
        
        print(f"✅ Dataset loaded successfully: {len(df)} questions")
        print(f"   Columns: {list(df.columns)}")
        
        # Show sample question
        if len(df) > 0:
            sample = df.iloc[0]
            print(f"\n📋 Sample Question:")
            print(f"   ID: {sample.get('id', 'N/A')}")
            print(f"   Question: {str(sample.get('question', ''))[:100]}...")
            print(f"   Options: {sample.get('options', {})}")
            print(f"   Correct Answer: {sample.get('label', 'N/A')}")
            print(f"   Medical Task: {sample.get('medical_task', 'N/A')}")
            print(f"   Body System: {sample.get('body_system', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return False


def test_question_formatting():
    """Test question formatting function."""
    print("\n🔄 Testing question formatting...")
    
    try:
        # Import the processor class
        from ablation_study_medxpert import MedXpertProcessor
        
        # Create a sample question
        sample_question = {
            'id': 'test-1',
            'question': 'What is the most common cause of chest pain in young adults?',
            'options': {
                'A': 'Myocardial infarction',
                'B': 'Pulmonary embolism', 
                'C': 'Musculoskeletal pain',
                'D': 'Gastroesophageal reflux',
                'E': 'Pneumonia'
            },
            'label': 'C',
            'medical_task': 'Diagnosis',
            'body_system': 'Cardiovascular',
            'question_type': 'Reasoning'
        }
        
        # Create processor and format question
        processor = MedXpertProcessor()
        formatted = processor.format_question_for_pipeline(sample_question)
        
        print("✅ Question formatting successful")
        print(f"📋 Formatted Question:")
        print(formatted[:300] + "..." if len(formatted) > 300 else formatted)
        
        return True
        
    except Exception as e:
        print(f"❌ Question formatting failed: {e}")
        return False


def test_answer_extraction():
    """Test answer extraction patterns."""
    print("\n🔄 Testing answer extraction...")
    
    try:
        from ablation_study_medxpert import MedXpertProcessor
        
        processor = MedXpertProcessor()
        
        # Test different answer formats
        test_outputs = [
            "The answer is A. This is because...",
            "Answer: B",
            "(C) is correct because...",
            "I choose option D.",
            "The correct choice is E",
            "A",
            "Looking at the options, B) would be the best choice.",
            "Final answer: C"
        ]
        
        expected_answers = ['A', 'B', 'C', 'D', 'E', 'A', 'B', 'C']
        
        correct_extractions = 0
        
        for i, output in enumerate(test_outputs):
            extracted = processor._extract_answer_choice(output)
            expected = expected_answers[i]
            
            if extracted == expected:
                correct_extractions += 1
                print(f"   ✅ '{output[:30]}...' → {extracted}")
            else:
                print(f"   ❌ '{output[:30]}...' → {extracted} (expected {expected})")
        
        success_rate = correct_extractions / len(test_outputs)
        print(f"\n✅ Answer extraction test: {correct_extractions}/{len(test_outputs)} correct ({success_rate:.1%})")
        
        return success_rate >= 0.7  # At least 70% success rate
        
    except Exception as e:
        print(f"❌ Answer extraction test failed: {e}")
        return False


def test_ablation_runner_creation():
    """Test creating the ablation study runner."""
    print("\n🔄 Testing ablation runner creation...")
    
    try:
        from ablation_study_medxpert import MedXpertAblationStudyRunner
        
        # Create runner with test parameters
        runner = MedXpertAblationStudyRunner(
            base_output_dir="test_output",
            start_idx=0,
            end_idx=2,  # Very small test
            debug=True
        )
        
        print(f"✅ Ablation runner created successfully")
        print(f"   Base output dir: {runner.base_output_dir}")
        print(f"   Max plan iterations range: {runner.max_plan_iterations_range}")
        print(f"   Dataset: {runner.dataset_name} ({runner.dataset_subset})")
        
        # Test processor creation
        processor = runner.create_processor(max_plan_iterations=2)
        print(f"✅ Processor creation successful")
        print(f"   Output dir: {processor.output_dir}")
        print(f"   Max plan iterations: {processor.max_plan_iterations}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ablation runner creation failed: {e}")
        return False


def main():
    """Run all setup tests."""
    print("🧪 MedXpertQA Ablation Study Setup Test")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Dataset Loading", test_dataset_loading),
        ("Question Formatting", test_question_formatting),
        ("Answer Extraction", test_answer_extraction),
        ("Ablation Runner Creation", test_ablation_runner_creation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} test failed")
        except Exception as e:
            print(f"❌ {test_name} test error: {e}")
    
    print(f"\n{'=' * 50}")
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! MedXpertQA ablation study setup is ready.")
        print("\n🚀 You can now run:")
        print("   python ablation_study_medxpert.py --sample-mode")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        print("\n🔧 Common fixes:")
        print("   - Install missing packages: pip install datasets pandas")
        print("   - Check internet connection for dataset download")
        print("   - Ensure src.workflow module is available")
    
    return passed == total


if __name__ == "__main__":
    main() 