# MedBrowse Dataset Processor

This set of scripts allows you to process questions from the [AIM-Harvard/MedBrowseComp](https://huggingface.co/datasets/AIM-Harvard/MedBrowseComp) dataset through your multi-agent pipeline and store the results.

## Overview

The MedBrowseComp dataset contains medical questions that can be processed through your existing DeepMed-RLX multi-agent workflow. This processor will:

1. 📥 Load questions from the HuggingFace dataset
2. 🔄 Process each question through your multi-agent pipeline
3. 💾 Store results in multiple formats (JSON, CSV)
4. 📊 Generate summary reports

## Files Created

- `process_medbrowse_questions.py` - Main processing script with full functionality
- `run_medbrowse_example.py` - Simple example script for testing
- `requirements_medbrowse.txt` - Additional dependencies needed
- `MEDBROWSE_README.md` - This documentation

## Setup

### 1. Install Dependencies

First, install the additional required packages:

```bash
pip install -r requirements_medbrowse.txt
```

Or install manually:

```bash
pip install datasets>=2.0.0 huggingface_hub>=0.20.0
```

### 2. Verify Your Environment

Make sure your existing pipeline works by testing with a simple question:

```bash
python main.py "What is the treatment for hypertension?"
```

## Usage

### Quick Start - Example Script

For a quick test with the first 3 questions:

```bash
# Interactive mode
python run_medbrowse_example.py

# Or directly specify sample mode
python run_medbrowse_example.py sample
```

For processing all questions:

```bash
python run_medbrowse_example.py full
```

### Advanced Usage - Main Script

The main script provides more control over the processing:

```bash
# Basic usage - process all questions from MedBrowseComp_50
python process_medbrowse_questions.py

# Process only questions 0-9 (first 10 questions)
python process_medbrowse_questions.py --start-idx 0 --end-idx 10

# Use different dataset split
python process_medbrowse_questions.py --split MedBrowseComp_605

# Custom output directory
python process_medbrowse_questions.py --output-dir my_results

# Enable debug mode
python process_medbrowse_questions.py --debug

# Disable background investigation for faster processing
python process_medbrowse_questions.py --disable-background-investigation

# Custom pipeline parameters
python process_medbrowse_questions.py --max-plan-iterations 2 --max-step-num 5
```

### Available Command Line Options

```bash
--dataset              # HuggingFace dataset name (default: "AIM-Harvard/MedBrowseComp")
--split                # Dataset split to process (default: "MedBrowseComp_50")
--output-dir           # Output directory for results (default: "outputs")
--start-idx            # Starting index for processing (default: 0)
--end-idx              # Ending index for processing (default: all)
--max-plan-iterations  # Maximum plan iterations per question (default: 1)
--max-step-num         # Maximum steps per plan (default: 3)
--disable-background-investigation  # Disable background investigation
--debug                # Enable debug logging
```

## Output Files

The processor creates several output files with timestamps:

### JSON Results
- `medbrowse_results_YYYYMMDD_HHMMSS.json` - Complete results with all data
- `medbrowse_results_intermediate_YYYYMMDD_HHMMSS.json` - Intermediate saves (every 5 questions)

### CSV Results  
- `medbrowse_results_YYYYMMDD_HHMMSS.csv` - Simplified tabular format

### Summary Report
- `medbrowse_summary_YYYYMMDD_HHMMSS.json` - Processing statistics

### Example Output Structure

```json
{
  "question_index": 0,
  "question_text": "What is the first-line treatment for type 2 diabetes?",
  "original_data": {
    "question": "What is the first-line treatment for type 2 diabetes?",
    "answer": "Metformin",
    "context": "..."
  },
  "workflow_output": "Based on current guidelines...",
  "processing_time_seconds": 45.2,
  "timestamp": "2025-01-14T10:30:00",
  "status": "success"
}
```

## Available Dataset Splits

According to the [HuggingFace dataset page](https://huggingface.co/datasets/AIM-Harvard/MedBrowseComp/viewer/default/MedBrowseComp_50), the dataset includes:

- `MedBrowseComp_50` (50 rows) - Default, good for testing
- `MedBrowseComp_605` (605 rows) - Larger dataset
- `MedBrowseComp_CUA` (484 rows) - Alternative split

## Performance Considerations

- **Processing Time**: Each question may take 30-120 seconds depending on your pipeline configuration
- **Rate Limiting**: The script processes questions sequentially to avoid overwhelming your pipeline
- **Intermediate Saves**: Results are saved every 5 questions to prevent data loss
- **Memory Usage**: Large datasets may require significant memory for storing results

## Resuming Interrupted Processing

If processing is interrupted, you can resume from a specific index:

```bash
# Resume from question 25
python process_medbrowse_questions.py --start-idx 25
```

## Troubleshooting

### Common Issues

1. **Dataset Loading Fails**
   ```bash
   # Try manually downloading first
   python -c "from datasets import load_dataset; load_dataset('AIM-Harvard/MedBrowseComp', split='MedBrowseComp_50')"
   ```

2. **Pipeline Errors**
   ```bash
   # Test your pipeline first
   python main.py "Test question"
   ```

3. **Memory Issues**
   ```bash
   # Process in smaller batches
   python process_medbrowse_questions.py --start-idx 0 --end-idx 10
   ```

4. **Permission Errors**
   ```bash
   # Check output directory permissions
   mkdir -p outputs
   chmod 755 outputs
   ```

### Debug Mode

Enable debug mode for detailed logging:

```bash
python process_medbrowse_questions.py --debug
```

## Customization

### Modifying Question Extraction

The script automatically detects question fields, but you can customize the `_extract_question_text` method in `MedBrowseProcessor` if needed.

### Custom Output Formats

Modify the `save_results` method to add your preferred output formats (e.g., Excel, database storage).

### Pipeline Integration

The script uses the existing `run_agent_workflow_async` function. You can modify `_run_workflow_with_capture` to:
- Capture structured output instead of strings
- Add custom preprocessing/postprocessing
- Implement custom error handling

## Examples

### Process First 5 Questions
```bash
python process_medbrowse_questions.py --end-idx 5 --debug
```

### Process Large Dataset in Batches
```bash
# Batch 1: Questions 0-99
python process_medbrowse_questions.py --split MedBrowseComp_605 --start-idx 0 --end-idx 100

# Batch 2: Questions 100-199  
python process_medbrowse_questions.py --split MedBrowseComp_605 --start-idx 100 --end-idx 200
```

### Fast Processing Mode
```bash
python process_medbrowse_questions.py --max-step-num 1 --disable-background-investigation
```

## Contributing

Feel free to modify the scripts to better suit your needs. Key areas for enhancement:
- Custom output formats
- Parallel processing capabilities
- Integration with other datasets
- Enhanced error recovery

## References

- [MedBrowseComp Dataset](https://huggingface.co/datasets/AIM-Harvard/MedBrowseComp)
- [HuggingFace Datasets Documentation](https://huggingface.co/docs/datasets/)
- [Your DeepMed-RLX Pipeline Documentation](README.md) 