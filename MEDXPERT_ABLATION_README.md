# MedXpertQA Ablation Study

This script runs an ablation study on the **MedXpertQA text-only dataset** with different `max_plan_iterations` values (1 to 5) to understand how the multi-agent pipeline's performance improves with more research planning iterations.

## Dataset

The [MedXpertQA dataset](https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA) from TsinghuaC3I contains medical multiple-choice questions with:
- **Text subset**: Text-only questions (2.46k questions)
- **MM subset**: Multimodal questions with images (2.01k questions)
- **Format**: Each question has options A-E with one correct answer
- **Metadata**: Medical task, body system, question type classification

## Quick Start

### Test Setup (Sample Mode)
```bash
# Run on first 10 questions to test the setup
python ablation_study_medxpert.py --sample-mode
```

### Default Run 
```bash
# Run on first 50 questions (default behavior)
python ablation_study_medxpert.py
```

### Custom Runs
```bash
# Run on specific question range
python ablation_study_medxpert.py --end-idx 100

# Run with debug logging
python ablation_study_medxpert.py --debug --sample-mode

# Disable background investigation
python ablation_study_medxpert.py --disable-background-investigation
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--sample-mode` | Run on first 10 questions only | Disabled |
| `--end-idx N` | Process first N questions | 50 |
| `--start-idx N` | Start from question N | 0 |
| `--debug` | Enable debug logging | Disabled |
| `--disable-background-investigation` | Disable background research | Enabled |
| `--dataset` | HuggingFace dataset name | TsinghuaC3I/MedXpertQA |
| `--subset` | Dataset subset (Text/MM) | Text |
| `--split` | Dataset split | test |
| `--output-dir` | Base output directory | outputs/ablation_study_medxpert |
| `--max-step-num` | Max steps per plan | 3 |

## Output Structure

The ablation study creates the following directory structure:

```
outputs/ablation_study_medxpert/
├── max_plan_iterations_1/          # Results for max_plan_iterations=1
│   ├── medxpert_results_TIMESTAMP.json
│   ├── medxpert_results_TIMESTAMP.csv  
│   └── medxpert_summary_TIMESTAMP.json
├── max_plan_iterations_2/          # Results for max_plan_iterations=2
│   └── ...
├── max_plan_iterations_3/          # Results for max_plan_iterations=3
│   └── ...
├── max_plan_iterations_4/          # Results for max_plan_iterations=4
│   └── ...
├── max_plan_iterations_5/          # Results for max_plan_iterations=5
│   └── ...
├── medxpert_ablation_study_summary_TIMESTAMP.json    # Overall study summary
└── medxpert_ablation_study_comparison_TIMESTAMP.csv  # Comparison table
```

## Key Features

### Answer Extraction
- **Multiple Pattern Recognition**: Extracts answer choices (A-E) using regex patterns
- **Flexible Parsing**: Handles various response formats like "Answer: A", "(B)", "The answer is C"
- **Fallback Detection**: Looks for standalone letters in output when patterns fail

### Accuracy Measurement
- **Overall Accuracy**: Correct answers / Total questions (including failures)
- **Accuracy Among Successful**: Correct answers / Successfully processed questions
- **Success Rate**: Questions processed without errors / Total questions

### Error Handling
- **Retry Logic**: Up to 5 retries for parsing errors with exponential backoff
- **Fallback Mode**: Simplified processing (max_plan_iterations=1, max_step_num=2, no background investigation) when primary processing fails
- **Comprehensive Error Tracking**: Detailed error information for debugging

### Timing Capture
- **Per-Question Timing**: Individual processing time for each question
- **Experiment-Level Timing**: Total time per max_plan_iterations value
- **Study-Level Timing**: Overall ablation study duration

## Results Interpretation

### Success Metrics
- **Success Rate**: Percentage of questions processed without critical errors
- **Accuracy Rate**: Percentage of questions answered correctly (out of all questions)
- **Accuracy Among Successful**: Percentage of questions answered correctly (out of successfully processed questions only)

### Performance Analysis
The study provides insights into:
1. **Reliability vs Planning**: How success rates change with more planning iterations
2. **Accuracy vs Planning**: How answer accuracy improves with more iterations
3. **Speed vs Planning**: Processing time trade-offs for different iteration counts
4. **Medical Domain Performance**: Breakdown by medical task, body system, and question type

### Comparison Table (CSV)
The comparison CSV file allows easy analysis with columns:
- `max_plan_iterations`: Number of planning iterations (1-5)
- `success_rate`: Percentage of questions processed successfully
- `accuracy_rate`: Percentage of questions answered correctly (overall)
- `accuracy_among_successful`: Percentage correct among successful processes
- `avg_processing_time`: Average seconds per question
- `total_experiment_time`: Total experiment duration
- `successful_questions`: Count of successfully processed questions
- `correct_answers`: Count of correctly answered questions
- `failed_questions`: Count of failed questions

## Technical Details

### Question Format
MedXpertQA questions are formatted as:
```
Question text here...

Answer Choices:
(A) Option A text
(B) Option B text
(C) Option C text
(D) Option D text
(E) Option E text

Please provide your answer as a single letter (A, B, C, D, or E) followed by a brief explanation.
```

### Workflow Integration
- Uses the existing `run_agent_workflow_async()` function
- Captures stdout/stderr to extract agent responses
- Integrates with the multi-agent pipeline's planning system
- Maintains consistency with the existing MedBrowse processing approach

### Dataset Integration
- Automatically downloads from HuggingFace Hub
- Supports both Text and MM subsets
- Preserves original question metadata (medical_task, body_system, question_type)
- Handles dataset pagination and filtering

## Troubleshooting

### Common Issues

**1. Dataset Loading Errors**
```bash
# Check your internet connection and HuggingFace access
pip install datasets
```

**2. Memory Issues with Large Question Sets**
```bash
# Run smaller batches
python ablation_study_medxpert.py --end-idx 20
```

**3. Parsing Failures**
- The script automatically retries parsing errors
- Check debug logs for pattern matching issues
- Fallback mode activates automatically for persistent failures

**4. Slow Processing**
```bash
# Disable background investigation for faster processing
python ablation_study_medxpert.py --disable-background-investigation
```

### Debug Mode
```bash
# Enable verbose logging to see detailed processing steps
python ablation_study_medxpert.py --debug --sample-mode
```

## Dependencies

Required Python packages:
```bash
pip install pandas datasets asyncio pathlib
```

The script also requires your existing DeepMed-RLX environment with:
- `src.workflow` module with `run_agent_workflow_async()`
- Multi-agent pipeline infrastructure
- OpenAI API access (if using GPT models)

## Comparison with MedBrowse Study

| Aspect | MedBrowse Ablation | MedXpertQA Ablation |
|--------|-------------------|---------------------|
| **Dataset** | AIM-Harvard/MedBrowseComp | TsinghuaC3I/MedXpertQA |
| **Question Type** | Open-ended research | Multiple choice (A-E) |
| **Answer Format** | Free text | Single letter + explanation |
| **Success Metric** | Completion without errors | Completion + answer extraction |
| **Accuracy Metric** | Not applicable | Correct vs incorrect answers |
| **Default Size** | 20 questions | 50 questions |
| **Processing Focus** | Research planning workflow | Medical knowledge assessment |

Both studies test the same `max_plan_iterations` parameter (1-5) to understand how research planning iterations affect the multi-agent pipeline's performance in different medical question-answering contexts. 