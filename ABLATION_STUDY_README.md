# MedBrowse Ablation Study

This script performs an ablation study on the MedBrowse dataset to understand how the multi-agent pipeline improves with different numbers of research planning iterations (`max_plan_iterations`).

## What is the Ablation Study?

The script tests the pipeline with `max_plan_iterations` values from 1 to 5:

- **1**: Single planning iteration (fastest, baseline)
- **2**: Two planning iterations (allows refinement)
- **3**: Three planning iterations (more thorough planning)
- **4**: Four planning iterations (extensive planning)
- **5**: Five planning iterations (maximum planning depth)

The study measures:
- **Success Rate**: Percentage of questions answered successfully
- **Processing Time**: Average time per question
- **Answer Quality**: Extracted from pipeline outputs

## Quick Start

### Test Run (Recommended First)
```bash
# Run on first 5 questions only for testing
python ablation_study_medbrowse.py --sample-mode
```

### Full Study
```bash
# Run on all 50 questions in the MedBrowseComp_50 dataset
python ablation_study_medbrowse.py
```

### Custom Range
```bash
# Run on specific question range (e.g., questions 0-9)
python ablation_study_medbrowse.py --start-idx 0 --end-idx 10
```

### Debug Mode
```bash
# Run with detailed logging for troubleshooting
python ablation_study_medbrowse.py --debug --sample-mode
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--sample-mode` | Process first 5 questions only | Disabled |
| `--start-idx` | Starting question index | 0 |
| `--end-idx` | Ending question index | All questions |
| `--output-dir` | Base output directory | `outputs/ablation_study` |
| `--max-step-num` | Max steps per plan (constant) | 3 |
| `--disable-background-investigation` | Disable background search | Enabled |
| `--debug` | Enable debug logging | Disabled |
| `--disable-fallback` | Disable fallback processing | Enabled |

## Output Structure

The script creates organized outputs in the specified directory:

```
outputs/ablation_study/
├── max_plan_iterations_1/          # Results for max_plan_iterations=1
│   ├── medbrowse_results_*.json
│   ├── medbrowse_results_*.csv
│   └── medbrowse_summary_*.json
├── max_plan_iterations_2/          # Results for max_plan_iterations=2
│   └── ...
├── max_plan_iterations_3/          # Results for max_plan_iterations=3
│   └── ...
├── max_plan_iterations_4/          # Results for max_plan_iterations=4
│   └── ...
├── max_plan_iterations_5/          # Results for max_plan_iterations=5
│   └── ...
├── ablation_study_summary_*.json   # Complete study analysis
└── ablation_study_comparison_*.csv # Comparison table
```

## Key Output Files

### 1. Individual Experiment Results
Each `max_plan_iterations_X` folder contains:
- **JSON results**: Detailed results for each question
- **CSV results**: Tabular format for analysis
- **Summary**: Processing statistics

### 2. Comparative Analysis
- **`ablation_study_summary_*.json`**: Complete analysis with recommendations
- **`ablation_study_comparison_*.csv`**: Easy-to-analyze comparison table

## Understanding the Results

### Success Rate Trends
- **Increasing**: More planning iterations improve success
- **Decreasing**: Additional planning may introduce complexity/errors
- **Mixed**: Optimal value exists in the middle range
- **Stable**: Planning iterations have minimal impact

### Processing Time Trends
- **Increasing**: More iterations require more time (expected)
- **Stable**: Efficient planning with consistent timing

### Recommendations
The script provides specific recommendations based on:
- **Highest Success Rate**: Best for quality-focused applications
- **Fastest Processing**: Best for time-sensitive applications  
- **Best Efficiency**: Balanced approach

## Example Output

```
🎉 ABLATION STUDY COMPLETED
================================================================================

📊 STUDY OVERVIEW
   Dataset: AIM-Harvard/MedBrowseComp (MedBrowseComp_50)
   Questions: 0-5
   Total Time: 458.2s
   Tested Values: [1, 2, 3, 4, 5]

📈 RESULTS BY MAX_PLAN_ITERATIONS
   Value  Success Rate Avg Time/Q  Total Time 
   ------ ------------ ----------- -----------
   1      60.00%       45.2s       90.4s      
   2      80.00%       52.1s       104.2s     
   3      90.00%       58.7s       117.4s     
   4      85.00%       65.3s       130.6s     
   5      85.00%       72.9s       145.8s     

💡 RECOMMENDATIONS
   1. For highest success rate (90.00%): Use max_plan_iterations = 3
   2. For fastest processing (45.2s avg): Use max_plan_iterations = 1
   3. For best efficiency balance: Use max_plan_iterations = 3

🎯 BEST CONFIGURATIONS
   Highest Success Rate: max_plan_iterations = 3 (90.00%)
   Fastest Processing:   max_plan_iterations = 1 (45.2s avg)
```

## Tips for Running the Study

1. **Start Small**: Always use `--sample-mode` for initial testing
2. **Monitor Resources**: Full studies can take hours and use significant API calls
3. **Check Logs**: Use `--debug` if you encounter issues
4. **Save Outputs**: Results are automatically saved with timestamps
5. **Compare Results**: Use the CSV files for detailed analysis in Excel/Python

## Troubleshooting

### Common Issues
- **API Rate Limits**: Reduce question range or add delays
- **Memory Issues**: Process smaller batches with `--start-idx` and `--end-idx`
- **Network Errors**: Enable fallback with longer timeouts

### Getting Help
- Check the debug logs in each experiment folder
- Review the error messages in the summary files
- Start with `--sample-mode --debug` for detailed diagnostics

## Integration with Analysis Tools

The CSV outputs can be easily imported into:
- **Python**: `pandas.read_csv("ablation_study_comparison_*.csv")`
- **Excel**: Direct import for charts and analysis
- **R**: `read.csv("ablation_study_comparison_*.csv")`
- **Jupyter Notebooks**: For interactive analysis and visualization 