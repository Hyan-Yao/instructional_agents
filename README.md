# EduAgents - Intelligent Instructional Design Workflow

An AI-powered instructional design system based on the ADDIE model for automated course creation and evaluation.

## Quick Start

### 1. Setup Configuration

Create or edit `config.json`:
```json
{
  "OPENAI_API_KEY": "your_openai_api_key_here"
}
```

### 2. Install Dependencies [Incomplete]

```bash
pip install openai pandas pathlib
```

## Usage Examples

### Basic Workflow Execution

**Entry Point**: `run.py` - Main workflow entry point

```bash
# Simple course generation
python run.py "Introduction to Machine Learning"

# With specific model
python run.py "Data Structures" --model gpt-4o-mini

# With experiment name
python run.py "Web Development" --exp web_dev_v1

# Interactive copilot mode
python run.py "Database Systems" --copilot

# Using catalog data
python run.py "Software Engineering" --catalog --exp se_course
```

### Command Line Arguments

```bash
python run.py <course_name> [OPTIONS]

Required:
  course_name           Name of the course to design

Options:
  --copilot            Enable interactive copilot mode
  --catalog            Use structured data from catalog/ directory
  --model MODEL        OpenAI model (default: gpt-4o-mini)
  --exp EXP_NAME       Experiment name (default: exp1)
```

### Automatic Evaluation

**Entry Point**: `evaluate.py` - Automatic assessment system

```bash
# Evaluate specific experiment
python evaluate.py --exp web_dev_v1
```

## Background Execution with Logging

### Using nohup for Long-Running Tasks

```bash
# Run workflow in background with logging
nohup python run.py "Advanced Machine Learning" --exp ml_advanced > logs/ml_course.log 2>&1 &

# Monitor progress
tail -f logs/ml_course.log
```

## Example Workflows

### Complete Course Development Pipeline

```bash
# Step 1: Generate course materials
python run.py "Python Programming Fundamentals" \
  --catalog \
  --model gpt-4o-mini \
  --exp python_course

# Step 2: Evaluate generated materials
python evaluate.py --exp python_course

# Step 3: Review results
ls exp/python_course/
ls eval/python_course/
```

### Interactive Development with Copilot

```bash
# Start interactive session
python run.py "Advanced Algorithms" --copilot --exp algorithms_v2

# System will pause after each phase for your input:
# - Analysis phase → Your feedback
# - Design phase → Your feedback  
# - Development phase → Your feedback
```

### View Results

```bash
# View generated course structure
tree exp/your_experiment_name/

# View evaluation summary
cat eval/your_experiment_name/evaluation_results/evaluation_summary.md

# View validation reports
ls eval/your_experiment_name/validation_reports/
```

