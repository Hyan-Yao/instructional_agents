# Instructional Design Copilot using ADDIE Framework

This project implements an AI-assisted instructional design pipeline based on the ADDIE framework (Analysis, Design, Development, Implementation, Evaluation). It utilizes OpenAI's GPT models to guide curriculum development, content generation, and slide preparation, optionally with human feedback via a "copilot mode".

## 📁 Directory Structure

```

.
├── ADDIE.py                 # Core ADDIE workflow logic
├── agents.py               # Agents for different ADDIE stages
├── catalog/                # Input JSON files (e.g., student profile, course structure)
├── eval/                   # Evaluation scripts and logs
├── evaluate.py             # Evaluation logic
├── exp/                    # Output logs and experiment results
├── run.py                  # Main entrypoint script
├── slides.py               # Slide generation module

````


## 🚀 How to Run

### Basic Usage

```bash
python run.py --model gpt-4o-mini --catalog --exp test_run
```

### Available Options

| Argument    | Type   | Description                                              |
| ----------- | ------ | -------------------------------------------------------- |
| `--copilot` | flag   | Enable copilot mode (manual feedback after each step)    |
| `--catalog` | flag   | Use catalog-based inputs from `catalog/` folder          |
| `--model`   | string | Specify OpenAI model to use (default: `gpt-4o-mini`)     |
| `--exp`     | string | Experiment name (output will be saved under `exp/{exp}`) |

### Example with Copilot

```bash
python run.py --copilot --catalog --model gpt-4o-mini --exp interactive_session
```

## 📦 Input Files (in `catalog/`)

The system expects the following structured JSON files:

* `student_profile.json`
* `instructor_preferences.json`
* `course_structure.json`
* `assessment_design.json`
* `teaching_constraints.json`
* `institutional_requirements.json`
* `prior_feedback.json`

These files represent background knowledge for curriculum design and must follow a key-value format.

## 🧠 Core Workflow

The pipeline follows the ADDIE phases:

1. **Analysis** — Understand learner profile, goals
2. **Design** — Draft learning objectives, course layout
3. **Development** — Generate slides, scripts, and activities
4. **Implementation** — Plan for delivery and tools
5. **Evaluation** — Generate rubrics and feedback prompts

You can run the entire pipeline end-to-end with or without user intervention (`--copilot` flag).

## 🛡️ Environment Variable

Set your OpenAI API key (required):

```bash
export OPENAI_API_KEY=your-key-here
```

Alternatively, the script will prompt you for the key interactively.

## 📂 Output

* Outputs (slides, assessments, scripts) are saved in `exp/{exp_name}/`
* Intermediate results are logged per ADDIE phase

## ✍️ Author

DaRL, ASU