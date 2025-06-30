import os
import time
import argparse
import json

from ADDIE import ADDIE


def load_catalog():
    # 设置目录路径
    catalog_dir = "catalog"

    # 要读取的文件名列表
    json_filenames = [
        "student_profile.json",
        "instructor_preferences.json",
        "course_structure.json",
        "assessment_design.json",
        "teaching_constraints.json",
        "institutional_requirements.json",
        "prior_feedback.json"
    ]

    # 初始化结果字典
    data_catalog = {}

    # 遍历文件名并加载 JSON 数据
    for filename in json_filenames:
        filepath = os.path.join(catalog_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                key = filename.replace(".json", "")
                data_catalog[key] = data
        except Exception as e:
            print(f"Failed to load {filename}: {e}")

    # 示例输出：打印每个部分的字段数
    for section, content in data_catalog.items():
        print(f"{section}: {list(content.keys())} fields loaded.")
    
    return data_catalog




def run_instructional_design(course_name: str, copilot: bool = False, catalog: bool = False, model_name: str = "gpt-4o-mini", exp_name: str = ""):
    """
    Main function to run the instructional design workflow by sequentially
    executing the six deliberation processes
    
    Args:
        copilot: Whether to enable copilot mode with user feedback
        model_name: Name of the LLM model to use
        exp_name: Name of the experiment for logging purposes
    
    Returns:
        List of results from each process
    """
    # Ensure the OPENAI_API_KEY is set
    if not os.environ.get("OPENAI_API_KEY"):
        api_key = input("Please enter your OpenAI API key: ").strip()
        if not api_key:
            print("Error: OpenAI API key is required to run this workflow.")
            return
        os.environ["OPENAI_API_KEY"] = api_key
    
    # load input files
    data_catalog = load_catalog()
    
    # Get information about copilot mode
    mode_str = "COPILOT" if copilot else "AUTOMATIC"
    print("\n" + "="*80)
    print(f"INSTRUCTIONAL DESIGN WORKFLOW EXECUTION - {mode_str} MODE")
    print(f"Using SlidesDeliberation for enhanced slide generation")
    print("="*80 + "\n")
    
    if copilot:
        print("copilot mode enabled. You will be prompted for suggestions after each deliberation.")
        print("You can also choose to re-run a deliberation with your suggestions.\n")
    
    # Start timer
    start_time = time.time()
    
    # Create ADDIE instance
    addie = ADDIE(course_name, model_name=model_name, copilot=copilot, catalog=catalog, data_catalog=data_catalog)
    
    # Run the workflow
    output_dir = f"./exp/{exp_name}/"
    os.makedirs(output_dir, exist_ok=True)
    addie.run(output_dir=output_dir)
    
    # Calculate execution time
    execution_time = time.time() - start_time
    hours, rem = divmod(execution_time, 3600)
    minutes, seconds = divmod(rem, 60)
    
    # Print completion message
    print("\n" + "="*80)
    print(f"WORKFLOW COMPLETED IN: {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}")
    print("="*80 + "\n")


if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)
    os.environ["OPENAI_API_KEY"] = config.get("OPENAI_API_KEY", "")

    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Run instructional design workflow")
    parser.add_argument("course_name", type=str, help="Name of the course")
    parser.add_argument(
        "--copilot", 
        action="store_true",
        help="Enable copilot mode with user feedback"
    )
    parser.add_argument(
        "--catalog", 
        action="store_true",
        help="Enable catalog mode with catalog files input"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--exp", 
        type=str,
        default="exp1",
        help="Experiment name for logging"
    )

    args = parser.parse_args()
    
    # Run workflow with specified options
    run_instructional_design(args.course_name, copilot=args.copilot, catalog=args.catalog, model_name=args.model, exp_name=args.exp)