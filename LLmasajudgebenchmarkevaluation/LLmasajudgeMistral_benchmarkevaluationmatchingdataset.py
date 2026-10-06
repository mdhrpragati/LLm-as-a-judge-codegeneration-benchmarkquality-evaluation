import os

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_XET"] = "1"

HF_ROOT = "/ediss_data/pmanandh/Thesis/hf_cache"
OUTPUT_DIR = "/ediss_data/pmanandh/Thesis/outputs"

os.environ["HF_HOME"] = HF_ROOT
os.environ["TRANSFORMERS_CACHE"] = f"{HF_ROOT}/transformers"
os.environ["HF_DATASETS_CACHE"] = f"{HF_ROOT}/datasets"
os.environ["HF_HUB_CACHE"] = f"{HF_ROOT}/hub"
os.environ["HUGGINGFACE_HUB_CACHE"] = f"{HF_ROOT}/hub"

os.makedirs(HF_ROOT, exist_ok=True)
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)
os.makedirs(os.environ["HF_HUB_CACHE"], exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
import re
import json
import time
import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import shutil
from json import JSONDecoder

user_tmp = f"{Path.home()}/hf_cache"
Path(user_tmp).mkdir(parents=True, exist_ok=True)

print(f" HPC cache directory set to: {user_tmp}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

def safe_load_dataset(name, split=None, retries=3):
    for i in range(retries):
        try:
            ds = load_dataset(name)
            return ds[split] if split else ds
        except Exception as e:
            print(f"⚠️ Dataset load failed ({name}) attempt {i+1}: {e}")
            time.sleep(5)

    raise RuntimeError(f"Failed to load dataset: {name}")

def load_model(name):
    print(f" Loading model: {name}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            name,
            trust_remote_code=True,
            cache_dir=os.environ["HF_HOME"]
        )
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=os.environ["HF_HOME"],
            low_cpu_mem_usage=True
        ).eval()

        return model, tokenizer

    except Exception as e:
        print(f" Failed to load {name}: {e}")
        return None, None

def extract_json(text):

    decoder = JSONDecoder()

    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = decoder.raw_decode(text[i:])

                if isinstance(obj, dict):
                    return obj

            except:
                pass

    return None



def clamp_scores(result):

    keys = [
        "correct_tests_score",
        "coverage_score",
        "solution_score",
        "clarity_score",
        "representativeness_score",
    ]
    
    result.setdefault("correct_tests_score", 1.0)
    result.setdefault("coverage_score", 1.0)
    result.setdefault("solution_score", 1.0)
    result.setdefault("clarity_score", 1.0)
    result.setdefault("representativeness_score", 1.0)
    result.setdefault("summary", "")

    for k in keys:
        if k in result:
            try:
                result[k] = float(result[k])
                result[k] = max(1.0, min(5.0, result[k]))
            except:
                result[k] = 1.0

    result["benchmark_quality_score"] = round(
        (
            result["correct_tests_score"] +
            result["coverage_score"] +
            result["solution_score"] +
            result["clarity_score"] +
            result["representativeness_score"]
        ) / 5,
        2
    )
    
    


    return result

def audit_with_rubric(model, tokenizer, prompt, tests, solution, log_path):

    # audit_prompt = f"""
    # You are an expert software engineering benchmark evaluator.

    # Problem:
    # {prompt}

    # Tests:
    # {tests}

    # Reference Solution:
    # {solution}

    # Evaluate the benchmark using the following criteria:

    # 1. Correctness of Tests (1-5)
    # 2. Test Coverage (1-5)
    # 3. Solution Quality (1-5)
    # 4. Problem Clarity (1-5)

    # Important:

    # - Do not assume the benchmark is high quality.
    # - Actively identify weaknesses.
    # - Consider missing edge cases.
    # - Consider weak or incomplete tests.
    # - Consider unclear task descriptions.
    # - Consider weaknesses in the reference solution.

    # Scoring Guidance:

    # - Score 5 only if no significant weaknesses exist.
    # - Score 4 if minor weaknesses exist.
    # - Score 3 for an average benchmark.
    # - Score 2 for major weaknesses.
    # - Score 1 for severe benchmark problems.
    
    # A score of 5 should only be assigned when no meaningful weaknesses are found.

    # Return ONLY valid JSON.
    # Do not output any text before or after the JSON.
    # Do not wrap the JSON in markdown.

    # Example:

    # {{
    #     "correct_tests_score": 4,
    #     "coverage_score": 3,
    #     "solution_score": 4,
    #     "clarity_score": 4,
    #     "summary": "Some edge cases are missing but the benchmark is generally well designed."
    # }}

    # Now evaluate this benchmark.
    # """
    
    audit_prompt = f"""
    You are an expert software engineering benchmark evaluator.

    Problem:
    {prompt}

    Tests:
    {tests}

    Reference Solution:
    {solution}

    Evaluate the benchmark using the following criteria:

    1. Correctness of Tests (1-5)
    2. Test Coverage (1-5)
    3. Solution Quality (1-5)
    4. Problem Clarity (1-5)
    5. Representativeness (1-5)

    Important:

    - Do not assume the benchmark is high quality.
    - Actively identify weaknesses.
    - Actively search for reasons to reduce scores.
    - Consider missing edge cases.
    - Consider weak or incomplete tests.
    - Consider unclear task descriptions.
    - Consider weaknesses in the reference solution.
    - Consider whether the task is realistic and representative of software engineering practice.

    Scoring Guidance:

    - Score 5 only if no significant weaknesses exist.
    - Score 4 if minor weaknesses exist.
    - Score 3 for an average benchmark.
    - Score 2 for major weaknesses.
    - Score 1 for severe benchmark problems.
    
    When assigning scores, be conservative.

    If any non-trivial weakness is identified, the score should be less than 5.

    Return ONLY valid JSON.
    Do not output any text before or after the JSON.
    Do not wrap the JSON in markdown.

    Example:

    {{
        "correct_tests_score": 4,
        "coverage_score": 3,
        "solution_score": 4,
        "clarity_score": 4,
        "representativeness_score": 3,
        "summary": "Some edge cases are missing but the benchmark is generally well designed."
    }}
    
    Now evaluate this benchmark.
    """

    messages = [
        {
            "role": "user",
            "content": audit_prompt
        }
    ]

    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        chat_text,
        return_tensors="pt",
        truncation=True,
        max_length=4096
    ).to(device)
    
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\nINPUT TOKENS: {inputs['input_ids'].shape[1]}\n")

    outputs = model.generate(
        **inputs,
        max_new_tokens=250,
        do_sample=False,
        temperature=0.0,
        repetition_penalty=1.05,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

    generated = outputs[0][inputs["input_ids"].shape[1]:]

    text = tokenizer.decode(
        generated,
        skip_special_tokens=True
    )
    
    # print("\nRAW OUTPUT:")
    # print(text[:1000])
    
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\nRAW JUDGE OUTPUT\n")
        f.write("-" * 50 + "\n")
        f.write(text)
        f.write("\n")


    try:
        result = extract_json(text)
        
        rename_map = {
            "correctness_score": "correct_tests_score",
            "correctness_of_tests_score": "correct_tests_score",

            "test_coverage_score": "coverage_score",
            "coverage": "coverage_score",

            "solution_quality_score": "solution_score",
            "solution_quality": "solution_score",

            "problem_clarity_score": "clarity_score",
            "clarity": "clarity_score",

            "representativeness": "representativeness_score",
            "representative_score": "representativeness_score",
            "representativeness_score_value": "representativeness_score",
        }
        
        if result is not None:
            for old, new in rename_map.items():
                if old in result and new not in result:
                    result[new] = result[old]
                
        required = [
            "correct_tests_score",
            "coverage_score",
            "solution_score",
            "clarity_score",
            "representativeness_score"
        ]
        
        if result is not None:
            if not all(k in result for k in required):
                result = None
    

        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\nEXTRACTED JSON\n")
            f.write("-" * 50 + "\n")
            f.write(json.dumps(result, indent=2))
            f.write("\n")

        if result is None:

            text2 = text.replace("'", '"')

            result = extract_json(text2)
            
            if result is not None:
                for old, new in rename_map.items():
                    if old in result and new not in result:
                        result[new] = result[old]

                if not all(k in result for k in required):
                    result = None

            if result is None:
                raise ValueError("No JSON found")
        
        # print("RAW OUTPUT:")
        # print(text[:1000])

        # print("JSON:")
        # print(json_str)


        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\nPARSED RESULT\n")
            f.write("-" * 50 + "\n")
            f.write(json.dumps(result, indent=2))
            f.write("\n")
        
        # print("PARSED:")
        # print(result)
        return clamp_scores(result)

    except Exception as e:
        
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\nJSON PARSE FAILED\n")
            f.write("-" * 50 + "\n")
            f.write(str(e))
            f.write("\n")
            f.write(text)
            f.write("\n")
        return {
            "correct_tests_score": 1,
            "coverage_score": 1,
            "solution_score": 1,
            "clarity_score": 1,
            "representativeness_score": 1,
            "benchmark_quality_score": 1.0,
            "summary": "parsing failed"
        }



def get_func_name(prompt):
    match = re.search(r"def\s+([a-zA-Z_]\w*)\s*\(", prompt)
    return match.group(1) if match else None


judge_models_local = [
    #"Qwen/Qwen2.5-Coder-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    #"deepseek-ai/deepseek-coder-6.7b-instruct",
]


def run_audit():

    print(" Loading datasets...")

    humaneval = safe_load_dataset("openai/openai_humaneval", "test")
    humanevalnext = safe_load_dataset("AISE-TUDelft/HumanEvalNext", "train")

    use_task_id = "task_id" in humaneval.column_names and "task_id" in humanevalnext.column_names

    if use_task_id:
        he_map = {x["task_id"]: x for x in humaneval}
        hn_map = {x["task_id"]: x for x in humanevalnext}
        common_keys = set(he_map) & set(hn_map)
    else:
        he_map = {get_func_name(x["prompt"]): x for x in humaneval}
        hn_map = {get_func_name(x["prompt"]): x for x in humanevalnext}

        he_map = {k: v for k, v in he_map.items() if k}
        hn_map = {k: v for k, v in hn_map.items() if k}
        common_keys = set(he_map) & set(hn_map)

    
    #common_keys = list(common_keys)[:5]
    
    print(f" Matched samples: {len(common_keys)}")
    
    for judge_model_name in judge_models_local:

        model, tokenizer = load_model(judge_model_name)

        if model is None:
            print(f" Skipping {judge_model_name}")
            continue
        
        log_path = (
            f"{OUTPUT_DIR}/"
            f"audit_log_{judge_model_name.replace('/', '_')}.txt"
        )

        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"JUDGE MODEL: {judge_model_name}\n")
            f.write("=" * 100 + "\n")

        results = []

        for key in tqdm(common_keys, desc=f"Evaluating {judge_model_name}"):

            he_item = he_map[key]
            hn_item = hn_map[key]
            
            
            
            with open(log_path, "a", encoding="utf-8") as f:
                f.write("\n")
                f.write("=" * 100 + "\n")
                f.write(f"TASK: {key}\n")
                f.write(f"DATASET: HumanEval\n")
                f.write("=" * 100 + "\n")

                f.write("\nPROMPT\n")
                f.write("-" * 50 + "\n")
                f.write(he_item["prompt"])
                f.write("\n")

                f.write("\nTESTS\n")
                f.write("-" * 50 + "\n")
                f.write(str(he_item.get("test", "")))
                f.write("\n")

                f.write("\nREFERENCE SOLUTION\n")
                f.write("-" * 50 + "\n")
                f.write(str(he_item.get("canonical_solution", "")))
                f.write("\n")


            # HumanEval
            res_he = audit_with_rubric(
                model,
                tokenizer,
                he_item["prompt"],
                he_item.get("test", ""),
                he_item.get("canonical_solution", ""),
                log_path
            )

            results.append({
                "judge_model": judge_model_name,
                "dataset": "HumanEval",
                "task_id": key,
                **res_he
            })
            
            with open(log_path, "a", encoding="utf-8") as f:

                f.write("\n")
                f.write("=" * 100 + "\n")
                f.write(f"TASK: {key}\n")
                f.write("DATASET: HumanEvalNext\n")
                f.write("=" * 100 + "\n")

                f.write("\nPROMPT\n")
                f.write("-" * 50 + "\n")
                f.write(hn_item["prompt"])
                f.write("\n")

                f.write("\nTESTS\n")
                f.write("-" * 50 + "\n")
                f.write(str(hn_item.get("test", "")))
                f.write("\n")

                f.write("\nREFERENCE SOLUTION\n")
                f.write("-" * 50 + "\n")
                f.write(str(hn_item.get("canonical_solution", "")))
                f.write("\n")

            # HumanEvalNext
            res_hn = audit_with_rubric(
                model,
                tokenizer,
                hn_item["prompt"],
                hn_item.get("test", ""),
                hn_item.get("canonical_solution", ""),
                log_path
            )

            results.append({
                "judge_model": judge_model_name,
                "dataset": "HumanEvalNext",
                "task_id": key,
                **res_hn
            })

   
        df = pd.DataFrame(results)
        
        dataset_summary = (
            df.groupby("dataset")[[
                "correct_tests_score",
                "coverage_score",
                "solution_score",
                "clarity_score",
                "representativeness_score",
                "benchmark_quality_score"
            ]].mean()
        )

        summary_file = (
            f"{OUTPUT_DIR}/"
            f"summary_{judge_model_name.replace('/', '_')}.xlsx"
        )

        dataset_summary.to_excel(summary_file)
        
        comparison = (
            df.pivot(
                index="task_id",
                columns="dataset",
                values="benchmark_quality_score"
            )
        )

        comparison["difference"] = (
            comparison["HumanEvalNext"]
            - comparison["HumanEval"]
        )

        output_file = (
            f"{OUTPUT_DIR}/"
            f"audit_results_{judge_model_name.replace('/', '_')}.xlsx"
        )

        comparison_file = (
            f"{OUTPUT_DIR}/"
            f"comparison_{judge_model_name.replace('/', '_')}.xlsx"
        )

        comparison.to_excel(
            comparison_file
        )
        df.to_excel(output_file, index=False)

        print(f"\n Saved: {output_file}")

        print("\n Averages:")
        print(df.groupby("dataset")[[
            "correct_tests_score",
            "coverage_score",
            "solution_score",
            "clarity_score",
            "representativeness_score",
            "benchmark_quality_score"
        ]].mean())

                # cleanup
        del model, tokenizer
        torch.cuda.empty_cache()


def clear_hf_cache():

    

    if os.path.exists(HF_ROOT):
        shutil.rmtree(HF_ROOT)

    print("HF cache removed")


if __name__ == "__main__":

    run_audit()

    clear_hf_cache()

    print("ALL CACHE CLEANED")