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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from pathlib import Path
import shutil

user_tmp = f"{Path.home()}/hf_cache"
Path(user_tmp).mkdir(parents=True, exist_ok=True)

print(f" HPC cache directory set to: {user_tmp}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

# ---------------------------------------------------------------------------
# WHAT CHANGED VS. THE ORIGINAL QWEN SCRIPT (LLmasajudgeQwen_...py)
# ---------------------------------------------------------------------------
# 1. judge_models_local now points at Qwen2.5-Coder-32B-Instruct instead of
#    the 7B model. Everything below extract_json / clamp_scores /
#    audit_with_rubric is UNCHANGED from your original Qwen script - the
#    parsing logic operates purely on the generated text string and does not
#    care how big the model behind it is, so it transfers as-is.
#
# 2. The original script loads every model in plain float16
#    (torch_dtype=torch.float16, no quantisation). That is fine at 7B
#    (~14 GB of weights) but NOT fine at 32B: float16 weights alone are
#    ~64 GB, which will not fit on a single 24-48 GB GPU and may not even
#    fit on an 80 GB A100/H100 once you add activations and KV cache.
#    load_model() below adds 4-bit NF4 quantisation via bitsandbytes
#    (BitsAndBytesConfig) so a 32B model needs roughly 18-20 GB instead -
#    comfortably within a single modern GPU, matching the "NF4" abbreviation
#    already listed in your thesis's List of Abbreviations. If your cluster
#    node has a large-memory GPU and you would rather not quantise, just
#    pass quantize=False to load_model().
#
# 3. Nothing else changes: same rubric prompt, same chat-template call,
#    same JSON extraction/rename/clamp pipeline, same output file naming
#    convention, so audit_results_/comparison_/summary_/audit_log_ files
#    for this model drop straight into your existing visualize_rd2.py /
#    analyze_rd2.py pipeline (JUDGE_ORDER, JUDGE_LABELS, SUFFIX in that
#    script just need the new model's filename suffix added).
# ---------------------------------------------------------------------------


def safe_load_dataset(name, split=None, retries=3):
    for i in range(retries):
        try:
            ds = load_dataset(name)
            return ds[split] if split else ds
        except Exception as e:
            print(f"⚠️ Dataset load failed ({name}) attempt {i+1}: {e}")
            time.sleep(5)

    raise RuntimeError(f"Failed to load dataset: {name}")


def load_model(name, quantize=True):
    print(f" Loading model: {name} (4-bit={quantize})")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            name,
            trust_remote_code=True,
            cache_dir=os.environ["HF_HOME"]
        )
        tokenizer.pad_token = tokenizer.eos_token

        if quantize:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                cache_dir=os.environ["HF_HOME"],
                low_cpu_mem_usage=True
            ).eval()
        else:
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

    matches = re.findall(
        r"\{[\s\S]*?\}",
        text
    )
    required = [
        "correct_tests_score",
        "coverage_score",
        "solution_score",
        "clarity_score",
        "representativeness_score"
    ]

    for m in reversed(matches):

        try:
            obj = json.loads(m)
            if all(k in obj for k in required):
                return m

        except:
            pass

    return None


def strip_reasoning_trace(text):
    """
    Only needed if you swap in a chain-of-thought / 'reasoning' model
    (e.g. QwQ-32B, DeepSeek-R1-Distill-Qwen-32B) instead of a plain
    instruct model. Those models emit a long <think>...</think> block
    BEFORE the actual answer, and extract_json()'s reversed-match scan
    already looks at the LAST JSON-shaped block in the text, so it
    usually finds the right one even with a reasoning trace present -
    but the trace can push the true JSON answer past max_new_tokens if
    you don't raise it. If you switch to a reasoning-style judge:
      1. Bump max_new_tokens well above 400 (try 1500-2500) so the
         model has room to finish thinking AND emit the JSON.
      2. Optionally call this function on the decoded text before
         extract_json(), to strip the trace and avoid accidentally
         matching a JSON-shaped fragment inside the reasoning itself.
    Not used by default here since Qwen2.5-Coder-32B-Instruct is a
    plain instruct model and does not emit a reasoning trace.
    """
    return re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()


def clamp_scores(result):

    keys = [
        "correct_tests_score",
        "coverage_score",
        "solution_score",
        "clarity_score",
        "representativeness_score"
    ]

    for k in keys:
        if k in result:
            try:
                result[k] = int(result[k])
                result[k] = max(1, min(5, result[k]))
            except:
                result[k] = 1

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

    result.setdefault("representativeness_score", 1)
    result.setdefault("summary", "")

    return result


def audit_with_rubric(model, tokenizer, prompt, tests, solution, log_path):

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

    outputs = model.generate(
        **inputs,
        max_new_tokens=400,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    generated = outputs[0][inputs["input_ids"].shape[1]:]

    text = tokenizer.decode(
        generated,
        skip_special_tokens=True
    )

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\nRAW JUDGE OUTPUT\n")
        f.write("-" * 50 + "\n")
        f.write(text)
        f.write("\n")

    try:
        json_str = extract_json(text)

        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\nEXTRACTED JSON\n")
            f.write("-" * 50 + "\n")
            f.write(str(json_str))
            f.write("\n")

        if not json_str:
            raise ValueError("No JSON found")

        result = json.loads(json_str)

        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\nPARSED RESULT\n")
            f.write("-" * 50 + "\n")
            f.write(json.dumps(result, indent=2))
            f.write("\n")

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
            "benchmark_quality_score": 1.0,
            "representativeness_score": 1,
            "summary": "parsing failed"
        }


def get_func_name(prompt):
    match = re.search(r"def\s+([a-zA-Z_]\w*)\s*\(", prompt)
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# Judge model to run. Swap the string below to try a different replacement
# for Mistral/DeepSeek (see the chat discussion for the full shortlist);
# everything else in this file works unchanged for any plain instruct model
# that supports tokenizer.apply_chat_template().
# ---------------------------------------------------------------------------
judge_models_local = [
    "Qwen/Qwen2.5-Coder-32B-Instruct",
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

    print(f" Matched samples: {len(common_keys)}")

    for judge_model_name in judge_models_local:

        model, tokenizer = load_model(judge_model_name, quantize=True)

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
            ]]
            .mean()
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
