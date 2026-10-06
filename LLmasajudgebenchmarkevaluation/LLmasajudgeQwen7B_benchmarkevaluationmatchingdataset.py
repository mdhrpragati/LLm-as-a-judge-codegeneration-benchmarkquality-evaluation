import os

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_XET"] = "1"
# Helps avoid fragmentation-related CUDA OOM on long-running jobs.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
import random
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
import pandas as pd
import transformers
import datasets as datasets_pkg
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

user_tmp = f"{Path.home()}/hf_cache"
Path(user_tmp).mkdir(parents=True, exist_ok=True)

print(f" HPC cache directory set to: {user_tmp}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

# ---------------------------------------------------------------------------
# Reproducibility: do_sample=False already makes generation itself
# deterministic, but seeding covers any other source of randomness touched
# during the run (e.g. dataset shuffling if that ever changes upstream).
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# JUDGE: Qwen2.5-Coder-7B-Instruct
# Your original RD2 baseline judge, now rebuilt on the shared hardened
# template (resumable, per-task fault-isolated, and with the merged
# extraction heuristics below) instead of its own standalone script. Also
# gives you a same-family, smaller-scale counterpart to the 14B Qwen judge
# in the panel, useful for checking whether the panel's Qwen-line results
# are scale-sensitive on top of the cross-architecture comparison. One of
# five judges in the panel (Qwen2.5-Coder-7B-Instruct, Qwen2.5-Coder-14B-Instruct, Phi-4, OLMo-2-13B-Instruct, Gemma-2-9B-it).
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# WHAT'S NEW IN THIS VERSION vs. the original per-judge scripts
# (LLmasajudgeQwen14B_/Phi4_/OLMo2_13B_/Gemma2_9B_benchmarkevaluationmatching
# dataset.py). All four scripts got the same fixes so that a cross-judge
# comparison (Qwen2.5-Coder-14B-Instruct, Phi-4, OLMo-2-13B-Instruct,
# Gemma-2-9B-it) is apples-to-apples: same rubric prompt, same rubric
# criteria, same clamping logic, same file-naming convention as before -
# only the robustness/reproducibility plumbing below changed.
#
# 1. RESUMABLE RUNS. Every scored (dataset, task_id) row is appended to a
#    checkpoint_<judge>.jsonl file as soon as it's produced. If the job is
#    killed or preempted (this looks like it's running under an HPC
#    scheduler, so pre-emption is a real risk for a 13-14B model over ~160
#    audits) rerunning the same script picks up where it left off instead
#    of re-scoring everything and burning GPU hours you already spent. To
#    force a clean rerun, delete checkpoint_<judge>.jsonl (and optionally
#    audit_log_<judge>.txt) first.
#
# 2. DETERMINISTIC TASK ORDER. common_keys is now sorted rather than a raw
#    Python set (whose iteration order is not guaranteed stable). This
#    matters once you're comparing four separate judge runs against each
#    other and want the audit log / progress bar to walk the tasks in the
#    same order every time.
#
# 3. PER-TASK FAULT ISOLATION. A single malformed sample, a template quirk
#    specific to one model family, or a transient CUDA error used to be
#    able to kill the entire multi-hour run. Every audit call is now
#    wrapped so a failure on one task is logged, recorded as a flagged row
#    (judge_error != None), and the run moves on to the next task instead
#    of losing everything already scored.
#
# 4. OOM RETRY. generate() now retries (with cache clearing) on a CUDA OOM
#    before giving up on a single sample - this is squarely aimed at the
#    OOM issues you hit earlier this session with the larger judge models.
#
# 5. MORE ROBUST JSON EXTRACTION. The four judges are architecturally
#    unrelated (Qwen, Microsoft, AllenAI, Google) and will not all follow
#    "return ONLY JSON, no markdown" equally well. extract_json() now
#    strips ```json fences first and uses a quote-aware balanced-brace
#    scanner (in addition to, not instead of, the original regex) so a
#    stray "}" inside a free-text "summary" value, or a markdown-fenced
#    response, doesn't fail parsing for one judge but not another.
#
# 6. CHAT-TEMPLATE FALLBACK. If tokenizer.apply_chat_template() ever raises
#    for a specific checkpoint/template quirk, the script now falls back to
#    a plain-text prompt instead of crashing the whole run.
#
# 7. RUN METADATA. Each judge run now also writes
#    run_metadata_<judge>.json - quantisation config, package versions,
#    seed, generation config, matched-task count, and a breakdown of
#    parse/OOM/exception failures. This is exactly the reproducibility
#    detail your methodology chapter needs to describe per judge, and the
#    failure breakdown is itself evidence for the judge-reliability
#    discussion (which architectures produced clean JSON most reliably).
#
# 8. clamp_scores() reordered so representativeness_score gets its default
#    BEFORE it's used in the mean, instead of after (the original order
#    only worked by accident, because extract_json's required-key check
#    already guaranteed the key was present).
#
# 9. RESTORED PER-JUDGE-CURATED EXTRACTION HEURISTICS from your original
#    three single-judge scripts (Qwen2.5-Coder-7B, Mistral-7B-Instruct,
#    DeepSeek-Coder-6.7B), which had each independently hand-tuned their
#    own extract_json() after observing that specific judge's failure
#    modes. These are folded into the shared extract_json() below so every
#    judge in the panel benefits from all three, rather than each judge
#    only getting the one heuristic it originally shipped with:
#      - key-aliasing (RUBRIC_KEY_RENAME_MAP): originally only in the
#        Mistral and DeepSeek scripts, for a judge that names a field
#        e.g. "correctness_score" instead of "correct_tests_score".
#      - the "representativeness_score"-anchored regex: originally only
#        in the DeepSeek script, for picking the right block when a
#        response contains more than one brace-delimited chunk.
#      - the single-quote-to-double-quote retry: originally only in the
#        Mistral script (which used a raw-decode JSONDecoder scan, not
#        regex, specifically to cope with this), for a judge that emits
#        Python-dict-style single-quoted pseudo-JSON.
#    extract_json() now returns the parsed, key-aliased dict directly
#    (previously it returned a raw JSON substring that audit_with_rubric
#    then re-parsed) - the log's "EXTRACTED JSON" and "PARSED RESULT"
#    sections are merged into one "EXTRACTED + PARSED RESULT" section
#    accordingly, since extraction and parsing are no longer two separate
#    steps worth logging separately.
#
# Rubric prompt, scoring dimensions, generation settings (do_sample=False,
# max_new_tokens=400, 4096-token truncation), quantisation (4-bit NF4), and
# the audit_results_/comparison_/summary_/audit_log_ output files are all
# UNCHANGED, so this stays a drop-in replacement for your existing
# visualize_rd2.py pipeline - just point JUDGE_ORDER/JUDGE_LABELS at the
# new four judges.
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

        # Quick sanity check so you can confirm quantisation actually
        # applied before burning HPC time on the full run - check the
        # printed device map / GPU memory against what you expect.
        try:
            print(" hf_device_map:", model.hf_device_map)
        except Exception:
            pass
        if torch.cuda.is_available():
            print(f" GPU memory allocated after load: "
                  f"{torch.cuda.memory_allocated() / 1e9:.2f} GB")

        return model, tokenizer

    except Exception as e:
        print(f" Failed to load {name}: {e}")
        return None, None


def _strip_code_fences(text):
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
    return text.replace("```", "")


def _balanced_json_objects(text):
    """Scan text for every top-level {...} substring with balanced braces,
    respecting quoted strings so a '}' inside a JSON string value (e.g. in
    the free-text "summary" field) doesn't end the object early. Used
    alongside the plain regex below rather than replacing it, since the
    four judge models differ in how strictly they follow the "return ONLY
    JSON" instruction."""
    objs = []
    depth = 0
    start = None
    in_string = False
    escape = False

    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    objs.append(text[start:i + 1])
                    start = None

    return objs


REQUIRED_RUBRIC_KEYS = [
    "correct_tests_score",
    "coverage_score",
    "solution_score",
    "clarity_score",
    "representativeness_score"
]

# Key-aliasing map carried over from the original per-judge scripts (the
# Mistral and DeepSeek judge pipelines each hand-tuned one of these after
# observing that specific judge use a slightly different key name than the
# rubric example asked for). Kept here as a single shared table rather than
# duplicated per script, and applied to every judge rather than guessed
# per-model, since any of the four architecturally-different judges could
# plausibly drift the same way.
RUBRIC_KEY_RENAME_MAP = {
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


def _apply_rubric_rename_map(obj):
    for old, new in RUBRIC_KEY_RENAME_MAP.items():
        if old in obj and new not in obj:
            obj[new] = obj[old]
    return obj


def _find_valid_rubric_object(text):
    """Try every extraction strategy against `text` and return the first
    parsed dict that - after key-aliasing - carries all five required
    rubric keys, or None."""

    cleaned = _strip_code_fences(text)

    candidates = []

    # Anchor pattern from the original DeepSeek-judge script: prioritise a
    # brace-delimited block that actually contains the
    # "representativeness_score" key, which helps when the response has
    # more than one brace-delimited chunk (e.g. a worked example echoed
    # before the real answer) and the real one isn't simply the last one.
    anchor = re.search(
        r'\{[\s\S]*"representativeness_score"[\s\S]*?\}', cleaned
    )
    if anchor:
        candidates.append(anchor.group(0))

    candidates += _balanced_json_objects(cleaned)
    # Keep the original non-greedy regex as a further pass too - belt and
    # braces (no pun intended) in case the balanced scanner is thrown off
    # by a genuinely unmatched brace somewhere in surrounding prose.
    candidates += re.findall(r"\{[\s\S]*?\}", cleaned)

    for m in reversed(candidates):
        try:
            obj = json.loads(m)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        obj = _apply_rubric_rename_map(obj)
        if all(k in obj for k in REQUIRED_RUBRIC_KEYS):
            return obj

    return None


def extract_json(text):
    """Returns a parsed, key-aliased rubric dict, or None if nothing
    parseable was found. Tries the raw text first, then - since the
    original Mistral-judge script needed this specifically - a
    single-quote-to-double-quote-normalised retry of the same text, for a
    judge that emits Python-dict-style output (single-quoted keys/strings)
    instead of valid JSON. The normalisation can also mangle a genuine
    apostrophe inside the free-text "summary" value (e.g. "the function's
    behaviour"); acceptable since this only runs once every double-quoted
    parse attempt above has already failed, and it's the five numeric
    scores this pipeline actually consumes, not the summary prose."""

    obj = _find_valid_rubric_object(text)
    if obj is not None:
        return obj

    requoted = text.replace("'", '"')
    return _find_valid_rubric_object(requoted)


def clamp_scores(raw_result):

    keys = REQUIRED_RUBRIC_KEYS

    # Build a clean output dict rather than mutating raw_result in place -
    # extract_json's rename map (RUBRIC_KEY_RENAME_MAP) can leave both the
    # judge's original field name (e.g. "correctness_score") and its
    # canonical alias ("correct_tests_score") sitting in the same dict.
    # Carrying the leftover alternate-name keys through to the final
    # per-task row would add inconsistent extra columns to the output
    # spreadsheet (present only for rows where that particular judge used
    # a non-standard key name) - so only the canonical keys, plus summary,
    # make it into the result this function returns.
    result = {}

    # Default representativeness_score BEFORE it's used in the mean below -
    # the original script called setdefault() after already summing the
    # keys, which only worked because extract_json's required-key check
    # guaranteed the key was present; this is now correct independent of
    # that.
    raw_result.setdefault("representativeness_score", 1)

    for k in keys:
        if k in raw_result:
            try:
                result[k] = int(raw_result[k])
            except Exception:
                result[k] = 1
            result[k] = max(1, min(5, result[k]))
        else:
            result[k] = 1

    result["benchmark_quality_score"] = round(
        sum(result[k] for k in keys) / len(keys),
        2
    )

    result["summary"] = raw_result.get("summary", "")

    return result


def build_chat_text(tokenizer, audit_prompt):
    messages = [
        {
            "role": "user",
            "content": audit_prompt
        }
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception as e:
        # A model/tokenizer-specific chat-template quirk shouldn't crash
        # the whole run - fall back to a plain instruction-style prompt so
        # this one judge still produces (probably lower-quality, but
        # present and clearly logged) results instead of nothing.
        print(f" apply_chat_template failed ({e}); using plain-text fallback")
        return audit_prompt + "\n\nAssistant:"


def audit_with_rubric(model, tokenizer, prompt, tests, solution, log_path, max_retries=2):

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

    chat_text = build_chat_text(tokenizer, audit_prompt)

    inputs = tokenizer(
        chat_text,
        return_tensors="pt",
        truncation=True,
        max_length=4096
    ).to(device)

    text = None
    last_error = None

    for attempt in range(max_retries + 1):
        try:
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
            break

        except RuntimeError as e:
            last_error = e
            if "out of memory" in str(e).lower():
                print(f" CUDA OOM during generate() "
                      f"(attempt {attempt + 1}/{max_retries + 1}): {e}")
                torch.cuda.empty_cache()
                time.sleep(5)
            else:
                raise

    if text is None:
        # Every retry OOM'd - record a clearly-flagged failure row instead
        # of crashing the run for the remaining tasks.
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\nGENERATION FAILED AFTER RETRIES (OOM)\n")
            f.write("-" * 50 + "\n")
            f.write(str(last_error))
            f.write("\n")
        return {
            "correct_tests_score": 1,
            "coverage_score": 1,
            "solution_score": 1,
            "clarity_score": 1,
            "representativeness_score": 1,
            "benchmark_quality_score": 1.0,
            "summary": "generation failed after retries (OOM)",
            "judge_error": "oom"
        }

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\nRAW JUDGE OUTPUT\n")
        f.write("-" * 50 + "\n")
        f.write(text)
        f.write("\n")

    try:
        result = extract_json(text)

        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\nEXTRACTED + PARSED RESULT (after key-aliasing)\n")
            f.write("-" * 50 + "\n")
            f.write(json.dumps(result, indent=2) if result else str(result))
            f.write("\n")

        if not result:
            raise ValueError("No JSON found")

        result = clamp_scores(result)
        result["judge_error"] = None
        return result

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
            "summary": "parsing failed",
            "judge_error": "parse_failed"
        }


def safe_audit(model, tokenizer, prompt, tests, solution, log_path, task_id, dataset_name):
    """Wraps audit_with_rubric so an unexpected exception on one sample
    (not just the OOM case audit_with_rubric already retries internally)
    can't take down the entire multi-hundred-task run."""
    try:
        return audit_with_rubric(model, tokenizer, prompt, tests, solution, log_path)
    except Exception as e:
        tb = traceback.format_exc()
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\nUNEXPECTED FAILURE on {dataset_name}/{task_id}\n")
            f.write("-" * 50 + "\n")
            f.write(tb)
            f.write("\n")
        print(f" Unexpected failure on {dataset_name}/{task_id}: {e}")
        torch.cuda.empty_cache()
        return {
            "correct_tests_score": 1,
            "coverage_score": 1,
            "solution_score": 1,
            "clarity_score": 1,
            "representativeness_score": 1,
            "benchmark_quality_score": 1.0,
            "summary": f"unexpected failure: {e}",
            "judge_error": "exception"
        }


def get_func_name(prompt):
    match = re.search(r"def\s+([a-zA-Z_]\w*)\s*\(", prompt)
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# Judge model to run.
# ---------------------------------------------------------------------------
judge_models_local = [
    "Qwen/Qwen2.5-Coder-7B-Instruct",
]


def checkpoint_path(judge_model_name):
    return f"{OUTPUT_DIR}/checkpoint_{judge_model_name.replace('/', '_')}.jsonl"


def load_checkpoint(judge_model_name):
    """Resume support: returns (rows_already_scored, done_keys) where
    done_keys is a set of (dataset, task_id) pairs already present in the
    checkpoint file, so run_audit() can skip re-scoring them."""
    path = checkpoint_path(judge_model_name)
    rows = []
    done = set()

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    # A trailing partially-written line from a mid-write
                    # crash - skip it rather than failing the whole resume.
                    continue
                rows.append(row)
                done.add((row["dataset"], row["task_id"]))

    return rows, done


def append_checkpoint(judge_model_name, row):
    path = checkpoint_path(judge_model_name)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


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

    # Sorted, not a raw set: keeps task order identical across repeated
    # runs and across the four different judge scripts, which matters once
    # you're diffing/resuming runs and eyeballing the audit logs side by
    # side.
    common_keys = sorted(common_keys)

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

        existing_rows, done_keys = load_checkpoint(judge_model_name)
        resuming = bool(existing_rows)

        with open(log_path, "a" if resuming else "w", encoding="utf-8") as f:
            if resuming:
                f.write(
                    f"\nRESUMING RUN: {datetime.now(timezone.utc).isoformat()} "
                    f"({len(existing_rows)} rows already scored from a prior "
                    f"attempt - skipping those)\n"
                )
                f.write("=" * 100 + "\n")
            else:
                f.write(f"JUDGE MODEL: {judge_model_name}\n")
                f.write(f"RUN STARTED (UTC): {datetime.now(timezone.utc).isoformat()}\n")
                f.write(
                    f"transformers={transformers.__version__} "
                    f"torch={torch.__version__} "
                    f"datasets={datasets_pkg.__version__} seed={SEED}\n"
                )
                f.write("=" * 100 + "\n")

        if resuming:
            print(
                f" Resuming {judge_model_name}: {len(existing_rows)} rows "
                f"already scored ({2 * len(common_keys) - len(existing_rows)} "
                f"remaining)"
            )

        results = list(existing_rows)

        for key in tqdm(common_keys, desc=f"Evaluating {judge_model_name}"):

            he_item = he_map[key]
            hn_item = hn_map[key]

            if ("HumanEval", key) not in done_keys:

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

                res_he = safe_audit(
                    model,
                    tokenizer,
                    he_item["prompt"],
                    he_item.get("test", ""),
                    he_item.get("canonical_solution", ""),
                    log_path,
                    key,
                    "HumanEval"
                )

                row = {
                    "judge_model": judge_model_name,
                    "dataset": "HumanEval",
                    "task_id": key,
                    **res_he
                }
                results.append(row)
                append_checkpoint(judge_model_name, row)

            if ("HumanEvalNext", key) not in done_keys:

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

                res_hn = safe_audit(
                    model,
                    tokenizer,
                    hn_item["prompt"],
                    hn_item.get("test", ""),
                    hn_item.get("canonical_solution", ""),
                    log_path,
                    key,
                    "HumanEvalNext"
                )

                row = {
                    "judge_model": judge_model_name,
                    "dataset": "HumanEvalNext",
                    "task_id": key,
                    **res_hn
                }
                results.append(row)
                append_checkpoint(judge_model_name, row)

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

        error_counts = (
            df["judge_error"].fillna("none").value_counts().to_dict()
            if "judge_error" in df.columns else {}
        )

        metadata = {
            "judge_model": judge_model_name,
            "quantization": "4-bit NF4 (bnb_4bit_compute_dtype=float16, "
                             "bnb_4bit_use_double_quant=True)",
            "matched_tasks": len(common_keys),
            "total_audits": int(len(df)),
            "judge_error_counts": error_counts,
            "seed": SEED,
            "max_new_tokens": 400,
            "do_sample": False,
            "context_truncation_tokens": 4096,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "datasets_version": datasets_pkg.__version__,
            "resumed_from_checkpoint": resuming,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }

        metadata_file = (
            f"{OUTPUT_DIR}/"
            f"run_metadata_{judge_model_name.replace('/', '_')}.json"
        )
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f" Saved run metadata: {metadata_file}")
        if error_counts:
            print(f" Judge error breakdown: {error_counts}")

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
