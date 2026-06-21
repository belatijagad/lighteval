"""
name:
Gsm8K

dataset:
openai/gsm8k

abstract:
GSM8K is a dataset of 8,000+ high-quality, single-step arithmetic word problems.

languages:
english

tags:
math, reasoning

paper:
https://arxiv.org/abs/2110.14168
"""

from inspect_ai.dataset import Sample
from inspect_ai.solver import generate, prompt_template

from lighteval.metrics.metrics import Metrics, math_scorer
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc


GSM8K_SYSTEM_PROMPT = """
Solve the following math problem step by step. You must write your reasoning inside explicit `<think>` tags using this format:
<think>
your reasoning here
</think>
After the closing `</think>` tag, give the final answer.
For closed-form answers, put the result in `\\boxed{}`.
The last line of your response must be exactly: Answer: \\boxed{answer}
""".strip()


# setup for problem + instructions for providing answer
MATH_PROMPT_TEMPLATE = f"""
{GSM8K_SYSTEM_PROMPT}

{{prompt}}
""".strip()


def record_to_sample(record):
    DELIM = "####"
    input = record["question"]
    answer = record["answer"].split(DELIM)
    target = answer.pop().strip()
    reasoning = DELIM.join(answer)
    return Sample(input=input, target=target, metadata={"reasoning": reasoning.strip()})


def sample_to_fewshot(sample):
    return f"{sample.input}\n\n<think>\n{sample.metadata['reasoning']}\n</think>\nAnswer: \\boxed{{{sample.target}}}"


def gsm8k_prompt(line, task_name: str = None):
    target = line["answer"].split("####")[-1].strip()
    return Doc(
        task_name=task_name,
        query=line["question"],
        choices=[f"\\boxed{{{target}}}"],
        gold_index=0,
        instruction=f"{GSM8K_SYSTEM_PROMPT}\n\n",
    )


gsm8k = LightevalTaskConfig(
    name="gsm8k",
    prompt_function=gsm8k_prompt,
    sample_fields=record_to_sample,
    sample_to_fewshot=sample_to_fewshot,
    solver=[prompt_template(MATH_PROMPT_TEMPLATE), generate(cache=True)],
    scorer=math_scorer(),
    hf_repo="openai/gsm8k",
    hf_subset="main",
    hf_avail_splits=["train", "test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select="random_sampling_from_train",
    generation_size=256,
    metrics=[
        Metrics.expr_gold_metric,
    ],
    stop_sequence=["Question:"],
    version=1,
)


def make_gsm8k_avg_task(n: int) -> LightevalTaskConfig:
    return LightevalTaskConfig(
        name=f"gsm8k_avg_{n}",
        prompt_function=gsm8k_prompt,
        sample_fields=record_to_sample,
        sample_to_fewshot=sample_to_fewshot,
        solver=[prompt_template(MATH_PROMPT_TEMPLATE), generate(cache=True)],
        scorer=math_scorer(),
        hf_repo="openai/gsm8k",
        hf_subset="main",
        hf_avail_splits=["train", "test"],
        evaluation_splits=["test"],
        few_shots_split=None,
        few_shots_select="random_sampling_from_train",
        generation_size=256,
        metrics=[
            Metrics.avg_at_n_math(sample_params={"n": n}),
        ],
        stop_sequence=["Question:"],
        version=1,
    )


gsm8k_avg_16 = make_gsm8k_avg_task(16)
gsm8k_avg_32 = make_gsm8k_avg_task(32)
gsm8k_avg_64 = make_gsm8k_avg_task(64)


TASKS_TABLE = [
    gsm8k,
    gsm8k_avg_16,
    gsm8k_avg_32,
    gsm8k_avg_64,
]
