# Copyright 2024 The DPriv-Bench Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import time
from typing import Any, Callable, Dict, Iterable, List, Optional

from openai import OpenAI, RateLimitError

try:
    import torch
except ImportError:
    torch = None

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None

try:
    from anthropic import Anthropic
except ImportError:  # pragma: no cover - optional dependency
    Anthropic = None  # type: ignore

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover - optional dependency
    genai = None  # type: ignore
    genai_types = None  # type: ignore


SUPPORTED_MODELS = {
    "gpt-4o",
    "gpt-5-high",   # GPT-5 with reasoning effort High
    "gpt-5-minimal", # GPT-5 with reasoning effort Minimal
    "gpt-5-low", # GPT-5 with reasoning effort Low
    "gemini-flash", # Gemini-2.5-flash 
    "gemini-pro", # Gemini-2.5-pro
    "gemini-3", # # gemini-3-pro-preview is not supported, it automatically redirect to gemini-3.1-pro-preview after March 26, 2026.
    "claude-sonnet", # Claude-sonnet-4-20250514
    "claude-opus", # Claude-opus-4-20250514
    "DeepSeek-V3.1",
    "DeepSeek-R1",
    "Goedel-Prover",
    "qwen3-30b-think",
    "qwen3-30b-instruct",
}

_IF_Batch_Response = {
    "gpt-4o": False,
    "gpt-5-high": False,
    "gpt-5-minimal": False,
    "gemini-flash": False,
    "gemini-pro": False,
    "gemini-3": False,
    "claude-sonnet": False,
    "claude-opus": False,
    "DeepSeek-V3.1": False,
    "DeepSeek-R1": False,
    "Goedel-Prover": True,
    "qwen3-30b-think": True,
    "qwen3-30b-instruct": True,
}


class _BatchResponseLookup(dict):
    """Dict subclass that returns sensible defaults for unknown models.

    - ``True``  (batch / vLLM) when ``--download_path`` was provided,
      i.e. the model is treated as a local model.
    - ``False`` (per-response / API) otherwise.

    Usage is identical to a plain dict:  ``IF_Batch_Response[model]``.
    """

    def __missing__(self, model: str) -> bool:  # type: ignore[override]
        # Custom local models (download_dir supplied) are batch; API models are not.
        # We cannot know the download_dir here, so we conservatively return False
        # (per-response) and let get_responses() handle the actual routing.
        return False


IF_Batch_Response = _BatchResponseLookup(_IF_Batch_Response)


ResponseCallback = Optional[Callable[[int, str], None]]
BatchResponseCallback = Optional[Callable[[List[str]], None]]

MODEL_TO_CHECKPOINT = {"qwen3-30b-think": "Qwen/Qwen3-30B-A3B-Thinking-2507",
                       "qwen3-30b-instruct": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                       "Goedel-Prover": "Goedel-LM/Goedel-Prover-V2-32B",
                       "gemini-3": "gemini-3-pro-preview",
                       "gemini-pro": "gemini-2.5-pro",
                       "gemini-flash": "gemini-2.5-flash",
                       "claude-sonnet": "claude-sonnet-4-20250514",
                       "claude-opus": "claude-opus-4-20250514",
                       "DeepSeek-V3.1": "deepseek-chat",
                       "DeepSeek-R1": "deepseek/deepseek-r1-0528",
                       "gpt-4o": "gpt-4o",
                       "gpt-5-high": "gpt-5-high",
                       "gpt-5-minimal": "gpt-5-minimal",
                       "gpt-5-low": "gpt-5-low",
                       }

def load_json_or_jsonl(path: str) -> List[Any]:
    """Load a file that is either a JSON array or a JSONL (one object per line) file.

    Args:
        path: Path to the JSON or JSONL file.

    Returns:
        A list of parsed objects. Returns an empty list if the file is empty.
        Single top-level JSON objects are wrapped in a list.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
        return [parsed]
    except json.JSONDecodeError:
        result: List[Any] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            result.append(json.loads(line))
        return result


_COT_ANSWER_SUFFIX = (
    " Please provide some explanations or deriviations first and then provide your final answer `yes` or `no` inside a Latex boxed format `\\boxed{}`."
)

# Synthetic example (not taken from benchmark items) for category-1 one-shot prompting.
_CATEGORY_1_ONE_SHOT_EXAMPLE = r"""Example Question:
For each item $j\in F$, define a score $s_j(D)$ with sensitivity 1. Consider the mechanism that samples independent noise $Z_j$ ~ Laplace(2/$\epsilon$) for each $j$ and outputs
$\argmax_j (s_j(D) + Z_j)$.
Claim: This mechanism satisfies $\epsilon$-differential privacy."""



_CATEGORY_1_ONE_SHOT_EXAMPLE_NEG = r"""Example Question:
Suppose query function $s$ has sensitivity 1. Consider the mechanism that samples independent noise $Z$ ~ Laplace(1/$\epsilon$) and outputs
$s(D) + Z$, where $D$ is the dataset.
Claim: This mechanism satisfies $\epsilon$-differential privacy."""


def prompt_construction(problem_description: str, qa_prompt_mode: str) -> str:
    """Wrap a problem statement with an answering instruction (shared default: CoT).

    Used by Category 2 and other tasks. Category 1 should use
    :func:`prompt_construction_category_1` so ``one_shot`` is available.

    Args:
        problem_description: The raw question text (may contain LaTeX).
        qa_prompt_mode: Only ``"cot"`` is supported here.

    Returns:
        The full prompt string to send to the model.

    Raises:
        NotImplementedError: If ``qa_prompt_mode`` is not ``"cot"``.
    """
    if qa_prompt_mode == "cot":
        return problem_description + _COT_ANSWER_SUFFIX
    raise NotImplementedError(f"Unsupported qa_prompt_mode: {qa_prompt_mode}")


def prompt_construction_category_1(problem_description: str, qa_prompt_mode: str) -> str:
    """Build the user prompt for Category 1 (mechanism-level DP judge).

    Args:
        problem_description: The raw question text from the benchmark JSON.
        qa_prompt_mode: ``"cot"`` (zero-shot) or ``"one_shot"`` (one worked
            example before the target question).

    Returns:
        Full prompt string passed to :func:`get_responses`.

    Raises:
        NotImplementedError: If ``qa_prompt_mode`` is not recognized.
    """
    if qa_prompt_mode == "cot":
        return problem_description + _COT_ANSWER_SUFFIX
    if qa_prompt_mode == "one_shot":
        return (
            "Below is an example of the question."
            + "\n\n" 
            + _CATEGORY_1_ONE_SHOT_EXAMPLE
            + "\n\n" 
            + "Now answer the following question."
            + "\n\n"
            + problem_description
            + "\n\n"
            + _COT_ANSWER_SUFFIX
        )
    if qa_prompt_mode == "one_shot_neg":
        return (
            "Below is an example of the question."
            + "\n\n" 
            + _CATEGORY_1_ONE_SHOT_EXAMPLE_NEG
            + "\n\n" 
            + "Now answer the following question."
            + "\n\n"
            + problem_description
            + "\n\n"
            + _COT_ANSWER_SUFFIX
        )
    raise NotImplementedError(f"Unsupported category-1 qa_prompt_mode: {qa_prompt_mode}")



def _retry_with_backoff(
    fn,
    *,
    max_retries: int = 5,
    initial_delay: float = 1.0,
    retriable_exceptions: Iterable[type] = (RateLimitError,),
):
    """Call ``fn()`` and retry on specified exceptions with exponential backoff.

    Args:
        fn: Zero-argument callable to invoke.
        max_retries: Total number of attempts before re-raising the exception.
        initial_delay: Sleep duration (seconds) before the first retry.
            Doubles on each subsequent retry, capped at 30 seconds.
        retriable_exceptions: Exception types that trigger a retry.
            Defaults to ``(RateLimitError,)``.

    Returns:
        The return value of ``fn()`` on success.

    Raises:
        The last caught exception if all retries are exhausted.
    """
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return fn()
        except tuple(retriable_exceptions):
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 30)


def _get_openai_client(base_url: Optional[str] = None, api_key_env: str = "OPENAI_API_KEY") -> OpenAI:
    """Create an OpenAI-compatible client, reading the API key from the environment.

    Args:
        base_url: Optional custom base URL (used for DeepSeek and other
            OpenAI-compatible endpoints). If ``None`` or empty, uses the
            default OpenAI endpoint.
        api_key_env: Name of the environment variable that holds the API key.

    Returns:
        A configured ``OpenAI`` client instance.

    Raises:
        ValueError: If the environment variable ``api_key_env`` is not set.
    """
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ValueError(f"Missing required environment variable: {api_key_env}")
    if base_url:
        return OpenAI(base_url=base_url, api_key=api_key)
    return OpenAI(api_key=api_key)


def get_responses_gpt(prompt_list: List[str], model: str, on_response: ResponseCallback = None) -> List[str]:
    """Query an OpenAI GPT model for each prompt in ``prompt_list``.

    ``gpt-4o`` uses the Chat Completions API. ``gpt-5-*`` aliases use the
    Responses API with the reasoning effort encoded in the alias suffix
    (e.g. ``"gpt-5-high"`` → effort ``"high"``).

    Args:
        prompt_list: Ordered list of prompt strings to send to the model.
        model: Model alias from ``SUPPORTED_MODELS`` (e.g. ``"gpt-4o"``).
        on_response: Optional callback invoked after each response as
            ``on_response(index, response_text)``.

    Returns:
        List of response strings in the same order as ``prompt_list``.

    Raises:
        ValueError: If ``OPENAI_API_KEY`` is not set in the environment.
    """
    client = _get_openai_client()
    responses: List[str] = []
    for idx, prompt in enumerate(prompt_list):
        if model == "gpt-4o":
            completion = _retry_with_backoff(
                lambda: client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                )
            )
            response_text = completion.choices[0].message.content or ""
            responses.append(response_text)
            if on_response is not None:
                on_response(idx, response_text)
        else:
            completion = _retry_with_backoff(
                lambda: client.responses.create(
                    model="gpt-5",
                    input=prompt,
                    reasoning={"effort": model.split("-")[-1]},
                )
            )
            response_text = getattr(completion, "output_text", "") or ""
            responses.append(response_text)
            if on_response is not None:
                on_response(idx, response_text)

    return responses


def get_responses_gpt_restricted_retrieval(prompt_list: List[str], model: str, on_response: ResponseCallback = None) -> List[str]:
    """Query an OpenAI GPT model for each prompt in ``prompt_list``.

    ``gpt-4o`` uses the Chat Completions API. ``gpt-5-*`` aliases use the
    Responses API with the reasoning effort encoded in the alias suffix
    (e.g. ``"gpt-5-high"`` → effort ``"high"``).

    Args:
        prompt_list: Ordered list of prompt strings to send to the model.
        model: Model alias from ``SUPPORTED_MODELS`` (e.g. ``"gpt-4o"``).
        on_response: Optional callback invoked after each response as
            ``on_response(index, response_text)``.

    Returns:
        List of response strings in the same order as ``prompt_list``.

    Raises:
        ValueError: If ``OPENAI_API_KEY`` is not set in the environment.
    """
    client = _get_openai_client()
    responses: List[str] = []
    for idx, prompt in enumerate(prompt_list):
        if model == "gpt-4o":
            completion = _retry_with_backoff(
                lambda: client.chat.completions.create(
                    model=model,
                    tools=[
                            {
                                "type": "file_search",
                                "vector_store_ids": ["enter your vector store id here"]
                            }
                    ],
                    tool_choice={"type": "file_search"},
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                )
            )
            response_text = completion.choices[0].message.content or ""
            responses.append(response_text)
            if on_response is not None:
                on_response(idx, response_text)
        else:
            completion = _retry_with_backoff(
                lambda: client.responses.create(
                    model="gpt-5.2",
                    tools=[
                            {
                                "type": "file_search",
                                "vector_store_ids": ["enter your vector store id here"]
                            }],
                    tool_choice={"type": "file_search"},
                    input=prompt,
                    reasoning={"effort": model.split("-")[-1]},
                )
            )
            response_text = getattr(completion, "output_text", "") or ""
            responses.append(response_text)
            if on_response is not None:
                on_response(idx, response_text)

    return responses





def get_responses_gemini_restricted_retrieval(prompt_list: List[str], model: str, on_response: ResponseCallback = None) -> List[str]:
    """Query a Google Gemini model for each prompt in ``prompt_list``.

    Args:
        prompt_list: Ordered list of prompt strings to send to the model.
        model: Model alias from ``SUPPORTED_MODELS``
            (``"gemini-flash"``, ``"gemini-pro"``, or ``"gemini-3"``).
        on_response: Optional callback invoked after each response as
            ``on_response(index, response_text)``.

    Returns:
        List of response strings in the same order as ``prompt_list``.

    Raises:
        ImportError: If the ``google-genai`` package is not installed.
        ValueError: If ``GOOGLE_API_KEY`` is not set in the environment.
        NotImplementedError: If ``model`` is not a recognized Gemini alias.
    """
    if genai is None or genai_types is None:
        raise ImportError("google-genai is not installed. Install it to use Gemini models.")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY environment variable.")

    model_name_map = {
        "gemini-flash": "gemini-2.5-flash",
        "gemini-pro": "gemini-2.5-pro",
        "gemini-3": "gemini-3.1-pro-preview"  # gemini-3-pro-preview is not supported, it automatically redirect to gemini-3.1-pro-preview after March 26, 2026.
    }
    if model not in model_name_map:
        raise NotImplementedError(f"Unsupported Gemini model alias: {model}")

    client = genai.Client(api_key=api_key)
    # File Search must be passed inside GenerateContentConfig (not as generate_content(tools=...)).
    file_search_store_names = [
        os.getenv(
            "GEMINI_FILE_SEARCH_STORE",
            "enter your file search store id here",
        )
    ]
    response_list: List[str] = []
    n = len(prompt_list)
    for idx, prompt in enumerate(prompt_list):
        # Progress bar only advances after each full response; File Search + long prompts can take minutes per item.
        print(f"Gemini restricted-retrieval: request {idx + 1}/{n} (waiting for API)...", flush=True)
        response = _retry_with_backoff(
            lambda p=prompt: client.models.generate_content(
                model=model_name_map[model],
                contents=p,
                config=genai_types.GenerateContentConfig(
                    temperature=1,
                    tools=[
                        genai_types.Tool(
                            file_search=genai_types.FileSearch(
                                file_search_store_names=file_search_store_names
                            )
                        )
                    ],
                ),
            ),
            retriable_exceptions=(Exception,),
        )
        parts = response.candidates[0].content.parts if response.candidates else []
        text = "".join(getattr(p, "text", "") for p in parts if getattr(p, "text", None))
        response_list.append(text)
        if on_response is not None:
            on_response(idx, text)
    return response_list





























def get_responses_gemini(prompt_list: List[str], model: str, on_response: ResponseCallback = None) -> List[str]:
    """Query a Google Gemini model for each prompt in ``prompt_list``.

    Args:
        prompt_list: Ordered list of prompt strings to send to the model.
        model: Model alias from ``SUPPORTED_MODELS``
            (``"gemini-flash"``, ``"gemini-pro"``, or ``"gemini-3"``).
        on_response: Optional callback invoked after each response as
            ``on_response(index, response_text)``.

    Returns:
        List of response strings in the same order as ``prompt_list``.

    Raises:
        ImportError: If the ``google-genai`` package is not installed.
        ValueError: If ``GOOGLE_API_KEY`` is not set in the environment.
        NotImplementedError: If ``model`` is not a recognized Gemini alias.
    """
    if genai is None or genai_types is None:
        raise ImportError("google-genai is not installed. Install it to use Gemini models.")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY environment variable.")

    model_name_map = {
        "gemini-flash": "gemini-2.5-flash",
        "gemini-pro": "gemini-2.5-pro",
        "gemini-3": "gemini-3-pro-preview"
    }
    if model not in model_name_map:
        raise NotImplementedError(f"Unsupported Gemini model alias: {model}")

    client = genai.Client(api_key=api_key)
    response_list: List[str] = []
    for idx, prompt in enumerate(prompt_list):
        response = _retry_with_backoff(
            lambda: client.models.generate_content(
                model=model_name_map[model],
                contents=prompt,
                config=genai_types.GenerateContentConfig(temperature=1),
            ),
            retriable_exceptions=(Exception,),
        )
        parts = response.candidates[0].content.parts if response.candidates else []
        text = "".join(getattr(p, "text", "") for p in parts if getattr(p, "text", None))
        response_list.append(text)
        if on_response is not None:
            on_response(idx, text)
    return response_list


def get_responses_claude(prompt_list: List[str], model: str, on_response: ResponseCallback = None) -> List[str]:
    """Query an Anthropic Claude model for each prompt in ``prompt_list``.

    Args:
        prompt_list: Ordered list of prompt strings to send to the model.
        model: Model alias from ``SUPPORTED_MODELS``
            (``"claude-sonnet"`` or ``"claude-opus"``).
        on_response: Optional callback invoked after each response as
            ``on_response(index, response_text)``.

    Returns:
        List of response strings in the same order as ``prompt_list``.

    Raises:
        ImportError: If the ``anthropic`` package is not installed.
        ValueError: If ``ANTHROPIC_API_KEY`` is not set in the environment.
        NotImplementedError: If ``model`` is not a recognized Claude alias.
    """
    if Anthropic is None:
        raise ImportError("anthropic package is not installed.")
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError("Missing required environment variable: ANTHROPIC_API_KEY")

    model_map = {
        "claude-sonnet": "claude-sonnet-4-20250514",
        "claude-opus": "claude-opus-4-20250514",
    }
    if model not in model_map:
        raise NotImplementedError(f"Unsupported Claude model alias: {model}")

    client = Anthropic()
    response_list: List[str] = []
    for idx, prompt in enumerate(prompt_list):
        kwargs: Dict[str, Any] = {
            "model": model_map[model],
            "max_tokens": 2000,
            "messages": [{"role": "user", "content": prompt}],
        }
        response = _retry_with_backoff(lambda: client.messages.create(**kwargs))
        response_text = response.content[0].text if response.content else ""
        response_list.append(response_text)
        if on_response is not None:
            on_response(idx, response_text)
    return response_list


def get_responses_deepseek(prompt_list: List[str], model: str, on_response: ResponseCallback = None) -> List[str]:
    """Query a DeepSeek model for each prompt in ``prompt_list``.

    Uses an OpenAI-compatible client pointed at the DeepSeek endpoint.

    Args:
        prompt_list: Ordered list of prompt strings to send to the model.
        model: Model alias from ``SUPPORTED_MODELS``
            (``"DeepSeek-V3.1"`` or ``"DeepSeek-R1"``).
        on_response: Optional callback invoked after each response as
            ``on_response(index, response_text)``.

    Returns:
        List of response strings in the same order as ``prompt_list``.

    Raises:
        ValueError: If ``DEEPSEEK_BASE_URL`` or ``DEEPSEEK_API_KEY`` is not
            set in the environment.
        NotImplementedError: If ``model`` is not a recognized DeepSeek alias.
    """
    DEEPSEEK_URL = os.getenv("DEEPSEEK_BASE_URL", "")
    if not DEEPSEEK_URL:
        raise ValueError("DEEPSEEK_BASE_URL environment variable is not set.")

    model_map = {
        "DeepSeek-V3.1": "deepseek-chat",
        "DeepSeek-R1": "deepseek/deepseek-r1-0528",
    }
    if model not in model_map:
        raise NotImplementedError(f"Unsupported DeepSeek model alias: {model}")

    client = _get_openai_client(base_url=DEEPSEEK_URL, api_key_env="DEEPSEEK_API_KEY")
    response_list: List[str] = []
    for idx, prompt in enumerate(prompt_list):
        completion = _retry_with_backoff(
            lambda: client.chat.completions.create(
                model=model_map[model],
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=8192,
            )
        )
        response_text = completion.choices[0].message.content or ""
        response_list.append(response_text)
        if on_response is not None:
            on_response(idx, response_text)
    return response_list



def get_responses_custom_api(prompt_list: List[str], model: str, on_response: ResponseCallback = None) -> List[str]:
    """Query any OpenAI-compatible API endpoint with a custom model.

    This is a template for adding new API-based models without modifying
    ``SUPPORTED_MODELS``.  Configure the endpoint via environment variables:

    - ``CUSTOM_BASE_URL`` — base URL of the OpenAI-compatible endpoint
      (e.g. ``"https://api.example.com/v1"``).  **Required.**
    - ``CUSTOM_API_KEY`` — API key for that endpoint.  Falls back to
      ``"EMPTY"`` if not set (useful for local servers that require a
      placeholder key).

    The ``model`` string is forwarded verbatim to the endpoint as the
    ``model`` field, so set it to the exact checkpoint/model-ID the
    server expects (e.g. ``"meta-llama/Llama-3-70b-instruct"``).

    Args:
        prompt_list: Ordered list of prompt strings to send to the model.
        model: Model identifier forwarded to the API endpoint.
        on_response: Optional callback invoked after each response as
            ``on_response(index, response_text)``.

    Returns:
        List of response strings in the same order as ``prompt_list``.

    Raises:
        ValueError: If ``CUSTOM_BASE_URL`` is not set.

    Example::

        import os
        os.environ["CUSTOM_BASE_URL"] = "https://api.example.com/v1"
        os.environ["CUSTOM_API_KEY"] = "sk-..."
        responses = get_responses_custom_api(prompts, model="my-model-id")
    """
    base_url = os.getenv("CUSTOM_BASE_URL", "")
    if not base_url:
        raise ValueError(
            "CUSTOM_BASE_URL is not set. "
            "Export it to the OpenAI-compatible endpoint base URL before calling get_responses_custom_api()."
        )
    api_key = os.getenv("CUSTOM_API_KEY", "EMPTY")
    client = OpenAI(base_url=base_url, api_key=api_key)

    responses: List[str] = []
    for idx, prompt in enumerate(prompt_list):
        completion = _retry_with_backoff(
            lambda: client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=4096,
            )
        )
        response_text = completion.choices[0].message.content or ""
        responses.append(response_text)
        if on_response is not None:
            on_response(idx, response_text)
    return responses


def get_responses_custom_local(
    prompt_list: List[str],
    model: str,
    seed: int,
    download_dir: str,
    on_response: BatchResponseCallback = None,
) -> List[str]:
    """Run prompts through a custom local model via vLLM.

    This is a template for adding new local open-source models without
    modifying ``SUPPORTED_MODELS``.  The ``model`` string is used directly
    as the HuggingFace checkpoint path (e.g.
    ``"meta-llama/Llama-3-70B-Instruct"``), so pass the full repo ID or the
    absolute path to a locally downloaded checkpoint.

    Args:
        prompt_list: Ordered list of prompt strings.
        model: HuggingFace checkpoint ID or local path used as both the
            model and tokenizer identifier.
        seed: Random seed passed to vLLM for reproducibility.
        download_dir: Directory where model weights are cached / downloaded.
        on_response: Optional callback invoked once with all response strings
            as ``on_response(responses=List[str])`` after generation completes.

    Returns:
        List of response strings in the same order as ``prompt_list``.

    Raises:
        ImportError: If ``vllm`` or ``torch`` is not installed.

    Example::

        responses = get_responses_custom_local(
            prompts,
            model="meta-llama/Llama-3-70B-Instruct",
            seed=0,
            download_dir="/path/to/model/cache",
        )
    """
    if LLM is None or SamplingParams is None:
        raise ImportError("vllm is not installed. Install it to use local models.")
    if torch is None:
        raise ImportError("torch is not installed. Install it to use local models.")

    checkpoint = model  # treat the model alias as the HuggingFace checkpoint directly
    vllm_model = LLM(
        checkpoint,
        tokenizer=checkpoint,
        dtype="bfloat16",
        tensor_parallel_size=torch.cuda.device_count(),
        seed=seed,
        trust_remote_code=True,
        max_model_len=4096,
        max_num_seqs=5,
        download_dir=download_dir,
    )
    tokenizer = vllm_model.get_tokenizer()
    messages = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        for prompt in prompt_list
    ]
    formatted = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages
    ]
    sampling_params = SamplingParams(temperature=1, max_tokens=4096)
    outputs = vllm_model.generate(prompts=formatted, sampling_params=sampling_params)
    responses = [output.outputs[0].text for output in outputs]
    if on_response is not None:
        on_response(responses=responses)
    return responses


def get_responses_open_source(prompt_list: List[str], model: str, seed: int, download_dir: str, on_response: BatchResponseCallback = None) -> List[str]:
    """Run all prompts through a locally hosted open-source model via vLLM.

    Loads the model onto all available GPUs, applies the tokenizer's chat
    template to each prompt, and generates responses in a single batched call.

    Args:
        prompt_list: Ordered list of prompt strings.
        model: Model alias from ``SUPPORTED_MODELS``
            (e.g. ``"Goedel-Prover"``, ``"qwen3-30b-think"``).
        seed: Random seed passed to vLLM for reproducibility.
        download_dir: Local directory where model weights are cached.
        on_response: Optional callback invoked once with all response strings
            as ``on_response(responses=List[str])`` after generation completes.

    Returns:
        List of response strings in the same order as ``prompt_list``.

    Raises:
        ImportError: If ``vllm`` or ``torch`` is not installed.
    """
    checkpoint = MODEL_TO_CHECKPOINT[model]
    vllm_model = LLM(checkpoint,
                     tokenizer=checkpoint,
                     dtype="bfloat16", 
                     tensor_parallel_size=torch.cuda.device_count(), 
                     seed=seed, 
                     trust_remote_code=True,
                     max_model_len=4096,
                     max_num_seqs=5,
                     download_dir=download_dir)
    tokenizer = vllm_model.get_tokenizer()
    
    messages = [[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ] for prompt in prompt_list]
    prompt_list = [tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True,
    ) for message in messages]

    sampling_params = SamplingParams(temperature=1, max_tokens=4096)
    outputs = vllm_model.generate(prompts=prompt_list, 
                                    sampling_params=sampling_params)
    responses = [output.outputs[0].text for output in outputs]
    if on_response is not None:
        on_response(responses = responses)
    return responses
    



def get_responses(
    prompt_list: List[str],
    model: str,
    on_response: ResponseCallback = None,
    seed: Optional[int] = None,
    download_dir: Optional[str] = None,
) -> List[str]:
    """Dispatch a list of prompts to the appropriate model backend.

    Routes to ``get_responses_gpt``, ``get_responses_gemini``,
    ``get_responses_claude``, ``get_responses_deepseek``, or
    ``get_responses_open_source`` based on the model alias prefix.

    For API-based models, ``on_response(index, text)`` is called after each
    individual response. For vLLM (batch) models, ``on_response(responses)``
    is called once with all responses.

    Args:
        prompt_list: Ordered list of prompt strings.
        model: Model alias. Known aliases from ``SUPPORTED_MODELS`` are
            dispatched automatically.  Unknown models fall back to
            ``get_responses_custom_local`` (when ``download_dir`` is set) or
            ``get_responses_custom_api`` (when ``CUSTOM_BASE_URL`` is set).
        on_response: Optional callback for incremental response handling.
            Signature depends on model type (see ``ResponseCallback`` vs
            ``BatchResponseCallback``).
        seed: Random seed forwarded to vLLM for open-source/custom-local models.
            Ignored for API-based models.
        download_dir: Model weight cache directory forwarded to vLLM.
            Also used to distinguish custom local vs custom API models.

    Returns:
        List of response strings in the same order as ``prompt_list``.

    Raises:
        NotImplementedError: If ``model`` is not in ``SUPPORTED_MODELS`` and
            neither ``download_dir`` nor ``CUSTOM_BASE_URL`` is configured.
    """
    if model.startswith("gpt"):
        return get_responses_gpt(prompt_list, model, on_response=on_response)
    if model.startswith("gemini"):
        return get_responses_gemini(prompt_list, model, on_response=on_response)
    if model.startswith("claude"):
        return get_responses_claude(prompt_list, model, on_response=on_response)
    if model.startswith("DeepSeek"):
        return get_responses_deepseek(prompt_list, model, on_response=on_response)
    if model in SUPPORTED_MODELS:
        return get_responses_open_source(prompt_list, model, seed, download_dir, on_response=on_response)

    # --- Fallback for unsupported / custom models ---
    # Route to a local vLLM backend when a download directory is supplied,
    # otherwise try an OpenAI-compatible API endpoint via CUSTOM_BASE_URL.
    if download_dir is not None:
        print(
            f"INFO: '{model}' is not in SUPPORTED_MODELS. "
            "Treating it as a custom local model (vLLM). "
            "The model string will be used as the HuggingFace checkpoint ID."
        )
        return get_responses_custom_local(prompt_list, model, seed, download_dir, on_response=on_response)

    custom_base_url = os.getenv("CUSTOM_BASE_URL", "")
    if custom_base_url:
        print(
            f"INFO: '{model}' is not in SUPPORTED_MODELS. "
            f"Routing to custom API endpoint: {custom_base_url}"
        )
        return get_responses_custom_api(prompt_list, model, on_response=on_response)

    raise NotImplementedError(
        f"Model '{model}' is not in SUPPORTED_MODELS and no custom backend was detected. "
        "To use a custom API model, set the CUSTOM_BASE_URL environment variable. "
        "To use a custom local model via vLLM, pass --download_path to the run script."
    )
    

def _try_extract_yes_no(response: str) -> Optional[str]:
    """Extract 'yes' or 'no' from response: check \\boxed{...} content first, then full text."""
    if not response:
        return None
    parts = response.split("boxed{")
    if len(parts) >= 2:
        boxed = parts[-1].split("}")[0].strip().lower()
        for token in ("yes", "no"):
            if token in boxed:
                return token
    lowered = response.lower()
    for token in ("yes", "no"):
        if token in lowered:
            return token
    return None


def _judge_prompt(response: str, category: str) -> str:
    """Build gpt-4o judge prompt for yes/no extraction. category is 'category_1' or 'category_2'."""
    if category == "category_1":
        return (
            "Read the following response and conclude whether the final answer is `yes` or `no` "
            "to the question: Does $\\tilde{f}$ satisfy $\\varepsilon$-differential privacy?\n"
            "Reply with only the conclusion in LaTeX boxed format: \\boxed{yes} or \\boxed{no}.\n\n"
            f"Response:\n{response}"
        )
    elif category == "category_2":
        return (
            "Could you read the following response and conclude if their final answer is `yes` or `no` "
            "for the question. Please make the conclusion `yes` or `no` inside a Latex boxed format "
            "`\\boxed{}`.\n\n"
            f"This is the response:\n{response}"
        )
    else:
        raise NotImplementedError(f"Unsupported category: {category}")



def _judge_prompt_complete_context(response: str, question: str, category: str) -> str:
    """Build gpt-4o judge prompt for yes/no extraction, attached with the question. category is 'category_1' or 'category_2'."""
    
    if category in ["category_1", "category_2"]:
        prompt = (
            "You are a strict grader.\n"
            "Determine whether the response's final answer to the question is YES or NO.\n"
            "If the response is unclear, choose the closest implied conclusion.\n\n"
            f"Question:\n{question}\n\n"
            f"Response:\n{response}\n\n"
            "Strict output rule: output exactly one of the following, and nothing else:\n"
            "\\boxed{yes}\n"
            "\\boxed{no}\n"
        )
        return prompt
    else:
        raise NotImplementedError(f"Unsupported category: {category}")



def _get_inside_box(response: str, depth: int = 1, category: str = "category_1") -> str:
    """Recursively extract a yes/no answer from a model response.

    First attempts direct extraction via ``_try_extract_yes_no``. If that
    fails, re-prompts GPT-4o as a judge (up to 3 levels of recursion).

    Args:
        response: Raw model response text.
        depth: Current recursion depth (1-indexed). Recursion stops at depth 3.
        category: Benchmark category (``"category_1"`` or ``"category_2"``),
            used to select the appropriate judge prompt template.

    Returns:
        ``"yes"``, ``"no"``, or ``""`` if extraction fails at all depths.
    """
    if not response or not response.strip():
        return ""
    answer = _try_extract_yes_no(response)
    if answer is not None:
        return answer
    if depth > 3:
        return ""
    print(f"WARNING: Unable to extract yes/no directly. Switch to gpt-4o Judge, depth {depth}")
    
    prompt = _judge_prompt(response, category)

    judged = get_responses_gpt([prompt], model="gpt-4o")[0]
    return _get_inside_box(judged, depth=depth + 1, category=category)


def parse_yes_no_from_response(response: str, category: str = "category_1") -> int:
    """Parse a yes/no answer from a model response and return it as an integer.

    Args:
        response: Raw model response text.
        category: Benchmark category (``"category_1"`` or ``"category_2"``),
            forwarded to the judge prompt if direct extraction fails.

    Returns:
        ``1`` for yes, ``0`` for no, ``-1`` if the answer cannot be determined.
    """
    answer = _get_inside_box(response, category=category)
    return 1 if answer == "yes" else (0 if answer == "no" else -1)



def get_predicted_labels_category_1(
    responses: List[Dict[str, Any]],
    dataset: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Parse yes/no predictions from raw response records for Category 1.

    Args:
        responses: List of ``{"question_id": ..., "response": "..."}`` dicts
            as written by the evaluation scripts.
        dataset: Full Category 1 dataset (list of question dicts), used to
            look up ground-truth labels and ``function_id`` values.

    Returns:
        List of prediction records, each containing:
        ``question_id``, ``pred`` (1/0/−1), ``label``, and ``function_id``.
        Questions whose ID is absent from ``dataset`` are skipped.
    """
    dataset_by_qid = {int(item["question_id"]): item for item in dataset}
    result: List[Dict[str, Any]] = []
    for record in responses:
        qid = int(record["question_id"])
        if qid not in dataset_by_qid:
            continue
        pred = parse_yes_no_from_response(record.get("response", ""), category="category_1")
        result.append({"question_id": qid, "pred": pred, "label": dataset_by_qid[qid]["label"], "function_id": dataset_by_qid[qid]["function_id"]})
    return result



def _ground_truth_label_to_int(meta_label: Any) -> int:
    """Normalize a Category-2 ground-truth label to ``1`` (yes) or ``0`` (no).

    Local ``meta_info.json`` uses ``\"yes\"`` / ``\"no\"`` strings; the
    Hugging Face release uses integer ``1`` / ``0``.
    """
    if meta_label in (1, "yes", "Yes", True):
        return 1
    if meta_label in (0, "no", "No", False):
        return 0
    raise ValueError(f"Invalid category-2 ground-truth label: {meta_label!r}")


def get_predicted_labels_category_2(
    responses: List[Dict[str, Any]],
    dataset: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Parse yes/no predictions from raw response records for Category 2.

    Converts ground-truth labels to integers (``1``/``0``): local
    ``meta_info.json`` uses ``\"yes\"`` / ``\"no\"`` strings, while the
    Hugging Face release uses integer labels.

    Args:
        responses: List of ``{"question_id": ..., "response": "..."}`` dicts
            as written by the evaluation scripts.
        dataset: Full Category 2 dataset (list of question dicts built from
            ``.tex`` files and ``meta_info.json``), used to look up labels
            and metadata.

    Returns:
        List of prediction records, each containing:
        ``question_id``, ``pred`` (1/0/−1), ``label`` (1/0), ``category``,
        and ``topic``. Questions absent from ``dataset`` are skipped.

    Raises:
        ValueError: If a ground-truth label is not a supported yes/no encoding.
    """
    dataset_by_qid = {int(item["question_id"]): item for item in dataset}
    result: List[Dict[str, Any]] = []
    for record in responses:
        qid = int(record["question_id"])
        if qid not in dataset_by_qid:
            continue
        pred = parse_yes_no_from_response(record.get("response", ""), category="category_2")
        meta_label = dataset_by_qid[qid]["label"]
        gt_label = _ground_truth_label_to_int(meta_label)
        result.append({"question_id": qid, "pred": pred, "label": gt_label, "category": dataset_by_qid[qid]["category"], "topic": dataset_by_qid[qid]["topic"]})
    return result




