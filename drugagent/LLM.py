""" This file contains the code for calling all LLM APIs. """

import os
from functools import partial
import tiktoken
from .schema import TooLongPromptError, LLMError
import anthropic
from litellm import completion
import time
enc = tiktoken.get_encoding("cl100k_base")

    
def log_to_file(log_file, prompt, completion, model):
    """ Log the prompt and completion to a file."""
    with open(log_file, "a") as f:
        f.write("\n===================prompt=====================\n")
        f.write(f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}")
        num_prompt_tokens = len(enc.encode(f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}"))
        f.write(f"\n==================={model} response =====================\n")
        f.write(completion)
        num_sample_tokens = len(enc.encode(completion))
        f.write("\n===================tokens=====================\n")
        f.write(f"Number of prompt tokens: {num_prompt_tokens}\n")
        f.write(f"Number of sampled tokens: {num_sample_tokens}\n")
        f.write("\n\n")

def complete_text(prompt, log_file, model, **kwargs):
    """Complete text using the specified model with appropriate API, with retries on failure."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = completion(
                model=model,
                messages=[{"content": prompt, "role": "user"}]
            ).choices[0].message.content

            if log_file is not None:
                log_to_file(log_file, prompt, response, model)

            return response  # Return response if successful

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Error occurred: {e}. Retrying in 10 seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(10)
            else:
                print(f"Error occurred: {e}. Max retries reached. Raising exception.")
                raise  # Re-raise the last exception after max retries

FAST_MODEL = "openai/gpt-4o-mini"
def complete_text_fast(prompt, **kwargs):
    """ Complete text using the specified model with appropriate API. """
    return complete_text(prompt = prompt, model = FAST_MODEL, temperature =0.01, **kwargs)






