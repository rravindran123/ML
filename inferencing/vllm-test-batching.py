import time
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_name = "Qwen/Qwen2.5-0.5B"
    #tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load model with vLLM
    llm = LLM(model=model_name, dtype="float16",  max_model_len=4096, max_num_batched_tokens=32768)

    # Define the prompt.
    # Prompts for batch generation, 4 input sequences
    prompts = [
        "What is the meaning of life?",
        "Write a short story about a robot learning to love.",
        "Explain quantum physics in simple terms.",
        "Translate 'Hello, world!' into Spanish."
    ]   

    # Create inference parameters.
    inference_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100, stop=None, ignore_eos=True)

    start_time = time.time()
    # process four input sequences together in one batch
    vllm_outputs = llm.generate(prompts, inference_params)
    end_time = time.time()
    vllm_time = end_time - start_time

    print(f"\nvLLM generation time for 4 prompts in a batch: {vllm_time:.4f} seconds")
    
    # process prompt one by one
    start_time = time.time()
    for prompt in prompts:
        vllm_outputs = llm.generate([prompt], inference_params)
    end_time = time.time()
    vllm_time = end_time - start_time
    print(f"\nvLLM generation time for 4 prompts one by one: {vllm_time:.4f} seconds")

if __name__== "__main__":
    main()