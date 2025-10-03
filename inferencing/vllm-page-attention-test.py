import torch
import time
import matplotlib.pyplot as plt
from vllm import LLM, SamplingParams
import gc
import numpy as np

def get_device_string():
    """
    Get the device to be used for training as a string.
    
    Returns:
        str: The device to be used ('cuda', 'mps', or 'cpu').
    """
    if torch.cuda.is_available():
        return "cuda"
    # vLLM does not support MPS, so fallback to CPU
    elif torch.backends.mps.is_available():
        return "cpu" 
    else:
        return "cpu"

def run_benchmark(prompt_workloads, all_prompts, device):
    """
    Runs the vLLM benchmark with different prompt workloads.
    """
    execution_times = []
    
    # Use the default block size that works on CPU
    block_size = 16
    print(f"Using fixed block_size: {block_size} (default for CPU execution)")

    for num_prompts in prompt_workloads:
        print(f"Testing with {num_prompts} prompts...")
        
        # Unload model and clear memory before the next run
        llm = None
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()

        try:
            # Load model with vLLM
            llm = LLM(
                model="Qwen/Qwen2.5-0.5B",
                dtype="float16" if device == 'cuda' else "auto",
                block_size=block_size,
                enforce_eager=True if device == 'cpu' else False,
                max_model_len=4096
            )

            # Create inference parameters.
            sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)
            
            current_prompts = all_prompts[:num_prompts]

            start_time = time.time()
            # Run token (text) generation with prompts.
            llm.generate(current_prompts, sampling_params)
            end_time = time.time()
            
            execution_time = end_time - start_time
            execution_times.append(execution_time)
            print(f"  Execution time: {execution_time:.4f} seconds")

        except Exception as e:
            print(f"  Error with {num_prompts} prompts: {e}")
            execution_times.append(float('inf')) # Indicate failure
        finally:
            # Clean up to be safe
            del llm
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()

    return execution_times

def plot_performance(prompt_workloads, execution_times, device):
    """
    Plots the performance for different prompt workloads.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(prompt_workloads, execution_times, marker='o')
    plt.title(f'vLLM Performance vs. Workload Size ({device.upper()})')
    plt.xlabel('Number of Prompts')
    plt.ylabel('Execution Time (seconds)')
    plt.xticks(prompt_workloads)
    plt.grid(True)
    
    # Save the plot
    output_path = "/Users/raviravindran/Documents/Code/ML/ML/inferencing/paged_attention_performance.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to {output_path}")
    plt.show()

def main():
    """
    Main function to run the benchmark and plot results.
    """
    device = get_device_string()
    print(f"Running on device: {device}")

    # A workload of prompts
    all_prompts = [
        "The history of human communication is a story of innovation.",
        "What is the meaning of life?",
        "Write a short story about a robot learning to love.",
        "Explain quantum physics in simple terms.",
        "Translate 'Hello, world!' into Spanish.",
        "What are the main challenges in developing AI?",
        "Describe the process of photosynthesis.",
        "Who was the first person to walk on the moon?",
        "The future of renewable energy is...",
        "How does a blockchain work?",
        "What is the capital of France?",
        "Tell me a joke.",
        "Summarize the plot of 'Hamlet'.",
        "What is the difference between a virus and a bacteria?",
        "Explain the theory of relativity.",
        "Write a poem about the sea."
    ]

    # Prompt workloads to test
    prompt_workloads = [4, 8, 12, 16]

    print(f"Starting vLLM performance benchmark.")
    
    execution_times = run_benchmark(prompt_workloads, all_prompts, device)
    
    # Filter out failed runs for plotting
    valid_results = [(wl, et) for wl, et in zip(prompt_workloads, execution_times) if et != float('inf')]
    if not valid_results:
        print("All benchmark runs failed. No plot will be generated.")
        return
        
    plot_workloads, plot_execution_times = zip(*valid_results)

    plot_performance(plot_workloads, plot_execution_times, device)

if __name__ == "__main__":
    main()