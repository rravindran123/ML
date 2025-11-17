def check_gpu_availability():
    """
    Checks for CUDA GPU availability and exits if not found.
    Returns the device string if available.
    """
    if not torch.cuda.is_available():
        print("Error: No CUDA-enabled GPU found. This script requires a GPU to run.")
        sys.exit(1)

    device = "cuda"
    print(f"Found CUDA-enabled GPU. Running on device: {device}")
    return device

def run_gpu_benchmark(block_sizes, prompts, device):
    """
    Runs the vLLM benchmark on a GPU with different block sizes.
    """
    execution_times = []

    for block_size in block_sizes:
        print(f"Testing with block_size: {block_size}")

        # Unload model and clear memory before the next run
        llm = None
        gc.collect()
        torch.cuda.empty_cache()

        try:
            # Load model with vLLM with the specified block size for GPU
            # Use default block_size if input block_size is None
            llm = LLM(
                model="Qwen/Qwen2.5-0.5B",
                dtype="float16", # Best for modern GPUs
                block_size=block_size if block_size is not None else None,
                max_model_len=4096
            )

            # Create inference parameters.
            sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)

            start_time = time.time()
            # Run token (text) generation with prompts.
            llm.generate(prompts, sampling_params)
            end_time = time.time()

            execution_time = end_time - start_time
            execution_times.append(execution_time)
            print(f"  Execution time: {execution_time:.4f} seconds")

        except Exception as e:
            print(f"  Error with block_size {block_size}: {e}")
            execution_times.append(float('inf')) # Indicate failure
        finally:
            # Clean up to be safe
            del llm
            gc.collect()
            torch.cuda.empty_cache()

    return execution_times

def plot_gpu_performance(block_sizes, execution_times_list, device):
    """
    Plots the performance of different block sizes on GPU.
    """
    plt.figure(figsize=(10, 6))
    markers = ['o', 'x']
    colors = ['green', 'red']
    labels = ['Large Prompt', 'Small Prompt']

    for list_idx, execution_times in enumerate(execution_times_list):
        plt.plot(block_sizes, execution_times, marker=markers[list_idx], color=colors[list_idx], label=labels[list_idx])

    plt.title(f'vLLM Paged Attention Performance on GPU ({device.upper()})')
    plt.xlabel('Block Size')
    plt.ylabel('Execution Time (seconds)')
    plt.xticks(block_sizes)
    plt.grid(True)
    plt.legend()

    # Save the plot
    output_path = "gpu_paged_attention_performance.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to {output_path}")
    plt.show()

def main():
    """
    Main function to run the GPU benchmark and plot results.
    """
    device = check_gpu_availability()

    # A constant workload of 16 prompts
    prompts = [
        """The history of human communication is a story of innovation. From ancient cave paintings and spoken language to the invention of writing systems,humans have
        constantly developed new methods to express ideas and share knowledge  The printing press revolutionized the spread of information, enabling books to be produced and distributed
        at an unprecedented scale. Centuries later, the invention of the telegraph, radio,
        and television further transformed how we connect with one another. But perhaps no advancement has reshaped communication more profoundly than the internet.
        Today, digital platforms allow billions of people to share messages, media, and experiences in real time. Social media, messaging apps, and video conferencing
        have broken down geographical barriers and created new ways of building communities. At the same time, these technologies raise important questions about privacy, i
        nformation overload, and the nature of human interaction.Looking ahead, emerging technologies such as virtual reality, brain-computer interfaces, and artificial intelligence promise
        to once again redefine how we communicate.
        As we reflect on this history and anticipate the future, one question arises: How might the next wave of communication tools shape our relationships, societies, and sense of identity?"""
      ]
    prompts2 = [
        "The history of human communication is a story of innovation.",
        # "What is the meaning of life?",
        #  "Write a short story about a robot learning to love.",
        #  "Explain quantum physics in simple terms.",
        # "Translate 'Hello, world!' into Spanish.",
        # "What are the main challenges in developing AI?",
        # "Describe the process of photosynthesis.",
        # "Who was the first person to walk on the moon?",
        # "The future of renewable energy is...",
        # "How does a blockchain work?",
        # "What is the capital of France?",
        # "Tell me a joke.",
        # "Summarize the plot of 'Hamlet'.",
        # "What is the difference between a virus and a bacteria?",
        # "Explain the theory of relativity.",
        # "Write a poem about the sea."
    ]

    # Block sizes to test, start with None for default block size
    block_sizes = [1,4,8, 16, 32]

    print(f"Starting vLLM GPU benchmark with a constant batch of {len(prompts)} prompts.")

    execution_times = run_gpu_benchmark(block_sizes, prompts, device)
    execution_times2 = run_gpu_benchmark(block_sizes, prompts2, device)

    # Filter out failed runs for plotting
    valid_results = [(bs, et) for bs, et in zip(block_sizes, execution_times) if et != float('inf')]
    if not valid_results:
        print("All benchmark runs failed. No plot will be generated.")
        return

    # Replace None with a label for plotting if needed
    plot_block_sizes = [bs if bs is not None else 'Default' for bs in block_sizes]
    plot_execution_times = [et for bs, et in zip(block_sizes, execution_times) if et != float('inf')]
    plot_execution_times_2 = [et for bs, et in zip(block_sizes, execution_times2) if et != float('inf')]

    plot_execution_times_list = [plot_execution_times, plot_execution_times_2]

    plot_gpu_performance(plot_block_sizes, plot_execution_times_list, device)

if __name__ == "__main__":
    main()