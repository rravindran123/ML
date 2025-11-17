import time
import random
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

# ----------------------------
# Configuration
# ----------------------------
LARGE_MODEL = "EleutherAI/pythia-1.4b-deduped"
SMALL_MODEL = "EleutherAI/pythia-160m-deduped"
#DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DEVICE="mps"

# Set random seed for reproducibility
random.seed(42)
torch.manual_seed(42)

# Number of test prompts
NUM_PROMPTS = 10
MAX_NEW_TOKENS = 50

# ----------------------------
# Load models and tokenizer
# ----------------------------
print("Loading models...")

tokenizer = AutoTokenizer.from_pretrained(LARGE_MODEL)
large_model = AutoModelForCausalLM.from_pretrained(LARGE_MODEL).to(DEVICE)
small_model = AutoModelForCausalLM.from_pretrained(SMALL_MODEL).to(DEVICE)

# ----------------------------
# Generate random prompts
# ----------------------------
base_prompts = [
    "Once upon a time in a distant galaxy,",
    "In a world where AI writes stories,",
    "The quick brown fox jumps over",
    "Quantum computing will revolutionize",
    "Alice and Bob went to the market to buy",
    "The sun was setting over the mountains, when",
    "In the future, robots will be able to",
    "The meaning of life is often debated by",
    "Under the deep blue ocean, a mysterious",
    "If humans could travel faster than light,"
]
prompts = random.sample(base_prompts, NUM_PROMPTS)

# ----------------------------
# Evaluation helper
# ----------------------------
def generate_with_timing(model, inputs, **gen_kwargs):
    start = time.time()
    _ = model.generate(**inputs, **gen_kwargs)
    end = time.time()
    return end - start

# ----------------------------
# Run benchmark
# ----------------------------
no_spec_times = []
spec_times = []

print(f"\nEvaluating {NUM_PROMPTS} prompts...\n")

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    # Normal generation (no speculative decoding)
    t_normal = generate_with_timing(
        large_model, inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False
    )

    # Speculative decoding with assistant model
    t_spec = generate_with_timing(
        large_model, inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        assistant_model=small_model
    )

    no_spec_times.append(t_normal)
    spec_times.append(t_spec)
    print(f"Prompt: {prompt[:35]:35s} | Normal: {t_normal:.2f}s | Speculative: {t_spec:.2f}s")

# ----------------------------
# Plot results
# ----------------------------
speedups = [no_spec_times[i] / spec_times[i] for i in range(NUM_PROMPTS)]
avg_speedup = sum(speedups) / len(speedups)

plt.figure(figsize=(10, 5))
x = range(NUM_PROMPTS)
plt.bar(x, speedups, color="skyblue")
plt.axhline(y=avg_speedup, color='r', linestyle='--', label=f"Avg speedup: {avg_speedup:.2f}×")
plt.title("Speculative Decoding Speedup per Prompt")
plt.xlabel("Prompt Index")
plt.ylabel("Speedup (Normal / Speculative)")
plt.legend()
plt.tight_layout()
plt.show()

print(f"\nAverage speculative decoding speedup: {avg_speedup:.2f}×")
