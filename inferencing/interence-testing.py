import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt

# Unload models and clean up gpu memory cache
def free_gpu(model):
  if model:
    # Removes the reference to the model's memory,
    # making it eligible for garbage collection.
    del model

  # Release any cached GPU memory that's no longer needed.
  torch.cuda.empty_cache()

  # Trigger garbage collection to ensure memory is fully released.
  gc.collect()


def get_device():
    """
    Get the device to be used for training.
    
    Returns:
        torch.device: The device to be used (CPU or GPU).
    """
    # check for GPU, or mac's accelerator
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device

first_token_generated = False

# (1) Specify the model and load tokenizer and model
model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
device = get_device()
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)

prompt = """The history of human communication is a story of innovation. From ancient cave paintings and spoken language to the invention of writing systems,
 humans have constantly developed new methods to express ideas and share knowledge. 
The printing press revolutionized the spread of information, enabling books to be produced and distributed at an unprecedented scale. Centuries later, the invention of the telegraph, radio, 
and television further transformed how we connect with one another. But perhaps no advancement has reshaped communication more profoundly than the internet.
Today, digital platforms allow billions of people to share messages, media, and experiences in real time. Social media, messaging apps, and video conferencing 
have broken down geographical barriers and created new ways of building communities. At the same time, these technologies raise important questions about privacy, i
nformation overload, and the nature of human interaction.
Looking ahead, emerging technologies such as virtual reality, brain-computer interfaces, and artificial intelligence promise to once again redefine how we communicate. 
As we reflect on this history and anticipate the future, one question arises:
How might the next wave of communication tools shape our relationships, societies, and sense of identity?"""

max_new_tokens =100

#tokenize the input prompt for the first output token
# PS: prompt is the initial input sequence for LLM Generation
idx = tokenizer(prompt , return_tensors="pt").input_ids.to(model.device)
start_time = total_time = time.time()
times =[]

past_key_values =None

# (4) main generation loop - generate tokens one by one
for _ in range(max_new_tokens):
    #print("input_ids size: " + str(idx.size()))
    # current context
    idx_cond = idx
    with torch.no_grad():
        # Generate predictions (token candidates) for next token
        outputs = model(idx_cond)
        #Get the logits (raw prediction scores) for each token precitions
        logits = outputs.logits

    # select next token for the predictions generated in step (B)
    logits = logits[:, -1, :] #Select only the lgoits for the last token
    probas = torch.softmax(logits, dim=-1) # Convert logits to probabilities using softmax

    #sample the next token from the prob distribution 
    idx_next = torch.multinomial(probas, num_samples=1)
    #print("Next Token is:", tokenizer.decode(idx_next[0], skip_special_tokens=True))

    time_cost = time.time() - start_time
    times.append(time_cost)

    #Track the time spent in token generation
    if not first_token_generated:
        #print(f"Time taken for generatign the first token:{time_cost:.4f} seconds ")
        first_token_generated =True
    else:
        pass
        #print(f"Time taken for generating a token: {time_cost:.4f} seconds")
    
    start_time = time.time()

    #Append the new token to the input sequnce
    idx = torch.cat((idx, idx_next), dim=1)

    # if idx_next.item() == tokenizer.eos_token_id:
    #     print("\m Generation compeleted")
    #     break

# Decode the entire generated sequence
generated_text = tokenizer.decode(idx[0], skip_special_tokens=True)
print(f"Total time take to generate : {time.time() - total_time:.4f} seconds")
print(generated_text)

# First chart: First bar red, others blue
plt.figure(figsize=(12, 4))
# plt.subplot(1, 2, 1)
plt.bar(range(len(times)), times, color=['red'] + ['blue'] * (len(times) - 1))
plt.xlabel("Token ID")
plt.ylabel("Time Spent in Token Generation (seconds)")
plt.title("LLM Generation Times for each token (no KV Caching)")
plt.savefig("/Users/raviravindran/Documents/Code/ML/ML/inferencing/latency-nocaching.png", dpi=300, bbox_inches="tight")
plt.show()










