import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
from transformers import pipeline

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

# Run model inference with Hugging Face library
# Load model and tokenizer in Hugging Face library   
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", device_map="auto", trust_remote_code=True)
 
start_time_basic = time.time()
# Create the model prediction pipeline.
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
# Generate prediction with prompt
outputs_basic = generator(prompt, max_length=100, temperature=0.8, top_p=0.95)
end_time_basic = time.time()
print(f"Time taken : {end_time_basic-start_time_basic}")

for output in outputs_basic:
    print(f"size of text {len(output['generated_text'])}, text: {output['generated_text']}")












