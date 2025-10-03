import time
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_name = "Qwen/Qwen2.5-0.5B"
    #tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load model with vLLM
    llm = LLM(model=model_name, dtype="float16",  max_model_len=4096, max_num_batched_tokens=32768)

    # Define the prompt.
    prompt = """The history of human communication is a story of innovation. From ancient cave paintings and spoken language to the invention of writing systems,
    humans have constantly developed new methods to express ideas and share knowledge  The printing press revolutionized the spread of information, enabling books to be produced and distributed at an unprecedented scale. Centuries later, the invention of the telegraph, radio, 
    and television further transformed how we connect with one another. But perhaps no advancement has reshaped communication more profoundly than the internet.
    Today, digital platforms allow billions of people to share messages, media, and experiences in real time. Social media, messaging apps, and video conferencing 
    have broken down geographical barriers and created new ways of building communities. At the same time, these technologies raise important questions about privacy, i
    nformation overload, and the nature of human interaction.Looking ahead, emerging technologies such as virtual reality, brain-computer interfaces, and artificial intelligence promise to once again redefine how we communicate. 
    As we reflect on this history and anticipate the future, one question arises: How might the next wave of communication tools shape our relationships, societies, and sense of identity?""" 


    prompt1 = """what is the capitol of USA ?"""

    # Create inference parameters.
    inference_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100, stop=None, ignore_eos=True)

    start_time = time.time()
    # Run token (text) generation with prompt and inference parameters.
    outputs = llm.generate([prompt], inference_params)

    print(f"Time taken {time.time()- start_time}")

    # Print the results.
    print(len(outputs))
    for output in outputs:
        for index , candidate in enumerate(output.outputs):
            num_tokens = len(candidate.token_ids)
            print(f"Generated text: {index}, Token length: {num_tokens} txt: {candidate.text}")
            print("Last token ID:", candidate.token_ids[-1])

if __name__== "__main__":
    main()