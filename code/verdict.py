import tiktoken
file_path = "verdict.txt"
with open(file_path, "r", encoding = "utf-8") as f:
    text_data = f.read()



tokenizer = tiktoken.get_encoding("gpt2")
total_tokens = len(tokenizer.encode(text_data))
print(f"Total tokens {total_tokens}")
print(f"Total Characters {len(text_data)}")
