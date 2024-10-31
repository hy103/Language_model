import torch
import tiktoken
from GPT_Model import GPTModel

def generate_text_simple(model, idx, 
                         max_new_tokens, context_size):
    for _ in  range(max_new_tokens):
        idx_cond = idx[:, -context_size :]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, : ]
        probas = torch.softmax(logits, dim =-1)
        idx_next = torch.argmax(probas, dim =-1, keepdim = True)
        idx = torch.cat([idx, idx_next], dim =-1)

    return idx

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special = {'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

GPT_CONFIG_124M = {
    "vocab_size" : 50257,
    "context_length" : 256,
    "emb_dim" : 768,
    "n_layers" : 12,
    "n_heads" : 12,
    "drop_rate" : 0.1,
    "qkv_bias" : False
}

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)

start_context = "Every efforts moves you"
tokenizer = tiktoken.get_encoding("gpt2")
token_ids = generate_text_simple(model = model,  
                                 idx = text_to_token_ids(start_context, tokenizer),
                                 max_new_tokens = 10,
                                 context_size= GPT_CONFIG_124M["context_length"])


def main():
    pass   
if __name__ == '__main__':
    main()