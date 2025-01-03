import torch
import tiktoken
from GPT_Model import GPTModel
from create_dataloader import create_dataloader_v1
from text_token_text import text_to_token_ids, token_ids_to_text, generate_text_simple



def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0,1), target_batch.flatten()
    )

    return loss


def calc_loss_loader(data_loader, model, device, num_batches = None):
    total_loss = 0
    if len(data_loader)==0:
        return float("nan")
    
    elif num_batches is None:
        num_batches = len(data_loader)

    else: 
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i< num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)

            total_loss+= loss.item()
        else:
            break
    return total_loss/num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches= eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches= eval_iter)

        model.train()
        return train_loss , val_loss
    
def generate_and_print(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids =generate_text_simple(model = model, idx = encoded,
                                        max_new_tokens = 50, context_size = context_size)
        
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()

def train_model_simple(model, train_loader, val_loader, 
                       optimizer, device, num_epochs, 
                       eval_freq, eval_iter, start_context, tokenizer):
    
    train_losses , val_losses ,track_tokens_seen = [], [], []

    tokens_seen , global_step = 0, -1
    print(f"Number of epochs {num_epochs}")
    for epoch in range(num_epochs):
        print(f"Epoch : {epoch}")
        model.train()
        for input_batch, target_batch in (train_loader):
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)

            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step+=1

            if global_step % eval_freq ==0:
                train_loss , val_loss  = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                print(f"Epoch {epoch+1} (Step {global_step:06d}):"
                      f"Train loss {train_loss : 3f}"
                      f"Val loss {val_loss : 3f}")
                

        generate_and_print(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen







GPT_CONFIG_124M = {
    "vocab_size" : 50257,
    "context_length" : 256,
    "emb_dim" : 768,
    "n_layers" : 12,
    "n_heads" : 12,
    "drop_rate" : 0.1,
    "qkv_bias" : False
}

file_path = "verdict.txt"

with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()
train_ratio = 0.9
split_idx = int(train_ratio*len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

train_loader = create_dataloader_v1(train_data,
                                    batch_size =2, 
                                     max_length =  GPT_CONFIG_124M["context_length"],
                                     stride = GPT_CONFIG_124M["context_length"],
                                     drop_last = True,
                                     shuffle = True,
                                     num_workers =0)


val_loader = create_dataloader_v1(val_data,
                                    batch_size =2, 
                                     max_length =  GPT_CONFIG_124M["context_length"],
                                     stride = GPT_CONFIG_124M["context_length"],
                                     drop_last = True,
                                     shuffle = True,
                                     num_workers =0)



torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr = 0.0004, weight_decay = 0.1)
tokenizer = tiktoken.get_encoding("gpt2")

num_epochs = 25
train_losses , val_losses, tokens_seen = train_model_simple(model,
                                                            train_loader, val_loader, optimizer,
                                                            device, num_epochs=num_epochs, eval_freq =5, eval_iter = 5,
                                                            start_context= "Every effort moves you", tokenizer=tokenizer)



torch.save(model.state_dict(), "model.pth")
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
},
"model_and_optimizer.pth")


def main():
    pass   
if __name__ == '__main__':
    main()