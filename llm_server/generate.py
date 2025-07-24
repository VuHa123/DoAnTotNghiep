# generate.py
import torch

def generate_stream(model, tokenizer, device, prompt: str, max_new_tokens: int = 100):
    input_ids = tokenizer(prompt,truncation=True, max_length=1024, return_tensors="pt").input_ids.to(device)
    model.eval()

    output_ids = input_ids.clone()

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids=output_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            output_ids = torch.cat([output_ids, next_token_id], dim=-1)

        next_token = tokenizer.decode(next_token_id.squeeze(), skip_special_tokens=True)
        yield next_token

        if next_token.strip() in tokenizer.eos_token or next_token.strip().endswith("<|end|>"):
            break
