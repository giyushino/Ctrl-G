#conda_env: modelmerge
import argparse
import requests
import torch
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Getting samples from base model")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--num_samples", type=int, default=10_000)

    args = parser.parse_args()
    return args

def pad_to_len(x, d, pad_value):
    # If 1D, make it 2D temporarily
    if x.ndim == 1:
        x = x.unsqueeze(0)  # [1, seq_len]
        squeeze_back = True
    else:
        squeeze_back = False

    if x.shape[1] < d:
        new_shape = x.shape[:1] + (d - x.shape[1],) + x.shape[2:]
        x = torch.cat((x, torch.full(new_shape, pad_value, dtype=x.dtype)), dim=1)

    if squeeze_back:
        x = x.squeeze(0)

    return x


def tensorize_outputs(outputs, prompt_length, max_new_tokens, batch_size):
    hidden_states = []
    sequences = []
    expected_length = prompt_length + max_new_tokens
    
    count = 0
    for output in outputs:
        count += 1
        if len(output["output_ids"]) < expected_length:
            short_sequence = torch.tensor(output["output_ids"][prompt_length:])
            short_sequence = pad_to_len(short_sequence, max_new_tokens, 151645)
            sequences.append(short_sequence) 
            per_step = [torch.tensor(step, dtype=torch.float32).squeeze(0) for step in output["meta_info"]["hidden_states"]]        
            per_step_stacked = torch.stack(per_step, dim=0)
            per_step_stacked = pad_to_len(per_step_stacked, max_new_tokens, 151645) # not sure why the original repo used the EOS token
            hidden_states.append(per_step_stacked)

        else:
            sequences.append(torch.tensor(output["output_ids"][prompt_length:]))
            per_step = [torch.tensor(step, dtype=torch.float32).squeeze(0) for step in output["meta_info"]["hidden_states"]]        
            per_step_stacked = torch.stack(per_step, dim=0)
            hidden_states.append(per_step_stacked)

    hidden_states = torch.stack(hidden_states, dim=0)
    sequences = torch.stack(sequences, dim=0)
    
    
    return hidden_states, sequences

def main(batch_size, temperature, max_new_tokens, port, num_samples, save_path):
    # use the EOS token for free generation
    prompts = ["<|endoftext|>" for _ in range(batch_size)]
    prompt_length = 1 # should be dynamic but since we're only using the EOS, we know this is 1
      
    sampling_params = {
        "temperature": temperature,
        "top_p": 0.95,
        "max_new_tokens": max_new_tokens,
    }
    json_data = {
        "text": prompts,
        "sampling_params": sampling_params,
        "return_hidden_states": True,
    }
    response = requests.post(
        f"http://localhost:{port}/generate",
        json=json_data,
    )
    
    t0 = time.time()
    hidden_states = []
    sequences = []

    for i in range(num_samples // batch_size):
        outputs = response.json()
        temp_hidden_states, temp_sequences = tensorize_outputs(outputs, prompt_length, max_new_tokens, batch_size)
        
        # super hacky
        if temp_hidden_states.size()[1] != max_new_tokens: 
            pass
        else:
            hidden_states.append(temp_hidden_states)
            sequences.append(temp_sequences)
        print(f"on {i * batch_size} number of samples!")
        if (i * batch_size) % 500 == 0 and i != 0:
            t1 = time.time()
            print(f"time elapsed: {(t1 - t0)/3600:.4f} Hours")
            print(f"estimated time: {(num_samples / (i * batch_size) * (t1 - t0) / 3600):.4f} Hours")

    sequences = torch.cat(sequences, dim=0)
    hidden_states = torch.cat(hidden_states, dim=0)
    torch.save(hidden_states, f"{save_path}hidden_states.pt")
    torch.save(sequences, f"{save_path}sequences.pt")

    #print(t1 - t0) 


if __name__ == "__main__":
    args = parse_args()
    sample_args = {
        "batch_size": args.batch_size,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "port": args.port,
        "num_samples": args.num_samples,
        "save_path": "/home/allanz/Ctrl-G/data/dev/" #args.save_path
    }
    
    main(**sample_args)
    #tensor = torch.load("/home/allanz/Ctrl-G/sequences.pt")
    #print(tensor.shape)

