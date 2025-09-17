#conda_env: ctrlg
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
    parser.add_argument("--chunk_size", type=int, default=100_000)
    parser.add_argument("--total_chunks", type=int, default=1)
    parser.add_argument('--save_embeddings', type=int, default=0)

    args = parser.parse_args()
    if args.save_embeddings == 1:
        args.save_embeddings = True
    else:
        args.save_embeddings = False
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

def get_prompt_hidden_states(outputs, prompt_length):
    for output in outputs:
        if len(output["meta_info"]["hidden_states"][0]) == prompt_length:
            return torch.tensor(output["meta_info"]["hidden_states"][0][-1], dtype=torch.float16)

    return None

def tensorize_outputs(outputs, prompt_length, max_new_tokens, pad_value, prompt_hidden_state, use_hidden_states):
    hidden_states = []
    sequences = []
    expected_length = prompt_length + max_new_tokens
    
    if use_hidden_states:
        for output in outputs:
            if len(output["output_ids"]) < expected_length:
                short_sequence = torch.tensor(output["output_ids"][prompt_length:])
                short_sequence = pad_to_len(short_sequence, max_new_tokens, pad_value)
                sequences.append(short_sequence) 
                per_step = [torch.tensor(step, dtype=torch.float32).squeeze(0) for step in output["meta_info"]["hidden_states"]]        
                per_step[0] = prompt_hidden_state

                if len(per_step) < max_new_tokens:
                    for _ in range(max_new_tokens - len(per_step)):
                        per_step.append(torch.full((1536,), pad_value))

                per_step_stacked = torch.stack(per_step, dim=0)
                hidden_states.append(per_step_stacked)

            else:
                sequences.append(torch.tensor(output["output_ids"][prompt_length:]))
                per_step = [torch.tensor(step, dtype=torch.float32).squeeze(0) for step in output["meta_info"]["hidden_states"]]        
                per_step[0] = prompt_hidden_state
                
                if len(per_step) < max_new_tokens:
                    for _ in range(max_new_tokens - len(per_step)):
                        per_step.append(torch.full((1536,), pad_value))

                per_step_stacked = torch.stack(per_step, dim=0)
                hidden_states.append(per_step_stacked)

        hidden_states = torch.stack(hidden_states, dim=0)
        sequences = torch.stack(sequences, dim=0)
        
        return hidden_states, sequences

    else:
        for output in outputs:
            if len(output["output_ids"]) < expected_length:
                short_sequence = torch.tensor(output["output_ids"][prompt_length:])
                short_sequence = pad_to_len(short_sequence, max_new_tokens, pad_value)
                sequences.append(short_sequence) 
            else:
                sequences.append(torch.tensor(output["output_ids"][prompt_length:]))

        sequences = torch.stack(sequences, dim=0)
        return None, sequences


def generate_output(temperature, max_new_tokens, save_embeddings, port, prompts):
    sampling_params = {
        "temperature": temperature,
        "top_p": 1,
        "max_new_tokens": max_new_tokens,
    }
    json_data = {
        "text": prompts,
        "sampling_params": sampling_params,
        "return_hidden_states": save_embeddings, 
    }
    response = requests.post(
        f"http://localhost:{port}/generate",
        json=json_data,
    )
    return response.json()

def main(batch_size, temperature, max_new_tokens, port, save_path, chunk_size, total_chunks, save_embeddings=False):
    # use the EOS token for free generation
    #prompts = ["<|endoftext|>" for _ in range(batch_size)]
    #prompt_length = 1 # should be dynamic but since we're only using the EOS, we know this is 1

    prompts = ["Respond ONLY in ENGLISH: <|endoftext|>" for _ in range(batch_size)]
    prompt_length = 7 # should be dynamic but since we're only using this prompt, we know it is 7

    prompt_hidden_state = None
    if save_embeddings:
        while prompt_hidden_state is None:
            outputs = generate_output(temperature, max_new_tokens, save_embeddings, port, prompts)
            # we can also get prompt length from output[0]["meta_info"]["prompt_tokens"]
            prompt_hidden_state = get_prompt_hidden_states(outputs, prompt_length)
    
    print(f"sampling {total_chunks * chunk_size} data points") 
    for chunk in range(total_chunks):
        print(f"on chunk {chunk}")
        t0 = time.time()
        hidden_states = []
        sequences = []

        for i in range(chunk_size // batch_size):
            outputs = generate_output(temperature, max_new_tokens, save_embeddings, port, prompts)
            temp_hidden_states, temp_sequences = tensorize_outputs(outputs, prompt_length, max_new_tokens, 151645, prompt_hidden_state, save_embeddings)
            if temp_hidden_states is not None:
                hidden_states.append(temp_hidden_states)
            sequences.append(temp_sequences)

            print(f"on {i * batch_size} number of samples!")
            if (i * batch_size) % 500 == 0 and i != 0:
                t1 = time.time()
                print(f"time elapsed: {(t1 - t0)/3600:.4f} Hours")
                print(f"estimated time: {(chunk_size / (i * batch_size) * (t1 - t0) / 3600):.4f} Hours")

        sequences = torch.cat(sequences, dim=0)
        if save_embeddings:
            hidden_states = torch.cat(hidden_states, dim=0)
        
        print(f"saving data to {save_path}")
        if total_chunks == 0:
            torch.save(hidden_states, f"{save_path}.embeddings")
            torch.save(sequences, f"{save_path}")
        else:
            #torch.save(hidden_states, f"{save_path}.embeddings.{chunk}")
            torch.save(sequences, f"{save_path}.{chunk}")

    #print(t1 - t0) 


if __name__ == "__main__":
    args = parse_args()
    sample_args = {
        "batch_size": args.batch_size,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "port": args.port,
        "chunk_size": args.chunk_size,
        "total_chunks": args.total_chunks,
        "save_path": args.save_path,
        "save_embeddings": args.save_embeddings
    }
    print(sample_args)
    main(**sample_args)
