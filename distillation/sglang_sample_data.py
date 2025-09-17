import os
import json
import argparse
import requests
import torch
from tqdm import tqdm

def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--model_name_or_path', default='', type=str, help='Not used with SGLang, kept for compatibility.')
    arg_parser.add_argument('--tokenizer_name_or_path', default='', type=str, help='Not used with SGLang, kept for compatibility.')
    arg_parser.add_argument('--cache_prompt_past_key_values', action='store_true', help='Not used with SGLang, kept for compatibility.')
    arg_parser.add_argument('--float16', action='store_true', help='Not used with SGLang, kept for compatibility.')
    arg_parser.add_argument('--bfloat16', action='store_true', help='Not used with SGLang, kept for compatibility.')

    arg_parser.add_argument('--input_file',  default='', type=str, help='A JSON file with a list of prompts.')
    arg_parser.add_argument('--chunk_size', default=32, type=int, help='Total number of samples to generate per chunk.')
    arg_parser.add_argument('--total_chunks',  default=1, type=int, help='Total number of chunks to generate.')
    arg_parser.add_argument('--batch_size', default=32, type=int, help='Batch size for requests to the SGLang server.')
    arg_parser.add_argument('--max_new_tokens', type=int, default=128, help='Maximum number of new tokens to generate.')
    arg_parser.add_argument('--save_embeddings', action='store_true', help='Whether to save hidden states (embeddings).')

    arg_parser.add_argument('--top_k', type=int, default=0, help='Top-k sampling parameter.')
    arg_parser.add_argument('--top_p', type=float, default=1.0, help='Top-p sampling parameter.')
    arg_parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature.')
    
    arg_parser.add_argument('--port', type=int, default=3000, help='Port of the SGLang server.')
    arg_parser.add_argument('--output_file',  default='', type=str, help='Base name for the output file.')

    args = arg_parser.parse_args()
    return args


def pad_to_len(x, d, pad_value):
    if x.ndim == 1:
        x = x.unsqueeze(0)
        squeeze_back = True
    else:
        squeeze_back = False

    if x.shape[1] < d:
        new_shape = x.shape[:1] + (d - x.shape[1],) + x.shape[2:]
        x = torch.cat((x, torch.full(new_shape, pad_value, dtype=x.dtype)), dim=1)

    if squeeze_back:
        x = x.squeeze(0)

    return x

def tensorize_outputs(outputs, prompt_length, max_new_tokens, pad_value, use_hidden_states):
    hidden_states = []
    sequences = []

    if use_hidden_states is True:
        for output in outputs:
            # so we need to do .squeeze() since the first step is a nested list, unlike the other steps
            per_step = [torch.tensor(step, dtype=torch.float32).squeeze() for step in output["meta_info"]["hidden_states"]]
            # padding so all sequences are the same length
            if len(per_step) < max_new_tokens:
                for _ in range(max_new_tokens - len(per_step)):
                    per_step.append(torch.full((1536,), pad_value))
             
            per_step_stacked = torch.stack(per_step, dim=0)
            hidden_states.append(per_step_stacked)

            full_sequence = torch.tensor(output["output_ids"][prompt_length:])
            padded_sequence = pad_to_len(full_sequence, max_new_tokens, pad_value)
            sequences.append(padded_sequence)

        hidden_states = torch.stack(hidden_states, dim=0)
        sequences = torch.stack(sequences, dim=0)
        return hidden_states, sequences

    else:
        for output in outputs:
            full_sequence = torch.tensor(output["output_ids"][prompt_length:])
            padded_sequence = pad_to_len(full_sequence, max_new_tokens, pad_value)
            sequences.append(padded_sequence)

        sequences = torch.stack(sequences, dim=0)
        return None, sequences

if __name__ == '__main__':
    args = parse_args()
    print(f"port {args.port}")
    # SGLang client setup
    base_url = f"http://localhost:{args.port}/generate"

    # Use the EOS token as the default prompt for "free generation" as per the user's SGLang code
    # NOTE: The original code loaded prompts from `input_file`. This SGLang version
    # simplifies to using a single EOS token prompt, as per the user's provided SGLang code.
    # To be more like the original, you'd need to read `input_data` and loop through each prompt.
    prompts = ["<|endoftext|>"] * args.batch_size
    prompt_ids = [151645] # Hardcoded EOS token ID for now. In a real scenario, you'd get this from the tokenizer.
    prompt_length = len(prompt_ids)
    pad_value = 151645 # EOS token id for padding

    # The original code's distributed logic is simulated here.
    # Since we're not using torch.distributed, we just set rank=0 and world_size=1
    # and process the full chunk_size.
    rank = 0
    world_size = 1

    # Load input data to know the number of prompts to process.
    # The original code uses a loop for this, we'll replicate that.
    """
    if args.input_file:
        with open(args.input_file, 'r') as fin:
            input_data = json.load(fin)
    else:
        # Fallback for "free generation" mode
    """ 
    input_data = ["<|endoftext|>"]
    for chunk_idx in range(args.total_chunks):
        if rank == 0:
            print(f'generating samples for chunk {chunk_idx} ...')

        sequences = []
        embeddings = []
        
        # The original code loops through prompts and then batches.
        # We'll follow that structure.
        for prompt_idx, prompt in enumerate(input_data):
            if rank == 0:
                print(f'generating samples for prompt {prompt_idx} ...')

            prompt_ids = [151645] 
            prompt_length = len(prompt_ids)
            
            # The original code's distributed chunking logic:
            chunk_size_per_process = args.chunk_size // world_size
            chunk_size_per_process_prompt = chunk_size_per_process // len(input_data) + 1
            chunk_size_per_process_prompt = min(chunk_size_per_process_prompt, chunk_size_per_process - prompt_idx * (chunk_size_per_process // len(input_data) + 1))
            
            for batch_idx in tqdm(range(0, chunk_size_per_process_prompt, args.batch_size)):
                batch_size_ = min(args.batch_size, chunk_size_per_process_prompt - batch_idx)
                
                # SGLang request payload
                sampling_params = {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    #"top_k": args.top_k,
                    "max_new_tokens": args.max_new_tokens,
                }
                
                json_data = {
                    "text": [prompt] * batch_size_,
                    "sampling_params": sampling_params,
                    "return_hidden_states": args.save_embeddings,
                }
                
                response = requests.post(base_url, json=json_data)
                response.raise_for_status()
                outputs = response.json()
                temp_hidden_states, temp_sequences = tensorize_outputs(outputs, prompt_length, args.max_new_tokens, pad_value, args.save_embeddings)

                if args.save_embeddings:
                    embeddings.append(temp_hidden_states)
                sequences.append(temp_sequences)
        
        # Concatenate all collected sequences and embeddings
        sequences_tensor = torch.cat(sequences, dim=0)

        # Replicate the original code's saving logic (shuffling and saving)
        if rank == 0:
            perm = torch.randperm(sequences_tensor.shape[0])
            sequences_tensor = sequences_tensor[perm, :]
            output_file = f'{args.output_file}.{chunk_idx}' if args.total_chunks > 1 else f'{args.output_file}'
            print(f"saving to {output_file}")
            torch.save(sequences_tensor, output_file)


            if args.save_embeddings:
                embeddings_tensor = torch.cat(embeddings, dim=0)
                embeddings_tensor = embeddings_tensor[perm, :]
                torch.save(embeddings_tensor, output_file + '.embeddings')
