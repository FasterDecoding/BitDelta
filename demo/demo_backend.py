from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import torch.nn as nn
from bitdelta.binary_gemm_kernel import binary_bmm
from fastchat.model.model_adapter import get_conversation_template
import json
import os
import fastapi
from fastapi.responses import StreamingResponse

from typing import Dict, List, Literal, Optional, Union
import gc
import logging

# log to file
logging.basicConfig(filename="demo_backend.log", level=logging.INFO)

app = fastapi.FastAPI()

base_model = "mistralai/Mistral-7B-v0.1"

model = AutoModelForCausalLM.from_pretrained(
    base_model, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto"
)

model_dict = json.load(open("supported_models.json", "r"))
for model_name in model_dict:
    tokenizer = model_dict[model_name]["tokenizer"] = AutoTokenizer.from_pretrained(model_name)
    if not os.path.exists(model_dict[model_name]["path"]):
        print("File not found:", model_dict[model_name]["path"])
        model_dict.pop(model_name)
    ckpt = torch.load(
        model_dict[model_name]["path"], map_location=model.device
    )
    # match dtype if it's float16
    for key in ckpt:
        if ckpt[key].dtype in [torch.float16, torch.float32, torch.bfloat16]:
            ckpt[key] = ckpt[key].to(model.dtype)
    model_dict[model_name]["ckpt"] = ckpt
    conv = model_dict[model_name]["conv"] = get_conversation_template(
        model_dict[model_name]["conv_template"]
    )
    if "system_prompt" in model_dict[model_name]:
        model_dict[model_name]["conv"].set_system_message(
            model_dict[model_name]["system_prompt"]
        )
    if conv.stop_str is None:
        model_dict[model_name]["stop_str"] = []
    elif isinstance(conv.stop_str, str):
        model_dict[model_name]["stop_str"] = [conv.stop_str]
    else:
        model_dict[model_name]["stop_str"] = conv.stop_str
    if conv.stop_token_ids is None:
        model_dict[model_name]["stop_token_ids"] = []
    else:
        model_dict[model_name]["stop_token_ids"] = conv.stop_token_ids
    if tokenizer.eos_token_id not in model_dict[model_name]["stop_token_ids"]:
        model_dict[model_name]["stop_token_ids"].append(tokenizer.eos_token_id)


class DataParallelModule(nn.Module):
    def __init__(self, module, weight_list):
        super().__init__()
        self.module = module
        self.weight_list = weight_list
        self.original_weight = module.weight.data

    def forward(self, hidden_states):
        # hidden_states: (B, ...)
        outputs = []
        for i in range(len(self.weight_list)):
            self.module.weight.data = self.weight_list[i]
            outputs.append(self.module(hidden_states[i, None]))

        # return torch.cat(outputs, dim=0)
        # Handle padding, padding -inf
        nt = torch.nested.as_nested_tensor([outputs[i][0] for i in range(len(outputs))])
        return torch.nested.to_padded_tensor(nt, torch.finfo(nt.dtype).min)


class DiffCompressModule(nn.Module):
    def __init__(self, module, mask_list, coeff_list):
        super().__init__()
        self.module = module
        # self.mask = torch.stack(mask_list, dim=0)
        # self.coeff = (
        #     torch.stack(coeff_list, dim=0)
        # )
        self.mask = mask_list
        self.coeff = coeff_list

    def forward(self, hidden_states):
        # hidden_states: (B, ...)
        output = self.module(hidden_states)
        # TODO: Fuse coeff
        diff = binary_bmm(hidden_states, self.mask) * self.coeff[:, None, None]
        return output + diff


# Assume batch size = len(checkpoint_list)
# Sample i uses checkpoint_list[i]

# Used to cache stack mask and coeff
cached_modules = {}

def register_diff_compress(model, checkpoint_list):
    for name, module in model.named_modules():
        # Detect leaf modules
        if len(list(module.named_children())) == 0:
            # print("-" * 50)
            if f"{name}.weight" in checkpoint_list[0]:
                # print(name, "data parallel")
                parent = model.get_submodule(".".join(name.split(".")[:-1]))
                setattr(
                    parent,
                    name.split(".")[-1],
                    DataParallelModule(
                        module,
                        [
                            checkpoint[f"{name}.weight"]
                            for checkpoint in checkpoint_list
                        ],
                    ),
                )

            elif f"{name}.mask" in checkpoint_list[0] or name in cached_modules:
                # print(name, "diff compress")
                assert isinstance(module, nn.Linear), "Only support linear layer"
                parent = model.get_submodule(".".join(name.split(".")[:-1]))
                if name not in cached_modules:
                    cached_modules[name] = (
                        torch.stack([checkpoint[f"{name}.mask"] for checkpoint in checkpoint_list], dim=0),
                        torch.stack([checkpoint[f"{name}.coeff"] for checkpoint in checkpoint_list], dim=0),
                    )
                    # clean up
                    for checkpoint in checkpoint_list:
                        checkpoint.pop(f"{name}.mask")
                        checkpoint.pop(f"{name}.coeff")
                    gc.collect()
                    torch.cuda.empty_cache()
                setattr(
                    parent,
                    name.split(".")[-1],
                    DiffCompressModule(
                        module,
                        cached_modules[name][0],
                        cached_modules[name][1],
                    ),
                )
            else:
                # print(name, "no weight")
                pass


def unregister_diff_compress(model):
    for name, module in model.named_modules():
        if isinstance(module, DataParallelModule):
            # print(name, "data parallel")
            module.module.weight.data = module.original_weight
            parent = model.get_submodule(".".join(name.split(".")[:-1]))
            setattr(parent, name.split(".")[-1], module.module)
        elif isinstance(module, DiffCompressModule):
            # print(name, "diff compress")
            parent = model.get_submodule(".".join(name.split(".")[:-1]))
            setattr(parent, name.split(".")[-1], module.module)


# context manager
class DiffCompress:
    def __init__(self, model, checkpoint_list):
        self.model = model
        self.checkpoint_list = checkpoint_list

    def __enter__(self):
        register_diff_compress(self.model, self.checkpoint_list)

    def __exit__(self, exc_type, exc_value, traceback):
        unregister_diff_compress(self.model)

# Directly use register for now
register_diff_compress(model, [model_dict[model_name]["ckpt"] for model_name in model_dict])


@app.get("/models")
def get_models():
    return list(model_dict.keys())


def streaming_generator(input_ids, attention_mask, max_new_tokens):
    with torch.inference_mode():
        stopped_pos = torch.tensor([-1] * len(model_dict)).to(model.device)
        new_token_list = []

        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )

        # TODO: extend greedy decoding
        for i in range(max_new_tokens):
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
            
            new_token_list.append(next_token)

            new_tokens = torch.stack(new_token_list, dim=1) if i > 0 else new_token_list[0].unsqueeze(1)

            response = []
            for j, model_name in enumerate(model_dict):
                if stopped_pos[j] != -1:
                    response.append(("", "stop"))

                else:
                    resp = model_dict[model_name]["tokenizer"].decode(new_tokens[j, :], skip_special_tokens=False)
                    # quick hack (this is so bad)
                    if "<|end_of_turn|>" in resp or "<|im_end|>" in resp:
                        resp = resp.split("<|")[0] + "</s>"
                    response.append((resp, "continue"))

                
            # print(response)
            # NDJSON format
            response = json.dumps({"response": response})
            # response = response.encode("utf-8")
            # print(f"RESPONSE: {response}")
            yield f"{response}\n\n"
            # yield response


            attention_mask = torch.cat(
                [attention_mask, torch.ones((attention_mask.shape[0], 1)).to(model.device)], dim=-1
            )
                
            # Check if model has stopped
            for j, model_name in enumerate(model_dict):
                # TODO: stop_str
                if next_token[j] in model_dict[model_name]["stop_token_ids"]:
                    if stopped_pos[j] == -1:
                        stopped_pos[j] = i
            
            if torch.all(stopped_pos != -1):
                break

            past_key_values = outputs.past_key_values
            outputs = model(
                next_token[:, None],
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                use_cache=True,
            )

    # free CUDA memory
    del input_ids
    del attention_mask
    del outputs
    gc.collect()
    torch.cuda.empty_cache()


@app.post("/generate")
def generate(
    messages: Union[str, List[Dict[str, str]]],
    max_new_tokens: int = 16
):
    """
    messages: list of messages in openai format
    {"role": "user", "content": "hello"}
    """
    logging.info(f"messages: {messages}")
    prompt_list = []
    input_ids_list = []
    for model_name in model_dict:
        conv = model_dict[model_name]["conv"]
        conv.messages = []
        for message in messages:
            role, content = message["role"], message["content"]
            if role == "system":
                conv.set_system_message(content)
            elif role == "user":
                conv.append_message(
                    conv.roles[0],
                    content,
                )
            else:
                conv.append_message(
                    conv.roles[1],
                    content,
                )
        assert role == "user", "The last message must be from user"
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        prompt_list.append(prompt)
        input_ids_list.append(model_dict[model_name]["tokenizer"](prompt)["input_ids"])
    print(prompt_list)
    # Left padding input_ids_list
    max_len = max([len(input_ids) for input_ids in input_ids_list])
    # use power of 2 that is larger than max_len for triton
    max_len = max(2 ** (max_len - 1).bit_length(), 64)
    if max_len > 1024:
        logging.warning(f"max_len {max_len} is larger than 1024, may cause CUDA OOM")
        return ["Error: max_len too large, please reduce the input length"] * len(model_dict)

    attention_mask_list = []
    for i, input_ids in enumerate(input_ids_list):
        attention_mask_list.append(
            [0] * (max_len - len(input_ids)) + [1] * len(input_ids)
        )
        input_ids_list[i] = [0] * (max_len - len(input_ids)) + input_ids
    input_ids = torch.tensor(input_ids_list).to(model.device)
    attention_mask = torch.tensor(attention_mask_list).to(model.device)

    # insert streaming_generator
    # return streaming_generator(input_ids, attention_mask, max_new_tokens)
    return StreamingResponse(streaming_generator(input_ids, attention_mask, max_new_tokens), media_type="text/event-stream")

# warmup triton kernels
print("Warming up triton kernels...")
for i in [32, 64, 128, 256, 512]:
    generate(
        [
            {
                "role": "user",
                "content": "hello "*i,
            }
        ],
        1,
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)