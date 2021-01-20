import argparse
import os
import random
from typing import List

import torch

from fairseq.data import Dictionary


def load_dict(path: str) -> Dictionary:
    d = Dictionary.load(path)
    # for l in langs:
    d.add_symbol("<mask>")
    return d


def main() -> None:
    parser = argparse.ArgumentParser(description="Trims pre-trained mBART model for fine-tuning.")
    parser.add_argument("--pre-train-dir", type=str, required=True, help="The pre-trained mBART model directory.")
    parser.add_argument("--ft-dict", type=str, required=True, help="The fine-tuning model dictionary.")
    args = parser.parse_args()

    pre_dict = load_dict(os.path.join(args.pre_train_dir, "dict.txt"))
    ft_dict = load_dict(args.ft_dict)
    data = torch.load(os.path.join(args.pre_train_dir, "bart_model.pt"))
    model = data["model"]

    dim = 768
    mapping: List[int] = []
    for i in range(len(ft_dict)):
        word = ft_dict[i]

        if word not in pre_dict:
            continue
        mapping.append(pre_dict.index(word))

    for name in ["encoder.embed_tokens.weight"]:
        pre_tensor: torch.Tensor = model[name]
        ft_tensor = torch.zeros(
            [len(ft_dict), dim], dtype=pre_tensor.dtype, layout=pre_tensor.layout, device=pre_tensor.device,
        )

        for ft_i, pre_i in enumerate(mapping):
            ft_tensor[ft_i] = pre_tensor[pre_i]
        
        # print(len(ft_tensor), dim)
        for i, ft_tens in enumerate(ft_tensor):
            tens = ft_tens.to('cpu').detach().numpy().copy().tolist()
            tens = [f"{x:3f}" for x in tens]
            # print(tens)
            print(ft_dict[i], " ".join(tens))



if __name__ == "__main__":
    main()