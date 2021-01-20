import argparse
import os
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
    parser.add_argument("--output", type=str, required=True, help="The trimmed mBART model.")
    parser.add_argument("--large", action='store_true', help="use large mBART")
    parser.add_argument("--muse-model",  type=str, required=True, help="muse pre-trained model")
    args = parser.parse_args()

    pre_dict = load_dict(os.path.join(args.pre_train_dir, "dict.txt"))
    ft_dict = load_dict(args.ft_dict)
    data = torch.load(os.path.join(args.pre_train_dir, "bart_model.pt"))
    model = data["model"]

    if args.large:
        dim = 1024
    else:
        dim = 768

    mapping: List[int] = []
    for i in range(len(ft_dict)):
        word = ft_dict[i]
        mapping.append(pre_dict.index(word))

    vectors = {}
    model_dtype = model["encoder.embed_tokens.weight"].dtype
    model_layout = model["encoder.embed_tokens.weight"].layout
    model_device = model["encoder.embed_tokens.weight"].device
    for line in open(args.muse_model):
        tab = line.strip().split()
        if len(tab) == 2:
            continue
        vec = torch.tensor([float(x) for x in tab[1:]], dtype=model_dtype, device=model_device)
        word = tab[0]
        idx = ft_dict.index(word)
        vectors[idx] = vec

    unk_id = pre_dict.index("<unk>")
    for name in ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "decoder.output_projection.weight"]:
        pre_tensor: torch.Tensor = model[name]
        ft_tensor = torch.zeros(
            [len(ft_dict), dim], dtype=pre_tensor.dtype, layout=pre_tensor.layout, device=pre_tensor.device,
        )
        print(ft_tensor.shape)
        for ft_i, pre_i in enumerate(mapping):
            word = ft_dict[ft_i]
            ft_tensor[ft_i] = pre_tensor[pre_i]

            if pre_i == unk_id and ft_i in vectors:
               ft_tensor[ft_i] = vectors[ft_i]
        model[name] = ft_tensor

    torch.save(data, args.output)


if __name__ == "__main__":
    main()