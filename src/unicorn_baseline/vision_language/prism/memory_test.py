import argparse

import torch
from transformers import AutoModel

from .perceiver import PerceiverResampler

parser = argparse.ArgumentParser()
parser.add_argument("b", type=int, help="batch size", default=4)
parser.add_argument("n", type=int, help="number of tiles", default=100_000)

args = parser.parse_args()

print(args)

with torch.autocast("cuda", torch.float16), torch.inference_mode():
    prism = AutoModel.from_pretrained("paige-ai/Prism", trust_remote_code=True)

    net = prism.image_resampler.to("cuda")
    assert isinstance(net, PerceiverResampler), type(net)

    mem = 0
    for p in net.parameters():
        acc = 1
        for s in p.data.shape:
            acc *= s
        mem += acc

    print("number of parameters: ", round(mem / 1e6, 1), "mn")

    print("init alloc:", torch.cuda.max_memory_allocated("cuda") / 1024**3, "GB")
    print("init resrv:", torch.cuda.max_memory_reserved("cuda") / 1024**3, "GB")

    batch_size = args.b
    seq_len = args.n
    e = torch.randn((batch_size, seq_len, 2560), device="cuda", dtype=torch.float16)
    m = torch.ones((batch_size, seq_len), device="cuda", dtype=torch.bool)

    print("data alloc:", torch.cuda.max_memory_allocated("cuda") / 1024**3, "GB")
    print("data resrv:", torch.cuda.max_memory_reserved("cuda") / 1024**3, "GB")

    net(tile_embeddings=e, tile_mask=m)

    print("forward alloc:", torch.cuda.max_memory_allocated("cuda") / 1024**3, "GB")
    print("forward resrv:", torch.cuda.max_memory_reserved("cuda") / 1024**3, "GB")
