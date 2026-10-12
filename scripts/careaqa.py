"""Stage: careaqa. Tier: big_gpu (envs/gpu). CaReAQA (Wang et al., CHIL 2025; hf tsnngw/CaReAQA @ cae506aa), a
cardiac-and-respiratory audio question-answering LLM, as a dedicated medical audio-LLM baseline (R1-W1).

The code repository (github.com/tsnng/CaReAQA) is not public; the model card gives the inference recipe
(preprocess_audio, generate_answer) which is followed exactly. Architecture from the checkpoint layout:
  audio  : OPERA COLA encoder (cnn1 + EfficientNet-B0), 1280-d feature (extract_feature dim=1280)
  mapper : ClipCap TransformerMapper (linear 1280 -> 3072, 8 learned prefix constants, 8 self-attention layers,
           8 heads, ReLU MLP x2), clip_length 1, prefix_length 8
  LLM    : Llama-3.2-3B with LoRA (r=8) on q/k/v/o/gate/up/down; base weights are in the checkpoint
           (config + tokenizer from the ungated mirror unsloth/Llama-3.2-3B-Instruct)
Unknowns (LoRA alpha, which eos id fills the prefix slots) are calibrated on CaReSound's OWN test split
(ICBHI diagnosis questions), never on our test queries.

Usage: $GPY scripts/careaqa.py --calibrate
       $GPY scripts/careaqa.py --queue <queue.jsonl> --alpha A --eos E     (see run_queue)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auscultrag.config import ARTIFACT_ROOT, RAW_ROOT  # noqa: E402

HF = RAW_ROOT / "hf_models"
REPO, REV = "tsnngw/CaReAQA", "cae506aa9c8b0ee835ee8005f48eccda284b712b"
LLAMA = "unsloth/Llama-3.2-3B-Instruct"
P, D = 8, 3072


class MHA(nn.Module):                                   # ClipCap MultiHeadAttention
    def __init__(self, d, h=8):
        super().__init__()
        self.h, self.scale = h, (d // h) ** -0.5
        self.to_queries = nn.Linear(d, d, bias=False)
        self.to_keys_values = nn.Linear(d, 2 * d, bias=False)
        self.project = nn.Linear(d, d)

    def forward(self, x):
        b, n, c = x.shape
        q = self.to_queries(x).reshape(b, n, self.h, c // self.h)
        kv = self.to_keys_values(x).reshape(b, n, 2, self.h, c // self.h)
        k, v = kv[:, :, 0], kv[:, :, 1]
        att = torch.einsum("bnhd,bmhd->bnmh", q, k) * self.scale
        att = att.softmax(dim=2)
        return self.project(torch.einsum("bnmh,bmhd->bnhd", att, v).reshape(b, n, c))


class Mlp(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(d, 2 * d), nn.Linear(2 * d, d)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class Layer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm1, self.attn, self.norm2, self.mlp = nn.LayerNorm(d), MHA(d), nn.LayerNorm(d), Mlp(d)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class Mapper(nn.Module):                                # ClipCap TransformerMapper, clip_length = 1
    def __init__(self, n_layers=8):
        super().__init__()
        self.linear = nn.Linear(1280, D)
        self.prefix_const = nn.Parameter(torch.zeros(P, D))
        self.transformer = nn.Module()
        self.transformer.layers = nn.ModuleList([Layer(D) for _ in range(n_layers)])

    def forward(self, a):
        x = torch.cat([self.linear(a).view(a.shape[0], 1, D), self.prefix_const[None].expand(a.shape[0], -1, -1)], 1)
        for l in self.transformer.layers:
            x = l(x)
        return x[:, 1:]


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        from efficientnet_pytorch import EfficientNet
        self.cnn1 = nn.Conv2d(1, 3, kernel_size=3)
        self.efficientnet = EfficientNet.from_name("efficientnet-b0", include_top=False)

    def forward(self, x):
        return self.efficientnet(self.cnn1(x.unsqueeze(1))).flatten(1)


def preprocess(path: str) -> torch.Tensor:              # model card preprocess_audio, verbatim
    x, sr = librosa.load(path, sr=16000)
    m = librosa.power_to_db(librosa.feature.melspectrogram(y=x, sr=sr, n_fft=1024, hop_length=512, n_mels=64), ref=np.max)
    return torch.tensor(m, dtype=torch.float32).unsqueeze(0)


class CaReAQA:
    def __init__(self, alpha: float = 16.0):
        from huggingface_hub import hf_hub_download
        from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM
        sd = torch.load(hf_hub_download(REPO, "CaReAQAmodel.pt", revision=REV, cache_dir=HF), map_location="cpu")
        self.tok = AutoTokenizer.from_pretrained(LLAMA, cache_dir=HF)
        cfg = AutoConfig.from_pretrained(LLAMA, cache_dir=HF)
        n_layers = 1 + max(int(k.split(".layers.")[1].split(".")[0]) for k in sd if k.startswith("llm.") and ".layers." in k)
        assert cfg.num_hidden_layers == n_layers and cfg.hidden_size == D, (cfg.num_hidden_layers, n_layers)
        cfg.tie_word_embeddings = False
        self.llm = LlamaForCausalLM(cfg)
        self.base, self.lora = {}, {}
        pre = "llm.base_model.model."
        for k, v in sd.items():
            if not k.startswith(pre):
                continue
            k2 = k[len(pre):]
            if ".lora_A." in k2 or ".lora_B." in k2:
                self.lora[k2] = v
            else:
                self.base[k2.replace(".base_layer.", ".")] = v
        self.llm.load_state_dict(self.base, strict=True)
        self.llm = self.llm.to(torch.bfloat16).cuda().eval()
        self._orig = {}
        self.set_alpha(alpha)
        self.enc = Encoder()
        self.enc.load_state_dict({k[len("audio_model.encoder."):]: v for k, v in sd.items() if k.startswith("audio_model.encoder.")})
        self.enc = self.enc.eval().cuda()
        self.map = Mapper()
        self.map.load_state_dict({k[len("prefix_project."):]: v for k, v in sd.items() if k.startswith("prefix_project.")})
        self.map = self.map.eval().cuda()

    @torch.no_grad()
    def set_alpha(self, alpha: float):
        """W = W0 + (alpha / r) B A for every LoRA pair (W0 restored first, so alpha can be re-set)."""
        for k in self.lora:
            if ".lora_A." not in k:
                continue
            mod = k.split(".lora_A.")[0]
            lin = self.llm.get_submodule(mod)
            if mod not in self._orig:
                self._orig[mod] = lin.weight.data.clone()
            A, B = self.lora[k].float().cuda(), self.lora[k.replace("lora_A", "lora_B")].float().cuda()
            lin.weight.data = (self._orig[mod].float() + (alpha / A.shape[0]) * B @ A).to(torch.bfloat16)
        self.alpha = alpha

    @torch.inference_mode()
    def answer(self, path: str, question: str, layout: str = "q_p_answer:", max_new: int = 60) -> str:
        """Greedy decoding over input embeddings. Layouts (calibrated):
          card        : model-card generate_answer verbatim ("question: Q" [prefix over " answer" + eos slots] eos)
          q_p_answer  : "question: Q" [prefix] " answer"
          q_p_answer: : "question: Q" [prefix] " answer:"
          p_q_answer: : [prefix] "question: Q answer:"   """
        a = self.enc(preprocess(path).cuda())
        pre = self.map(a).to(torch.bfloat16)
        E = self.llm.get_input_embeddings()
        enc = lambda t: E(torch.tensor([self.tok.encode(t, add_special_tokens=False)], device="cuda"))
        if layout == "card":
            qp = self.tok.encode("question: ", add_special_tokens=False)
            qt = self.tok.encode(question, add_special_tokens=False)
            ap = self.tok.encode(" answer", add_special_tokens=False)
            eos = self.tok.convert_tokens_to_ids("<|end_of_text|>")
            e = E(torch.tensor([qp + qt + ap + [eos] * P], device="cuda")).clone()
            e[0, len(qp + qt): len(qp + qt) + P] = pre[0]
        elif layout == "q_p_answer":
            e = torch.cat([enc(f"question: {question}"), pre, enc(" answer")], 1)
        elif layout == "q_p_answer:":
            e = torch.cat([enc(f"question: {question}"), pre, enc(" answer:")], 1)
        elif layout == "p_q_answer:":
            e = torch.cat([pre, enc(f"question: {question} answer:")], 1)
        else:
            raise ValueError(layout)
        r = self.llm(inputs_embeds=e, use_cache=True)
        past, nxt, out = r.past_key_values, r.logits[:, -1].argmax(-1), []
        stops = {self.tok.convert_tokens_to_ids("<|eot_id|>"), self.tok.convert_tokens_to_ids("<|end_of_text|>")}
        for _ in range(max_new):
            t = int(nxt)
            if t in stops:
                break
            out.append(t)
            r = self.llm(input_ids=nxt[:, None], past_key_values=past, use_cache=True)
            past, nxt = r.past_key_values, r.logits[:, -1].argmax(-1)
        return self.tok.decode(out, skip_special_tokens=True).strip()


DIAG = ["COPD", "URTI", "Bronchiectasis", "Pneumonia", "Bronchiolitis", "Asthma", "LRTI", "Healthy"]


def calib_items(n: int = 80):
    """CaReSound TEST-split ICBHI diagnosis questions; audio = the patient's first recording."""
    import pandas as pd
    from huggingface_hub import hf_hub_download
    d = pd.read_csv(hf_hub_download("tsnngw/CaReSound", "CaReSoundQA_test.csv", repo_type="dataset", cache_dir=HF))
    d = d[(d.dataset == "ICBHI") & d.question.str.contains("diagnosis", case=False)]
    root = RAW_ROOT / "icbhi" / "ICBHI_final_database"
    items, seen = [], set()
    for _, r in d.sample(frac=1, random_state=0).iterrows():
        pid = str(r.patient_id)
        g = [x for x in DIAG if re.search(rf"\b{x}\b", r.answer, re.I)]
        files = sorted(root.glob(f"{pid}_*.wav"))
        if pid in seen or len(g) != 1 or not files:
            continue
        seen.add(pid)
        items.append({"path": str(files[0]), "question": r.question, "gold": g[0], "answer": r.answer})
        if len(items) == n:
            break
    return items


def calibrate() -> None:
    items = calib_items()
    import collections
    print(f"{len(items)} CaReSound test items; gold {collections.Counter(i['gold'] for i in items)}", flush=True)
    m = CaReAQA()
    rows = []
    for alpha in (8.0, 16.0, 32.0):
        m.set_alpha(alpha)
        for layout in ("card", "q_p_answer", "q_p_answer:", "p_q_answer:"):
            t0 = time.time()
            outs = [m.answer(i["path"], i["question"], layout) for i in items]
            pred = [next((x for x in DIAG if re.search(rf"\b{x}\b", o, re.I)), None) for o in outs]
            acc = float(np.mean([p == i["gold"] for p, i in zip(pred, items)]))
            rows.append({"alpha": alpha, "layout": layout, "acc": acc, "pred_dist": dict(collections.Counter(pred)),
                         "sec": round(time.time() - t0), "examples": outs[:4]})
            print({k: v for k, v in rows[-1].items() if k != "examples"}, "| e.g.", [o[:60] for o in outs[:3]], flush=True)
    (ARTIFACT_ROOT / "llm" / "careaqa_calibration.json").write_text(json.dumps(rows, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    a = ap.parse_args()
    if a.calibrate:
        calibrate()


if __name__ == "__main__":
    main()
