"""Stage: stetholm. Tier: big_gpu (envs/gpu). StethoLM (Wang et al., TMLR 2026; hf askyishan/StethoLM @ ecc9a5c1),
the cardiopulmonary audio-LLM baseline the reviewers asked for (R1-W1).

The authors' code repository (github.com/askyishan/StethoLM) is not public, so inference is rebuilt from the
model card and the checkpoint layout:
  audio  : COLA encoder = cnn1 (1->3 conv) + EfficientNet-B0 (same as OPERA-CE), fine-tuned weights from the
           checkpoint; input = OPERA mel front-end (16 kHz, 64 mels), mean-pooled 1280-d feature
  prefix : 3-layer MLP 1280 -> 5120 -> 10240 -> 10240, reshaped to 4 prefix tokens x 2560, prepended to the
           token embeddings of the chat-templated instruction
  LM     : google/medgemma-4b-it with the checkpoint's LoRA (r=8, alpha=32, q/k/v/o) merged into the weights
Unknowns (projector activation, mel f_max, prefix position) are calibrated on StethoBench's OWN test split
(--calibrate), never on our test queries.

Usage (GPU env):
  $GPY scripts/stetholm.py --calibrate
  $GPY scripts/stetholm.py --queue <dir>/queue.jsonl --name stetholm   (native-instruction run, see main())
"""
from __future__ import annotations

import argparse
import glob
import json
import os
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
REPO, REV = "askyishan/StethoLM", "ecc9a5c1325fc5f9d63e9c31db15d7b10b0ffdbb"
BENCH_REV = "943a321c514780daf75e94aa608eab062496a317"
BASE = "google/medgemma-4b-it"
N_PREFIX, D_LM = 4, 2560
SR = 16000


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        from efficientnet_pytorch import EfficientNet
        self.cnn1 = nn.Conv2d(1, 3, kernel_size=3)
        self.efficientnet = EfficientNet.from_name("efficientnet-b0", include_top=False)

    def forward(self, x):                       # x: [B, T, 64]
        return self.efficientnet(self.cnn1(x.unsqueeze(1))).flatten(1)


def projector(act: str) -> nn.Sequential:
    A = {"gelu": nn.GELU, "relu": nn.ReLU, "tanh": nn.Tanh}[act]
    return nn.Sequential(nn.Linear(1280, 5120), A(), nn.Dropout(0.0), nn.Linear(5120, 10240), A(), nn.Dropout(0.0),
                         nn.Linear(10240, N_PREFIX * D_LM))


def mel(path: str, f_max: int, max_sec: float) -> np.ndarray:
    x, _ = librosa.load(path, sr=SR, duration=max_sec)
    if len(x) < SR:
        x = np.tile(x, int(np.ceil(SR / max(len(x), 1))))[:SR]
    S = librosa.power_to_db(librosa.feature.melspectrogram(y=x, sr=SR, n_mels=64, fmin=50, fmax=f_max,
                                                           n_fft=1024, hop_length=512), ref=np.max)
    S = (S - S.min()) / (S.max() - S.min()) if S.max() > S.min() else S
    return S.T.astype(np.float32)


class StethoLM:
    def __init__(self, act: str = "gelu"):
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
        ck = torch.load(hf_hub_download(REPO, "stetholm_adapter.pt", revision=REV, cache_dir=HF), map_location="cpu")
        self.proc = AutoTokenizer.from_pretrained(BASE, cache_dir=HF)
        self.lm = Gemma3ForConditionalGeneration.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda",
                                                                 cache_dir=HF).eval()
        n = self._merge_lora(ck)
        self.enc = Encoder()
        self.enc.load_state_dict({k[len("audio_encoder."):]: v for k, v in ck.items() if k.startswith("audio_encoder.")})
        self.enc = self.enc.eval().cuda()
        self.ck, self.n_lora = ck, n
        self.set_act(act)

    def set_act(self, act: str):
        self.proj = projector(act)
        self.proj.load_state_dict({k[len("prefix_project.model."):]: v for k, v in self.ck.items()
                                   if k.startswith("prefix_project.")})
        self.proj = self.proj.eval().cuda()

    def _merge_lora(self, ck) -> int:
        """W += (alpha / r) * B @ A for every language-model LoRA pair stored under 'lm.'"""
        lm = self.lm.model.language_model
        scale, n = 32 / 8, 0
        for k in ck:
            if k.startswith("lm.") and k.endswith("lora_A.default.weight"):
                mod_path = k[len("lm."): -len(".lora_A.default.weight")]
                A, B = ck[k].float(), ck[k.replace("lora_A", "lora_B")].float()
                lin = lm.get_submodule(mod_path)
                lin.weight.data += (scale * B @ A).to(lin.weight.dtype).to(lin.weight.device)
                n += 1
        return n

    def _greedy(self, x: torch.Tensor, max_new: int) -> list[int]:
        """Greedy decoding from input embeddings (generate() does not accept inputs_embeds for Gemma3-VL)."""
        stop = {self.proc.eos_token_id, self.proc.convert_tokens_to_ids("<end_of_turn>")}
        r = self.lm(inputs_embeds=x, use_cache=True)
        past, toks = r.past_key_values, []
        nxt = r.logits[:, -1].argmax(-1)
        for _ in range(max_new):
            t = int(nxt)
            if t in stop:
                break
            toks.append(t)
            r = self.lm(input_ids=nxt[:, None], past_key_values=past, use_cache=True)
            past, nxt = r.past_key_values, r.logits[:, -1].argmax(-1)
        return toks

    @torch.inference_mode()
    def ids(self, prompt: str, template: str) -> torch.Tensor:
        if template == "chat":
            msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            return self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                                 return_tensors="pt")["input_ids"]
        text = {"plain": prompt, "qa": f"Question: {prompt} Answer:"}[template]
        return self.proc(text, return_tensors="pt").input_ids          # <bos> + text

    def generate(self, paths: list[str], prompts: list[str], f_max=8000, max_sec=30.0, place="before_bos",
                 max_new=256, template="chat") -> list[str]:
        feats = [torch.from_numpy(mel(p, f_max, max_sec)) for p in paths]
        out = []
        for f, prompt in zip(feats, prompts):   # variable-length mels: one at a time (the 4B LM is fast)
            a = self.enc(f[None].cuda())
            pre = self.proj(a).view(1, N_PREFIX, D_LM).to(torch.bfloat16)
            ids = self.ids(prompt, template).cuda()
            emb = self.lm.get_input_embeddings()(ids)
            x = torch.cat([pre, emb], 1) if place == "before_bos" else torch.cat([emb[:, :1], pre, emb[:, 1:]], 1)
            out.append(self.proc.decode(self._greedy(x, max_new), skip_special_tokens=True).strip())
        return out


def bench_items(n_per: int = 30):
    """StethoBench TEST-split binary items whose audio we hold: CirCor murmur yes/no, ICBHI/SPR normal/abnormal."""
    from huggingface_hub import hf_hub_download
    b = hf_hub_download("askyishan/StethoBench", "stethobench.jsonl", repo_type="dataset", revision=BENCH_REV, cache_dir=HF)
    L = [json.loads(l) for l in open(b)]
    roots = {"heart-circor": RAW_ROOT / "circor" / "circor-heart-sound-1.0.3" / "training_data",
             "icbhi": RAW_ROOT / "icbhi" / "ICBHI_final_database"}
    rng = np.random.default_rng(0)
    items = []
    for ds, pat, neg in (("heart-circor", r"murmur present\?", r"\bno murmur|not present|absent"),
                         ("icbhi", r"normal or abnormal", r"\bis normal|\bnormal\b(?!ity)")):
        xs = [x for x in L if x["dataset"] == ds and x["split"] == "test" and re.search(pat, x["instruction"], re.I)
              and (roots[ds] / x["filename"]).exists()]
        for i in rng.permutation(len(xs))[:n_per]:
            x = xs[i]
            items.append({**x, "path": str(roots[ds] / x["filename"]),
                          "gold": "neg" if re.search(neg, x["response"], re.I) else "pos", "neg_re": neg})
    return items


def calibrate() -> None:
    items = bench_items()
    print(f"{len(items)} StethoBench test items; gold pos rate {np.mean([i['gold'] == 'pos' for i in items]):.2f}", flush=True)
    m = StethoLM("gelu")
    print(f"merged {m.n_lora} LoRA pairs", flush=True)
    rows = []
    for act in ("gelu", "relu"):
        m.set_act(act)
        for template in ("chat", "plain", "qa"):
            for place in ("before_bos", "after_bos"):
                t0 = time.time()
                outs = m.generate([i["path"] for i in items], [i["instruction"] for i in items], place=place,
                                  template=template, max_new=40)
                pred = ["neg" if re.search(i["neg_re"], o, re.I) else "pos" for i, o in zip(items, outs)]
                acc = {ds: float(np.mean([p == i["gold"] for p, i in zip(pred, items) if i["dataset"] == ds]))
                       for ds in ("heart-circor", "icbhi")}
                rows.append({"act": act, "template": template, "place": place, **acc, "sec": round(time.time() - t0),
                             "examples": outs[:3]})
                print({k: v for k, v in rows[-1].items() if k != "examples"}, "| e.g.", [o[:70] for o in outs[:2]], flush=True)
    out = ARTIFACT_ROOT / "llm" / "stetholm_calibration.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    a = ap.parse_args()
    if a.calibrate:
        calibrate()


if __name__ == "__main__":
    main()
