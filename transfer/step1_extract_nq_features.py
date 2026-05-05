# step1_extract_nq_features.py
"""
NQ Feature Extraction for Transfer Evaluation
==============================================

Runs LLaMA-2-7B-Chat on NQ-Open questions, extracting the same three
feature families as step01_extract_features.py (AggreFact):

  1. Lookback Ratio    — (L, H, T) per-layer per-head context ratio
  2. Attention Entropy — (L, H, T) per-layer per-head entropy
  3. Output Logit Feats — (T,) x3: chosen_prob, output_entropy, top_margin

Output format matches AggreFact features exactly:
  {"config": ..., "examples": [...]}
  Each example has: lookback_ratio (L,H,T), attn_entropy (L,H,T),
                    logit_chosen_prob (T,), logit_output_entropy (T,),
                    logit_top_margin (T,), label, split="test"

Labels come from nq_annotated.jsonl (GPT-4o annotations from Step 2).
All NQ examples are assigned split="test" since they are transfer-only.
"""

import json, os, torch, logging, datetime
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

login(token=os.environ["HF_TOKEN"])

# ── Paths ──
MODEL_NAME     = "meta-llama/Llama-2-7b-chat-hf"
NQ_PATH        = "/projectnb/cs505am/students/hannasam/Faithfulness-Probes-for-Summarization/data/nq-open-10_total_documents_gold_at_4.jsonl"
ANNO_PATH      = "/projectnb/cs505am/students/hannasam/Faithfulness-Probes-for-Summarization/transfer/nq_annotated.jsonl"
OUT_PATH       = "/projectnb/cs505am/students/hannasam/Faithfulness-Probes-for-Summarization/transfer/nq_features_full.pt"
MAX_NEW_TOKENS = 100

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/nq-extract-full.log",
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ── Prompt format (matches Lookback Lens Appendix B) ──
def build_prompt(question, ctxs):
    ctx_text = ""
    for i, ctx in enumerate(ctxs):
        ctx_text += f"Document [{i+1}] (Title: {ctx['title']}) {ctx['text']}\n"
    prompt = (
        f"[INST] Answer the question based on the given documents. "
        f"Only give me the answer and do not output any other words.\n\n"
        f"{ctx_text}\n"
        f"Question: {question} [/INST]"
    )
    return prompt

# ── Feature extraction helpers ──
def compute_lookback_ratio(attn, prompt_len):
    """attn: (H, seq_len) -> (H,) lookback ratio per head"""
    ctx_attn = attn[:, :prompt_len].sum(dim=-1)
    gen_attn = attn[:, prompt_len:].sum(dim=-1)
    return ctx_attn / (ctx_attn + gen_attn + 1e-9)

def compute_attn_entropy(attn):
    """attn: (H, seq_len) -> (H,) entropy per head"""
    p = attn.clamp(min=1e-9)
    return -(p * p.log()).sum(dim=-1)

def compute_logit_features(logits_step, chosen_id):
    """logits_step: (vocab_size,) -> chosen_prob, output_entropy, top_margin"""
    probs = torch.softmax(logits_step, dim=-1)
    chosen_prob = probs[chosen_id].item()
    p = probs.clamp(min=1e-9)
    output_entropy = -(p * p.log()).sum().item()
    top2 = torch.topk(probs, k=2).values
    top_margin = (top2[0] - top2[1]).item()
    return chosen_prob, output_entropy, top_margin

# ── Load model ──
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="eager",
)
model.eval()
print("Model loaded.")

# ── Load GPT-4o annotations ──
print("Loading annotations...")
labels_by_question = {}
with open(ANNO_PATH) as f:
    for line in f:
        d = json.loads(line)
        if d["label"] is not None:
            labels_by_question[d["question"]] = d["label"]
print(f"  Loaded {len(labels_by_question)} labeled examples.")

# ── Load NQ data ──
print("Loading NQ data...")
nq_examples = []
with open(NQ_PATH) as f:
    for line in f:
        ex = json.loads(line)
        if ex["question"] in labels_by_question:
            nq_examples.append(ex)
print(f"  {len(nq_examples)} examples with labels.")

# ── Resume support ──
results = []
done_questions = set()
if os.path.exists(OUT_PATH):
    print(f"Resuming from {OUT_PATH}...")
    blob = torch.load(OUT_PATH, weights_only=False)
    results = blob["examples"] if isinstance(blob, dict) else blob
    done_questions = {r["question"] for r in results}
    print(f"  {len(done_questions)} examples already done.")

# ── Main extraction loop ──
with torch.no_grad():
    for i, ex in enumerate(tqdm(nq_examples)):
        question = ex["question"]

        if question in done_questions:
            continue

        label = labels_by_question[question]
        prompt = build_prompt(question, ex["ctxs"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        try:
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                output_attentions=True,
                output_scores=True,
                return_dict_in_generate=True,
            )
        except Exception as e:
            logging.error(f"Generation failed for: {question[:60]} — {e}")
            continue

        generated_ids = out.sequences[0][prompt_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        T = len(out.attentions)
        if T == 0:
            logging.warning(f"Empty generation for: {question[:60]}")
            continue

        L = len(out.attentions[0])
        H = out.attentions[0][0].shape[1]

        lookback_steps   = []
        entropy_steps    = []
        chosen_probs     = []
        output_entropies = []
        top_margins      = []

        for t, (step_attns, step_scores) in enumerate(zip(out.attentions, out.scores)):
            lr_per_layer  = []
            ent_per_layer = []
            for layer_attn in step_attns:
                attn = layer_attn[0, :, 0, :]  # (H, seq_len)
                lr_per_layer.append(compute_lookback_ratio(attn, prompt_len))
                ent_per_layer.append(compute_attn_entropy(attn))

            lookback_steps.append(torch.stack(lr_per_layer))   # (L, H)
            entropy_steps.append(torch.stack(ent_per_layer))   # (L, H)

            chosen_id = generated_ids[t].item()
            logits = step_scores[0]  # (vocab_size,)
            cp, oe, tm = compute_logit_features(logits, chosen_id)
            chosen_probs.append(cp)
            output_entropies.append(oe)
            top_margins.append(tm)

        # Stack to (L, H, T) — matches AggreFact format exactly
        lookback_ratio        = torch.stack(lookback_steps, dim=-1).cpu()   # (L, H, T)
        attn_entropy          = torch.stack(entropy_steps,  dim=-1).cpu()   # (L, H, T)
        logit_chosen_prob     = torch.tensor(chosen_probs,     dtype=torch.float32)
        logit_output_entropy  = torch.tensor(output_entropies, dtype=torch.float32)
        logit_top_margin      = torch.tensor(top_margins,      dtype=torch.float32)

        results.append({
            "lookback_ratio":       lookback_ratio,
            "attn_entropy":         attn_entropy,
            "logit_chosen_prob":    logit_chosen_prob,
            "logit_output_entropy": logit_output_entropy,
            "logit_top_margin":     logit_top_margin,
            "question":             question,
            "generated":            generated_text,
            "label":                label,
            "split":                "test",  # all NQ = transfer/eval only
            "data_index":           i,
        })
        done_questions.add(question)

        if (i + 1) % 50 == 0:
            logging.info(f"Processed {i+1}/{len(nq_examples)} | "
                         f"Q: {question[:60]} | Gen: {generated_text[:60]}")

        if (i + 1) % 500 == 0:
            torch.save({"config": {"created_at": datetime.datetime.now().isoformat()},
                        "examples": results}, OUT_PATH)
            logging.info(f"Checkpoint saved at {i+1}")

# ── Final save ──
config = {
    "model_name":       MODEL_NAME,
    "data_type":        "nq_open",
    "data_path":        NQ_PATH,
    "annotation_path":  ANNO_PATH,
    "max_new_tokens":   MAX_NEW_TOKENS,
    "teacher_forcing":  False,
    "feature_families": [
        "lookback_ratio", "attn_entropy",
        "logit_chosen_prob", "logit_output_entropy", "logit_top_margin",
    ],
    "created_at":  datetime.datetime.now().isoformat(),
    "n_examples":  len(results),
}
torch.save({"config": config, "examples": results}, OUT_PATH)
logging.info(f"Finished. Saved {len(results)} examples to {OUT_PATH}")
print(f"\nDone. Saved {len(results)} examples to {OUT_PATH}")

if results:
    r = results[0]
    L, H, T = r["lookback_ratio"].shape
    labels = [r["label"] for r in results]
    print(f"  Layers={L}, Heads={H}, First example T={T}")
    print(f"  Faithful: {sum(labels)} | Hallucinated: {len(labels)-sum(labels)}")