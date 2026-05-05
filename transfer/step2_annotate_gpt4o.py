# step2_annotate_gpt4o.py
import json, os, time, torch, logging, tiktoken
from openai import OpenAI
from tqdm import tqdm

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

FEATURES_PATH = "/projectnb/cs505am/students/hannasam/Faithfulness-Probes-for-Summarization/nq_features.pt"
NQ_PATH       = "/projectnb/cs505am/students/hannasam/Faithfulness-Probes-for-Summarization/data/nq-open-10_total_documents_gold_at_4.jsonl"
OUT_PATH      = "/projectnb/cs505am/students/hannasam/Faithfulness-Probes-for-Summarization/transfer/nq_annotated.jsonl"

logging.basicConfig(
    filename="logs/nq-annotate-py.log",
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Exact prompts from Lookback Lens Appendix B / step02_eval_gpt4o.py
EVAL_PROMPT_BEFORE = "You will be provided with a document and a proposed answer to a question. Your task is to determine if the proposed answer can be directly inferred from the document. If the answer contains any information not found in the document, it is considered false. Even if the answer is different from a ground truth answer, it might still be true, as long as it doesn't contain false information.\nFor each proposed answer, explain why it is true or false based on the information from the document. Focus only on the original document's content, disregarding any external context.\nAfter your explanation, give your final conclusion as **Conclusion: True** if the proposed answer is completely accurate based on the document, or **Conclusion: False** if it contains any incorrect or unsupported information. If your conclusion is 'False', identify the exact phrases or name entities from the answer that is incorrect by stating **Problematic Spans: [the inaccurate text spans from the answer, in Python list of strings format]**."

EVAL_PROMPT_AFTER = "Write your explanation first, and then give your final conclusion as **Conclusion: True** if the proposed answer is completely accurate based on the document, or **Conclusion: False** if it contains any incorrect or unsupported information. Add **Problematic Spans: [the exact inaccurate text spans from the answer, in a list of strings]** if your conclusion is 'False'."

tokenizer = tiktoken.get_encoding("o200k_base")

def build_context(nq_item):
    """Match Lookback Lens: use 1 neg + 1 pos + 1 neg context (not all 10)"""
    ctxs = nq_item['ctxs']
    pos_ctxs = [ctx for ctx in ctxs if ctx.get('hasanswer', False)]
    neg_ctxs = [ctx for ctx in ctxs if not ctx.get('hasanswer', False)]
    assert len(pos_ctxs) > 0, "No positive context found"
    assert len(neg_ctxs) >= 2, "Need at least 2 negative contexts"
    context = neg_ctxs[0]['text'] + '\n' + pos_ctxs[0]['text'] + '\n' + neg_ctxs[1]['text']
    return context

def evaluate_one(document, gt_response, generated_answer):
    """Exact Lookback Lens evaluation logic"""
    prompt = (
        f"{EVAL_PROMPT_BEFORE}\n\n"
        f"#Document#: {document}\n\n"
        f"#Ground Truth Answers (a list of valid answers)#: {gt_response}\n\n"
        f"#Proposed Answer#: {generated_answer}\n\n"
        f"{EVAL_PROMPT_AFTER}"
    )

    input_tokens = len(tokenizer.encode(prompt))

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-2024-05-13",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            text = resp.choices[0].message.content
            output_tokens = len(tokenizer.encode(text))
            cost = (input_tokens / 1_000_000 * 5) + (output_tokens / 1_000_000 * 15)

            # Parse problematic spans (exact Lookback Lens logic)
            problematic_spans = []
            if "Problematic Spans: " in text:
                spans_text = text.split('Problematic Spans: ')[1]
                if '**' in spans_text:
                    spans_text = spans_text.split('**')[0].strip()
                try:
                    problematic_spans = eval(spans_text)
                except:
                    problematic_spans = spans_text[1:-1].split(', ')

            # Parse Conclusion: True/False (exact Lookback Lens logic)
            decision = None
            if "Conclusion: " in text:
                dec = text.split('Conclusion: ')[1]
                if '**' in dec:
                    dec = dec.split('**')[0]
                if "True" in dec:
                    decision = 1
                elif "False" in dec:
                    decision = 0

            return decision, text, problematic_spans, cost

        except Exception as e:
            logging.error(f"API error on attempt {attempt+1}: {e}")
            print(f"API error on attempt {attempt+1}: {e}")
            time.sleep(5)

    return None, "", [], 0.0  # all retries failed

# Load features
print("Loading features...")
data = torch.load(FEATURES_PATH, weights_only=False)
logging.info(f"Loaded {len(data)} examples from {FEATURES_PATH}")

# Load NQ file
print("Loading NQ file...")
nq_by_question = {}
nq_list = []
with open(NQ_PATH) as f:
    for line in f:
        ex = json.loads(line)
        nq_by_question[ex['question']] = ex
        nq_list.append(ex)

# Resume support — skip already annotated by index
already_done = set()
if os.path.exists(OUT_PATH):
    with open(OUT_PATH) as f:
        for line in f:
            d = json.loads(line)
            already_done.add(d["index"])
    logging.info(f"Resuming — {len(already_done)} examples already annotated")
    print(f"Resuming — skipping {len(already_done)} already annotated examples")

total_cost = 0.0
n_failed = 0
n_correct = 0
n_total = 0

with open(OUT_PATH, "a") as out_f:
    for i, item in enumerate(tqdm(data)):
        if i in already_done:
            continue

        question = item["question"]

        if question not in nq_by_question:
            logging.warning(f"Question not found in NQ file: {question[:60]}")
            continue

        nq_item = nq_by_question[question]

        try:
            context = build_context(nq_item)
        except AssertionError as e:
            logging.warning(f"Context build failed for: {question[:60]} — {e}")
            continue

        gt_answers = str(nq_item['answers'])
        generated = item["generated"]

        decision, explanation, problematic_spans, cost = evaluate_one(context, gt_answers, generated)
        total_cost += cost
        n_total += 1

        if decision is None:
            n_failed += 1
            logging.error(f"Failed to parse decision for: {question[:60]}")
        elif decision == 1:
            n_correct += 1

        out_f.write(json.dumps({
            "index":             i,
            "question":          question,
            "document":          context.strip(),
            "ground_truth":      gt_answers.strip(),
            "generated":         generated,
            "label":             decision,       # 1=faithful, 0=hallucinated, None=failed
            "gpt4_explanation":  explanation,
            "problematic_spans": problematic_spans,
            "cost":              cost,
        }) + "\n")
        out_f.flush()

        if (i + 1) % 100 == 0:
            acc = n_correct / n_total if n_total > 0 else 0
            logging.info(f"Annotated {i+1}/{len(data)} | cost: ${total_cost:.2f} | acc: {acc:.3f} | failed: {n_failed}")
            print(f"[{i+1}/{len(data)}] Cost: ${total_cost:.2f} | Accuracy: {acc:.3f} | Failed: {n_failed}")

acc = n_correct / n_total if n_total > 0 else 0
logging.info(f"Finished. Total cost: ${total_cost:.2f} | Accuracy: {acc:.3f} | Failed: {n_failed}/{n_total}")
print(f"\nDone. Total cost: ${total_cost:.2f} | Accuracy: {acc:.3f} | Failed: {n_failed}/{n_total}")