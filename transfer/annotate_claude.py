"""
Step 2: Claude-based Faithfulness Annotation for NQ-Open
=========================================================

Drop-in replacement for step02_eval_gpt4o.py using the Anthropic SDK instead
of OpenAI.  Prompt structure, parsing logic, output JSONL schema, and resume
behaviour are all identical to the original so that step03 can consume the
output unchanged.

Output JSONL schema (one JSON object per line):
    {
      "index":              int,
      "document":           str,
      "ground_truth":       str,
      "response":           str,
      "decision":           bool | null,
      "claude_explanation": str,
      "problematic_spans":  list[str],
      "cost_approx":        float
    }

Usage
-----
    export ANTHROPIC_API_KEY=sk-ant-...

    python step02_annotate_claude.py \\
        --hyp features_nq_llama2_7b.pt \\
        --ref data/nq-open-test.jsonl \\
        --out anno_nq_claude.jsonl

    # Resume a partial run (existing output file is safe to reuse):
    python step02_annotate_claude.py \\
        --hyp features_nq_llama2_7b.pt \\
        --ref data/nq-open-test.jsonl \\
        --out anno_nq_claude.jsonl

    # Debug (5 examples, verbose):
    python step02_annotate_claude.py \\
        --hyp features_nq_llama2_7b.pt \\
        --ref data/nq-open-test.jsonl \\
        --out anno_debug.jsonl \\
        --limit 5 --debug
"""

import argparse
import json
import os
import re
import time

import anthropic
import torch

# ---------------------------------------------------------------------------
#  Prompt templates  (identical to original step02_eval_gpt4o.py)
# ---------------------------------------------------------------------------

eval_prompt_before = {
    'nq_open': (
        "You will be provided with a document and a proposed answer to a question. "
        "Your task is to determine if the proposed answer can be directly inferred "
        "from the document. If the answer contains any information not found in the "
        "document, it is considered false. Even if the answer is different from a "
        "ground truth answer, it might still be true, as long as it doesn't contain "
        "false information.\n"
        "For each proposed answer, explain why it is true or false based on the "
        "information from the document. Focus only on the original document's "
        "content, disregarding any external context.\n"
        "After your explanation, give your final conclusion as "
        "**Conclusion: True** if the proposed answer is completely accurate based "
        "on the document, or **Conclusion: False** if it contains any incorrect or "
        "unsupported information. If your conclusion is 'False', identify the exact "
        "phrases or name entities from the answer that is incorrect by stating "
        "**Problematic Spans: [the inaccurate text spans from the answer, in Python "
        "list of strings format]**."
    ),
}

eval_prompt_after = {
    'nq_open': (
        "Write your explanation first, and then give your final conclusion as "
        "**Conclusion: True** if the proposed answer is completely accurate based "
        "on the document, or **Conclusion: False** if it contains any incorrect or "
        "unsupported information. Add "
        "**Problematic Spans: [the exact inaccurate text spans from the answer, "
        "in a list of strings]** if your conclusion is 'False'."
    ),
}

data_response_names = {
    'nq_open': 'Answer',
}

data_response_names_gt = {
    'nq_open': 'Answers (a list of valid answers)',
}

# ---------------------------------------------------------------------------
#  Data loaders  (identical structure to original step02_eval_gpt4o.py)
# ---------------------------------------------------------------------------

def load_nq_open(file_path, debug=False, subsample=None):
    """Return dict keyed by data_index with context + ground-truth answers."""
    list_data_dict = {}
    is_train = 'nq_train' in file_path
    with open(file_path, 'r', encoding='utf-8') as f:
        data = []
        for line in f:
            data.append(json.loads(line))
        if debug:
            data = data[:5]
        if subsample is not None:
            data = [data[i] for i in range(len(data)) if i % subsample == 0]

        for idx in range(len(data)):
            data_index = idx
            question = data[idx]['question']
            question = question[0].upper() + question[1:]
            if question[-1] != '?':
                question += '?'
            answers = data[idx]['answers']
            if is_train:
                pos_ctxs = data[idx]['positive_ctxs']
                neg_ctxs = data[idx]['negative_ctxs']
            else:
                ctxs = data[idx]['ctxs']
                pos_ctxs = [ctx for ctx in ctxs if ctx['hasanswer']]
                neg_ctxs  = [ctx for ctx in ctxs if not ctx['hasanswer']]
            assert len(pos_ctxs) > 0, "No positive context found."
            assert len(neg_ctxs) >= 2, "At least two negative contexts required."
            context = (
                f"#Document#: "
                + neg_ctxs[0]['text'] + '\n'
                + pos_ctxs[0]['text'] + '\n'
                + neg_ctxs[1]['text']
            )
            context += f"\n#Question#: {question}"
            response = f"\n#Answer#:"
            new_item = dict(
                context=context,
                response=response,
                net_response=str(answers),
                answer=answers[0],
                data_index=data_index,
            )
            list_data_dict[data_index] = new_item
    return list_data_dict


def load_jsonl(file_path):
    list_data_dict = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            list_data_dict.append(json.loads(line))
    return list_data_dict


# ---------------------------------------------------------------------------
#  Claude annotation  (mirrors evaluate_response from original)
# ---------------------------------------------------------------------------

def evaluate_response(client, model, document, gt_response, response,
                       data_type='nq_open', debug=False):
    """
    Call Claude to judge faithfulness.  Returns
    (decision, explanation, problematic_spans, approx_cost).
    Mirrors the signature and parsing of the original evaluate_response().
    """
    prompt = (
        f"{eval_prompt_before[data_type]}\n\n"
        f"#Document#: {document}\n\n"
        f"#Ground Truth {data_response_names_gt[data_type]}#: {gt_response}\n\n"
        f"#Proposed {data_response_names[data_type]}#: {response}\n\n"
        f"{eval_prompt_after[data_type]}"
    )

    for attempt in range(2):
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            text = msg.content[0].text
            input_tokens  = msg.usage.input_tokens
            output_tokens = msg.usage.output_tokens
            break
        except anthropic.APIError as e:
            if attempt == 0:
                print(f"  [API error, retrying in 5s] {e}")
                time.sleep(5)
            else:
                print(f"  [API error, skipping] {e}")
                return None, str(e), [], 0.0

    if debug:
        print('-------------------')
        print(prompt)
        print('\n' + text + '\n')
        print('-------------------', flush=True)

    # ── parse problematic spans — identical logic to original ────────────────
    problematic_spans = []
    if "Problematic Spans: " in text:
        problematic_spans = text.split('Problematic Spans: ')[1]
        if '**' in problematic_spans:
            problematic_spans = problematic_spans.split('**')[0].strip()
        try:
            problematic_spans = eval(problematic_spans)  # noqa: S307
        except Exception:
            print("Error in parsing problematic spans:", problematic_spans)
            problematic_spans = problematic_spans[1:-1].split(', ')

        if debug:
            print(problematic_spans)

    # ── parse verdict — identical logic to original ──────────────────────────
    if "Conclusion: " in text:
        dec = text.split('Conclusion: ')[1]
        if '**' in dec:
            dec = dec.split('**')[0]
        if debug:
            print(dec)
        if "True" in dec:
            decision = True
        elif "False" in dec:
            decision = False
        else:
            decision = None
    else:
        decision = None

    # Approximate cost: claude-sonnet-4-5 pricing ($3/$15 per M in/out tokens).
    # Adjust the rates below if you switch models.
    approx_cost = (input_tokens / 1_000_000 * 3) + (output_tokens / 1_000_000 * 15)

    return decision, text, problematic_spans, approx_cost


# ---------------------------------------------------------------------------
#  Main  (identical flow to original step02_eval_gpt4o.py main())
# ---------------------------------------------------------------------------

def main(hyp_path, ref_path, output_path, model, limit=None, debug=False,
         subsample=None):

    data_type = 'nq_open'
    gold_data = load_nq_open(ref_path, debug=debug, subsample=subsample)

    # Load hypotheses — .pt from step01 or plain jsonl
    if hyp_path.endswith('.pt'):
        blob = torch.load(hyp_path, weights_only=False)
        # step01 now saves {"config": ..., "examples": [...]}
        response_data = blob["examples"] if isinstance(blob, dict) else blob
        if limit is not None:
            response_data = response_data[:limit]
        responses = [item['model_completion'] for item in response_data]
    else:
        response_data = load_jsonl(hyp_path)
        if limit is not None:
            response_data = response_data[:limit]
        responses = [list(item.values())[0] for item in response_data]

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY environment variable not set.")
    client = anthropic.Anthropic(api_key=api_key)

    # ── resume from existing output file — identical logic to original ────────
    done_dict = {}
    if os.path.exists(output_path):
        print("Try to resume from existing output file.")
        with open(output_path, 'r') as fr:
            for line in fr.readlines():
                data = json.loads(line)
                done_dict[data['index']] = data

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, 'w') as fw:
        results    = []
        total_cost = 0
        corr       = 0
        total      = 0

        for idx in range(len(responses)):
            response = responses[idx]
            assert idx in gold_data, f"Index {idx} not found in gold_data"
            document    = gold_data[idx]['context']
            gt_response = gold_data[idx]['net_response']

            if idx in done_dict:
                fw.write(json.dumps(done_dict[idx]) + '\n')
                continue

            decision, explanation, problematic_spans, cost = evaluate_response(
                client, model,
                document, gt_response, response,
                data_type=data_type, debug=debug,
            )

            record = {
                'index':              idx,
                'document':           document.strip(),
                'ground_truth':       gt_response.strip(),
                'response':           response,
                'decision':           decision,
                'claude_explanation': explanation,
                'problematic_spans':  problematic_spans,
                'cost_approx':        cost,
            }
            results.append(record)
            fw.write(json.dumps(record) + '\n')
            fw.flush()

            total_cost += cost
            if decision:
                corr += 1
            total += 1

        print(f"Total cost: ${total_cost:.9f}")
        print(f"Accuracy: {corr / total:.3f}" if total > 0 else "No examples evaluated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate faithfulness of NQ-Open answers using Claude."
    )
    parser.add_argument('--hyp',   type=str, required=True,
                        help='.pt from step01 or jsonl with model completions.')
    parser.add_argument('--ref',   type=str, required=True,
                        help='NQ-Open reference jsonl (gold contexts+answers).')
    parser.add_argument('--out',   type=str, required=True,
                        help='Output jsonl path.')
    parser.add_argument('--limit', type=int, default=None,
                        help='Cap number of examples.')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--model', type=str, default='claude-sonnet-4-5',
                        help='Anthropic model string.')
    parser.add_argument('--subsample', type=int, default=None)

    args = parser.parse_args()
    # Usage: ANTHROPIC_API_KEY=... python step02_annotate_claude.py \
    #   --hyp features_nq_llama2_7b.pt --ref data/nq-open-test.jsonl \
    #   --out anno_nq_claude.jsonl
    main(args.hyp, args.ref, args.out, args.model,
         limit=args.limit, debug=args.debug, subsample=args.subsample)