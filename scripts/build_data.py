"""Builds the training mixture from public HuggingFace datasets.

Every source dataset is rewritten into one common shape, the same shape the
model sees at inference time:

    {"mode": "one" | "any" | "span", "q": question, "options": [...],
     "labels": [option indices], "state": text, "span": [start, end] | null, "src": name}

To make the model follow *instructions* rather than memorize datasets, each
example randomly varies the question wording, the label names, the option
subset and order, and sometimes wraps the state as JSON-like program state.

Datasets in HELD_OUT are never touched here. scripts/evaluate.py uses them to
measure zero-shot quality on tasks the model has never seen.

Usage: uv run python scripts/build_data.py [--scale 1.0] [--out data]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import string
from pathlib import Path

from datasets import load_dataset

HELD_OUT = ["stanfordnlp/imdb", "fancyzhx/ag_news", "mteb/banking77", "ucirvine/sms_spam",
            "deepset/prompt-injections", "zeroshot/twitter-financial-news-sentiment"]

rng = random.Random(0)
YES_NO = ["yes", "no"]
OTHER_NAMES = ["other", "none of the above", "none of these", "something else", "unknown"]


# ----------------------------------------------------------------------------- helpers

def load(name, cfg=None, split="train", n=None):
    ds = load_dataset(name, cfg, split=split)
    if n is not None and n < len(ds):
        ds = ds.shuffle(seed=0).select(range(n))
    return ds


def style(label: str) -> str:
    """Randomly restyle a label the way different programmers would write it."""
    words = re.sub(r"[_\-/&.]+", " ", label).split()
    r = rng.random()
    if r < 0.45:
        return " ".join(words).lower()
    if r < 0.65:
        return "_".join(words).lower()
    if r < 0.80:
        return " ".join(w.capitalize() for w in words)
    if r < 0.90:
        return "_".join(words).upper()
    return label


def wrap_state(text: str) -> str:
    """Sometimes present the state as structured program state instead of bare text."""
    r = rng.random()
    if r > 0.15:
        return text
    key = rng.choice(["text", "message", "input", "body", "content", "document", "user_message", "payload"])
    if r < 0.08:
        extra = rng.choice([{}, {"id": rng.randint(1000, 999999)}, {"source": rng.choice(["email", "chat", "web", "api"])},
                            {"lang": "en", "version": rng.randint(1, 5)}])
        items = list({**extra, key: text}.items())
        rng.shuffle(items)
        return json.dumps(dict(items), ensure_ascii=False, indent=rng.choice([None, 2]))
    return f"{key}: {text}"


def ex(src, mode, q, options, labels, state, span=None, wrap=True):
    return {"src": src, "mode": mode, "q": q, "options": options, "labels": labels,
            "state": wrap_state(state) if (wrap and span is None) else state, "span": span}


def choice_example(src, state, true_label, all_labels, questions, multi_q=None, ordered=False, restyle=True,
                   other_ok=True):
    """Single-label example with option subsampling, shuffling, 'other' options, and
    an occasional conversion into a multi-label ("any") query."""
    names = {l: (style(l) if restyle else l) for l in all_labels}
    if len(set(names.values())) < len(names):
        names = {l: l for l in all_labels}
    labels = list(all_labels)
    q = rng.choice(questions)

    if multi_q and rng.random() < 0.15:  # as a multi-label query; the true label may be absent
        k = rng.randint(2, min(len(labels), 12))
        subset = rng.sample([l for l in labels if l != true_label], min(k, len(labels) - 1))
        if rng.random() < 0.7:
            subset[rng.randrange(len(subset))] = true_label
        rng.shuffle(subset)
        return ex(src, "any", rng.choice(multi_q), [names[l] for l in subset],
                  [subset.index(true_label)] if true_label in subset else [], state)

    if ordered:
        return ex(src, "one", q, [names[l] for l in labels], [labels.index(true_label)], state)

    if len(labels) > 3 and rng.random() < 0.5:  # random option subset
        k = rng.randint(2, len(labels) - 1)
        others = rng.sample([l for l in labels if l != true_label], k - 1)
        labels = others + [true_label]
    opts = [names[l] for l in labels]
    target = names[true_label]
    if other_ok and len(all_labels) > 3:
        r = rng.random()
        if r < 0.08 and len(opts) > 2:  # remove the truth; the right answer becomes "other"
            opts.remove(target)
            target = rng.choice(OTHER_NAMES)
            opts.append(target)
        elif r < 0.20:  # "other" as a distractor
            opts.append(rng.choice(OTHER_NAMES))
    rng.shuffle(opts)
    return ex(src, "one", q, opts, [opts.index(target)], state)


def binary_example(src, state, positive: bool, pos_qs, neg_qs, enum_qs, neg_names, pos_names):
    """Binary task, phrased either as a yes/no predicate or as a two-way enum."""
    r = rng.random()
    if r < 0.45:
        return ex(src, "one", rng.choice(pos_qs), YES_NO, [0 if positive else 1], state)
    if r < 0.60 and neg_qs:
        return ex(src, "one", rng.choice(neg_qs), YES_NO, [1 if positive else 0], state)
    i = rng.randrange(min(len(neg_names), len(pos_names)))
    opts = [style(neg_names[i]), style(pos_names[i])]
    target = opts[1 if positive else 0]
    rng.shuffle(opts)
    return ex(src, "one", rng.choice(enum_qs), opts, [opts.index(target)], state)


# ----------------------------------------------------------------------------- task builders

NLI_BOOL_ENT = ["Is this claim supported by the text? Claim: {h}", "Does the text imply the following? {h}",
                "Based on the text, is it true that: {h}", "Is the following statement backed by the document? Statement: {h}",
                "Can we conclude this from the input: {h}", "Is this answer grounded in the source? Answer: {h}",
                "{h} True?", "Does the state satisfy this condition: {h}"]
NLI_BOOL_CON = ["Does the text contradict this statement? {h}", "Is the following claim inconsistent with the text? Claim: {h}",
                "Does this sentence conflict with the source? {h}"]
NLI_ENUM_Q = ["Based on the text, is the following statement true? {h}", "How does the text relate to this claim: {h}",
              "Verify the claim against the document. Claim: {h}", "Statement: {h}"]
NLI_ENUM_NAMES = [("entailment", "neutral", "contradiction"), ("supported", "not enough information", "contradicted"),
                  ("true", "unknown", "false"), ("yes", "maybe", "no"), ("supported", "unverifiable", "refuted"),
                  ("correct", "cannot tell", "incorrect")]


def nli(src, rows, n):
    out = []
    for r in rows:
        if r["label"] not in (0, 1, 2):
            continue
        p, h, y = r["premise"].strip(), r["hypothesis"].strip(), r["label"]
        if src == "fever_nli":  # this dataset stores the evidence in "hypothesis" and the claim in "premise"
            p, h = h, p
        t = rng.random()
        if t < 0.45:
            out.append(ex(src, "one", rng.choice(NLI_BOOL_ENT).format(h=h), YES_NO, [0 if y == 0 else 1], p))
        elif t < 0.60:
            out.append(ex(src, "one", rng.choice(NLI_BOOL_CON).format(h=h), YES_NO, [0 if y == 2 else 1], p))
        else:
            names = list(rng.choice(NLI_ENUM_NAMES))
            target = names[y]
            rng.shuffle(names)
            out.append(ex(src, "one", rng.choice(NLI_ENUM_Q).format(h=h), names, [names.index(target)], p))
        if len(out) >= n:
            break
    return out


SENT_POS_Q = ["Is the sentiment of this text positive?", "Is the author happy with what they are describing?",
              "Does this express a favorable opinion?", "Is this a positive review?"]
SENT_NEG_Q = ["Is the sentiment of this text negative?", "Is the author unhappy or dissatisfied?", "Is this a negative review?"]
SENT_ENUM_Q = ["What is the sentiment of this text?", "Classify the sentiment.", "How does the author feel?", "sentiment"]


def build_all(scale: float):
    N = lambda n: max(200, int(n * scale))
    tasks = {}

    def add(name, fn):
        try:
            rows = fn()
            rng.shuffle(rows)
            tasks[name] = rows
            print(f"  {name:28s} {len(rows):7d}", flush=True)
        except Exception as e:  # a missing dataset should not sink the whole build
            print(f"  {name:28s} FAILED: {type(e).__name__}: {str(e)[:120]}", flush=True)

    # --- natural language inference: the backbone of arbitrary "smart if" predicates
    add("mnli", lambda: nli("mnli", load("nyu-mll/multi_nli", n=N(70000) + 100), N(70000)))
    add("snli", lambda: nli("snli", load("stanfordnlp/snli", n=N(25000) + 500), N(25000)))
    for rd in (1, 2, 3):
        add(f"anli_r{rd}", lambda rd=rd: nli("anli", load("facebook/anli", split=f"train_r{rd}", n=N(20000)), N(20000)))
    add("fever_nli", lambda: nli("fever_nli", load("pietrolesci/nli_fever", n=N(35000)), N(35000)))

    # --- yes/no reading comprehension
    def boolq():
        qs = ["{q}?", "Question: {q}?", "Answer using the passage: {q}?"]
        return [ex("boolq", "one", rng.choice(qs).format(q=r["question"].strip().capitalize()), YES_NO,
                   [0 if r["answer"] else 1], r["passage"]) for r in load("google/boolq")]
    add("boolq", boolq)

    # --- sentiment
    add("sst2", lambda: [binary_example("sst2", r["sentence"].strip(), r["label"] == 1, SENT_POS_Q, SENT_NEG_Q, SENT_ENUM_Q,
                                        ["negative", "bad", "unfavorable"], ["positive", "good", "favorable"])
                         for r in load("stanfordnlp/sst2", n=N(20000))])

    def yelp():
        out = []
        star_q = ["How many stars did the reviewer give, from 1 to 5?", "Rate the customer's satisfaction from 1 (worst) to 5 (best).",
                  "Predict the star rating of this review (1 to 5).", "On a 1 to 5 scale, how positive is this review?"]
        for r in load("Yelp/yelp_review_full", n=N(45000)):
            text, s = r["text"].replace("\\n", "\n"), r["label"]
            if rng.random() < 0.55:
                out.append(ex("yelp_stars", "one", rng.choice(star_q), ["1", "2", "3", "4", "5"], [s], text))
            else:
                lab = "negative" if s < 2 else "neutral" if s == 2 else "positive"
                if rng.random() < 0.5:
                    lab = {"negative": "negative", "neutral": "mixed", "positive": "positive"}[lab]
                    out.append(choice_example("yelp_sent", text, lab, ["negative", "mixed", "positive"], SENT_ENUM_Q, other_ok=False))
                else:
                    out.append(choice_example("yelp_sent", text, lab, ["negative", "neutral", "positive"], SENT_ENUM_Q, other_ok=False))
        return out
    add("yelp", yelp)

    add("tweet_sentiment", lambda: [choice_example("tweet_sentiment", r["text"], ["negative", "neutral", "positive"][r["label"]],
                                                   ["negative", "neutral", "positive"], SENT_ENUM_Q, other_ok=False)
                                    for r in load("cardiffnlp/tweet_eval", "sentiment", n=N(15000))])

    # --- emotion (single and genuinely multi-label)
    EMO = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    add("emotion", lambda: [choice_example("emotion", r["text"], EMO[r["label"]], EMO,
                                           ["What emotion does the author express?", "Which emotion best fits this message?", "emotion"],
                                           multi_q=["Which of these emotions are expressed?"])
                            for r in load("dair-ai/emotion", "split", n=N(14000))])

    def go_emotions():
        ds = load("google-research-datasets/go_emotions", "simplified", n=N(40000))
        names = ds.features["labels"].feature.names
        out = []
        for r in ds:
            true = {names[i] for i in r["labels"]} - {"neutral"}
            k = rng.randint(3, 27)
            pool = [n for n in names if n != "neutral"]
            subset = set(rng.sample(pool, k))
            for t in true:  # keep true labels most of the time
                if rng.random() < 0.8:
                    subset.add(t)
            subset = list(subset)
            rng.shuffle(subset)
            q = rng.choice(["Which emotions are expressed in this comment?", "Select every emotion the author conveys.",
                            "Tag the message with all applicable emotions.", "emotions present"])
            out.append(ex("go_emotions", "any", q, [style(s) for s in subset], [i for i, s in enumerate(subset) if s in true], r["text"]))
        return out
    add("go_emotions", go_emotions)

    # --- topics
    def dbpedia():
        ds = load("fancyzhx/dbpedia_14", n=N(22000))
        names = [re.sub(r"(?<=[a-z])(?=[A-Z])", " ", n) for n in ds.features["label"].names]
        qs = ["What kind of entity is described?", "What is this article about?", "Classify the subject of the text.", "category"]
        return [choice_example("dbpedia", f'{r["title"]}. {r["content"].strip()}', names[r["label"]], names, qs,
                               multi_q=["Which of these categories fit the subject of the text?"]) for r in ds]
    add("dbpedia", dbpedia)

    def yahoo():
        ds = load("community-datasets/yahoo_answers_topics", n=N(28000))
        names = ds.features["topic"].names
        qs = ["What topic is this question about?", "Which category should this post be filed under?", "Pick the best forum section for this.", "topic"]
        return [choice_example("yahoo", (r["question_title"] + " " + r["question_content"]).strip() +
                               ("\n" + r["best_answer"] if rng.random() < 0.4 else ""), names[r["topic"]], names, qs,
                               multi_q=["Which of these topics are relevant to the post?"]) for r in ds]
    add("yahoo", yahoo)

    def newsgroups():
        pretty = {"alt.atheism": "atheism", "comp.graphics": "computer graphics", "comp.os.ms-windows.misc": "microsoft windows",
                  "comp.sys.ibm.pc.hardware": "pc hardware", "comp.sys.mac.hardware": "mac hardware", "comp.windows.x": "x window system",
                  "misc.forsale": "items for sale", "rec.autos": "cars", "rec.motorcycles": "motorcycles", "rec.sport.baseball": "baseball",
                  "rec.sport.hockey": "hockey", "sci.crypt": "cryptography", "sci.electronics": "electronics", "sci.med": "medicine",
                  "sci.space": "space", "soc.religion.christian": "christianity", "talk.politics.guns": "gun politics",
                  "talk.politics.mideast": "middle east politics", "talk.politics.misc": "general politics", "talk.religion.misc": "religion"}
        qs = ["Which newsgroup topic does this post belong to?", "What is this message mainly about?", "topic"]
        return [choice_example("20ng", r["text"].strip(), pretty[r["label_text"]], list(pretty.values()), qs,
                               multi_q=["Which of these subjects does the post touch on?"])
                for r in load("SetFit/20_newsgroups", n=N(10000)) if len(r["text"].strip()) > 30]
    add("20ng", newsgroups)

    # --- intents and routing (high cardinality)
    INTENT_Q = ["What is the user's intent?", "Which intent does this utterance express?", "Route this request to the right handler.",
                "What does the user want to do?", "intent"]

    def clinc():
        ds = load("clinc/clinc_oos", "plus", n=N(15000))
        names = ds.features["intent"].names
        real = [n for n in names if n != "oos"]
        out = []
        for r in ds:
            lab = names[r["intent"]]
            if lab == "oos":  # out of scope: the right answer is an "other" option
                k = rng.randint(3, 60)
                opts = [style(l) for l in rng.sample(real, k)]
                other = rng.choice(OTHER_NAMES + ["out of scope"])
                opts.append(other)
                rng.shuffle(opts)
                out.append(ex("clinc", "one", rng.choice(INTENT_Q), opts, [opts.index(other)], r["text"]))
            else:
                out.append(choice_example("clinc", r["text"], lab, real, INTENT_Q, multi_q=["Which of these intents apply?"]))
        return out
    add("clinc", clinc)

    def massive(cfg, src, qs):
        ds = load(f"mteb/amazon_massive_{cfg}", "en")
        names = sorted(set(ds["label_text"]))
        return [choice_example(src, r["text"], r["label_text"], names, qs) for r in ds]
    add("massive_intent", lambda: massive("intent", "massive_intent", INTENT_Q))
    add("massive_scenario", lambda: massive("scenario", "massive_scenario",
                                            ["Which domain does this voice command belong to?", "Which skill should handle this?", "scenario"]))

    bitext_rows = []

    def bitext():
        ds = load("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
        cats, intents = sorted(set(ds["category"])), sorted(set(ds["intent"]))
        out = []
        for r in ds:
            text = fill_placeholders(r["instruction"])[0]
            bitext_rows.append(r["instruction"])
            if rng.random() < 0.5:
                out.append(choice_example("bitext", text, r["category"], cats,
                                          ["Which department should handle this ticket?", "Route this support request to a team.", "Ticket category?"]))
            else:
                out.append(choice_example("bitext", text, r["intent"], intents, INTENT_Q + ["What is the customer asking for?"]))
        return out[:N(22000)]
    add("bitext", bitext)

    # --- safety, moderation, spam, jailbreaks
    def tweet_bin(cfg, pos_qs, neg_qs, enum_qs, neg_names, pos_names, n):
        return [binary_example(f"tweet_{cfg}", r["text"], r["label"] == 1, pos_qs, neg_qs, enum_qs, neg_names, pos_names)
                for r in load("cardiffnlp/tweet_eval", cfg, n=N(n))]
    add("tweet_irony", lambda: tweet_bin("irony", ["Is this tweet ironic or sarcastic?", "Is the author being sarcastic?"], ["Is this meant literally?"],
                                         ["Is this ironic?", "tone"], ["not ironic", "literal", "sincere"], ["ironic", "sarcastic", "ironic"], 2800))
    add("tweet_offensive", lambda: tweet_bin("offensive", ["Is this message offensive?", "Does this contain offensive language?"],
                                             ["Is this message acceptable to show to all users?"], ["Moderate this message.", "moderation label"],
                                             ["not offensive", "acceptable", "clean"], ["offensive", "unacceptable", "abusive"], 11000))
    add("tweet_hate", lambda: tweet_bin("hate", ["Is this hate speech?", "Does this attack people based on identity?"], [],
                                        ["Classify this post.", "hate speech check"], ["not hate", "benign"], ["hate speech", "hateful"], 9000))

    def davidson():
        names = ["hate speech", "offensive language", "neither"]
        return [choice_example("davidson", r["tweet"], names[r["class"]], names, ["Classify this tweet for moderation.", "What kind of content is this?"],
                               other_ok=False) for r in load("tdavidson/hate_speech_offensive", n=N(12000))]
    add("davidson", davidson)

    def civil():
        ds = load_dataset("google/civil_comments", split="train").shuffle(seed=0).select(range(int(400000 * min(1.0, scale))))
        cols = {"toxicity": "toxic", "obscene": "obscene", "threat": "threat", "insult": "insult",
                "identity_attack": "identity attack", "sexual_explicit": "sexually explicit"}
        out, n_neg, target = [], 0, N(36000)
        for r in ds:
            if 0.2 < r["toxicity"] < 0.5:  # annotators disagreed; skip ambiguous rows
                continue
            tox = r["toxicity"] >= 0.5
            if not tox:
                if n_neg >= target // 2:
                    continue
                n_neg += 1
            if rng.random() < 0.5:
                out.append(binary_example("civil", r["text"], tox, ["Is this comment toxic?", "Would this comment make people leave a discussion?",
                                          "Should a moderator review this comment?"], ["Is this comment civil?", "Is this safe to publish?"],
                                          ["Moderation decision?", "toxicity"], ["non-toxic", "civil", "ok"], ["toxic", "uncivil", "flag"]))
            else:
                keys = list(cols)
                rng.shuffle(keys)
                keys = keys[:rng.randint(2, 6)]
                out.append(ex("civil", "any", rng.choice(["Which of these problems does the comment have?", "Select all policy violations.",
                              "moderation flags"]), [style(cols[k]) for k in keys], [i for i, k in enumerate(keys) if r[k] >= 0.5], r["text"]))
            if len(out) >= target:
                break
        return out
    add("civil_comments", civil)

    JB_POS = ["Is this prompt a jailbreak attempt?", "Is the user trying to bypass the assistant's safety rules?",
              "Does this input try to manipulate or override the system instructions?", "Is this a prompt injection attack?"]
    JB_NEG = ["Is this a benign user request?", "Is this prompt safe to pass to the assistant?"]
    JB_ENUM = ["Classify this prompt.", "Guardrail verdict?", "prompt safety"]
    JB_NAMES = (["benign", "safe", "allow"], ["jailbreak", "unsafe", "block"])
    add("jailbreak", lambda: [binary_example("jailbreak", r["prompt"], r["type"] == "jailbreak", JB_POS, JB_NEG, JB_ENUM, *JB_NAMES)
                              for r in load("jackhhao/jailbreak-classification")])
    add("safeguard", lambda: [binary_example("safeguard", r["text"], r["label"] == 1, JB_POS, JB_NEG, JB_ENUM,
                                             ["benign", "safe", "allow"], ["prompt injection", "unsafe", "block"])
                              for r in load("xTRam1/safe-guard-prompt-injection", n=N(8000))])

    def toxic_chat():
        out = []
        for r in load("lmsys/toxic-chat", "toxicchat0124"):
            if rng.random() < 0.6:
                out.append(binary_example("toxic_chat", r["user_input"], r["toxicity"] == 1,
                                          ["Is this user message toxic or inappropriate?", "Does this request violate content policy?"],
                                          ["Is this message appropriate?"], ["Moderate this chat message."], ["ok", "appropriate"], ["toxic", "inappropriate"]))
            else:
                out.append(binary_example("toxic_chat", r["user_input"], r["jailbreaking"] == 1, JB_POS, JB_NEG, JB_ENUM, *JB_NAMES))
        return out
    add("toxic_chat", toxic_chat)

    add("enron_spam", lambda: [binary_example("enron_spam", (r["subject"] or "") + "\n" + (r["message"] or ""), r["label"] == 1,
                                              ["Is this email spam?", "Is this unsolicited bulk email?"], ["Is this a legitimate email?"],
                                              ["Spam filter verdict?", "email type"], ["ham", "legitimate", "not spam"], ["spam", "junk", "spam"])
                               for r in load("SetFit/enron_spam", n=N(12000))])

    # --- sentence pair tasks
    add("paws", lambda: [binary_example("paws", f'A: {r["sentence1"]}\nB: {r["sentence2"]}', r["label"] == 1,
                                        ["Do sentences A and B mean the same thing?", "Is B a paraphrase of A?"], ["Do A and B differ in meaning?"],
                                        ["Compare the two sentences."], ["different meaning", "not paraphrase"], ["same meaning", "paraphrase"])
                         for r in load("google-research-datasets/paws", "labeled_final", n=N(18000))])

    def stsb():
        out = []
        for r in load("sentence-transformers/stsb"):
            state = f'Sentence 1: {r["sentence1"]}\nSentence 2: {r["sentence2"]}'
            q = rng.choice(["How similar in meaning are the two sentences, from 0 (unrelated) to 5 (equivalent)?",
                            "Semantic similarity score between 0 and 5."])
            s = r["score"] * 5  # this mirror stores scores normalized to [0, 1]
            if rng.random() < 0.5:
                out.append(ex("stsb", "one", q, [str(i) for i in range(6)], [int(round(s))], state))
            else:
                vals = [i * 0.5 for i in range(11)]
                out.append(ex("stsb", "one", q, [f"{v:g}" for v in vals], [int(round(s * 2))], state))
        return out
    add("stsb", stsb)

    # --- multiple choice with free-form options
    def csqa():
        return [ex("csqa", "one", r["question"], r["choices"]["text"], [r["choices"]["label"].index(r["answerKey"])],
                   rng.choice(["", "Use common sense.", "(no additional context)"]), wrap=False)
                for r in load("tau/commonsense_qa") if r["answerKey"] in r["choices"]["label"]]
    add("csqa", csqa)

    def race():
        out = []
        for r in load("ehovy/race", "all", n=N(35000)):
            opts = list(r["options"])
            if len(set(opts)) < 4:
                continue
            target = opts["ABCD".index(r["answer"])]
            rng.shuffle(opts)
            out.append(ex("race", "one", r["question"], opts, [opts.index(target)], r["article"], wrap=False))
        return out
    add("race", race)

    # --- extraction
    def squad():
        out = []
        for r in load("rajpurkar/squad_v2", n=N(75000)):
            a = r["answers"]
            span = [a["answer_start"][0], a["answer_start"][0] + len(a["text"][0])] if a["text"] else None
            out.append(ex("squad_v2", "span", r["question"], [], [], r["context"], span=span or [-1, -1]))
        return out
    add("squad_v2", squad)

    def fewnerd():
        ds = load("DFKI-SLT/few-nerd", "supervised", n=N(45000))
        types = ds.features["ner_tags"].feature.names
        q_for = {"person": ["Which person is mentioned?", "Extract the person's name.", "person name"],
                 "location": ["Which location is mentioned?", "Extract the place name.", "location"],
                 "organization": ["Which organization is mentioned?", "Extract the organization name.", "company or organization"],
                 "building": ["Which building or facility is mentioned?", "Extract the name of the building."],
                 "art": ["Which work of art is mentioned?", "Extract the title of the creative work."],
                 "event": ["Which event is mentioned?", "Extract the event name."],
                 "product": ["Which product is mentioned?", "Extract the product name.", "product"]}
        out = []
        for r in ds:
            starts, text = [], ""
            for t in r["tokens"]:
                starts.append(len(text))
                text += t + " "
            text = text.rstrip()
            ents = {}  # type -> list of (char_start, char_end)
            i = 0
            tags = r["ner_tags"]
            while i < len(tags):
                if tags[i] != 0:
                    j = i
                    while j + 1 < len(tags) and tags[j + 1] == tags[i]:
                        j += 1
                    ents.setdefault(types[tags[i]], []).append((starts[i], starts[j] + len(r["tokens"][j])))
                    i = j + 1
                else:
                    i += 1
            single = [t for t, v in ents.items() if len(v) == 1 and t in q_for]
            absent = [t for t in q_for if t not in ents]
            if single and rng.random() < 0.75:
                t = rng.choice(single)
                out.append(ex("fewnerd", "span", rng.choice(q_for[t]), [], [], text, span=list(ents[t][0])))
            elif absent:
                out.append(ex("fewnerd", "span", rng.choice(q_for[rng.choice(absent)]), [], [], text, span=[-1, -1]))
        return out
    add("fewnerd", fewnerd)


    # --- numeric scales with varied meaning, so that Int/Float follow the question instead of assuming "higher = more positive"
    def scales():
        out = []
        up = ["Rate the customer's satisfaction from {lo} (worst) to {hi} (best).", "How happy is the author, from {lo} to {hi}?",
              "Score this review from {lo} to {hi}, higher is better."]
        down = ["How dissatisfied is the reviewer, from {lo} (very happy) to {hi} (very unhappy)?",
                "How angry is the customer, from {lo} (calm) to {hi} (furious)?", "How negative is this text, from {lo} to {hi}?",
                "Rate the severity of the complaint from {lo} (none) to {hi} (severe)."]
        for r in load("Yelp/yelp_review_full", split="test", n=N(30000)):
            text, s = r["text"].replace("\\n", "\n"), r["label"]  # s in 0..4
            inverted = rng.random() < 0.6
            v = 4 - s if inverted else s
            q = rng.choice(down if inverted else up)
            kind = rng.random()
            if kind < 0.5:
                opts, y = ["1", "2", "3", "4", "5"], v
            elif kind < 0.8:
                opts, y = [str(i) for i in range(11)], round(v * 2.5)
            else:
                opts, y = [f"{i / 10:g}" for i in range(11)], round(v * 2.5)
            out.append(ex("scales_yelp", "one", q.format(lo=opts[0], hi=opts[-1]), opts, [y], text))
        return out
    add("scales_yelp", scales)

    def graded_toxicity():
        ds = load_dataset("google/civil_comments", split="validation").shuffle(seed=0)
        cols = {"toxicity": ["How toxic is this comment, from {lo} to {hi}?", "Toxicity score between {lo} and {hi}.",
                             "What share of readers would find this comment toxic? Scale {lo} to {hi}."],
                "insult": ["How insulting is this comment, from {lo} to {hi}?"],
                "identity_attack": ["How strongly does this comment attack someone's identity, from {lo} to {hi}?"],
                "obscene": ["How obscene is this comment, from {lo} to {hi}?"], "threat": ["How threatening is this comment, from {lo} to {hi}?"]}
        out, zeros, target = [], 0, N(30000)
        for r in ds:
            col = "toxicity" if rng.random() < 0.6 else rng.choice(list(cols))
            v = r[col]
            if v < 0.05:
                if zeros > target * 0.3:
                    continue
                zeros += 1
            opts = [f"{i / 10:g}" for i in range(11)] if rng.random() < 0.6 else [str(i) for i in range(11)]
            out.append(ex("scales_toxicity", "one", rng.choice(cols[col]).format(lo=opts[0], hi=opts[-1]), opts, [round(v * 10)], r["text"]))
            if len(out) >= target:
                break
        return out
    add("scales_toxicity", graded_toxicity)

    # --- multi-label queries where several options are true at once. Without these, the model learns that
    #     "any" queries have at most one positive, because most multi-label data is recast single-label data.
    def entity_types():
        ds = load("DFKI-SLT/few-nerd", "supervised", split="validation")
        types = ds.features["ner_tags"].feature.names
        pretty = {"person": ["person", "people"], "location": ["location", "place"], "organization": ["organization", "company or organization"],
                  "building": ["building", "building or facility"], "art": ["work of art", "creative work"], "event": ["event"],
                  "product": ["product"], "other": ["other named thing"]}
        out = []
        for r in ds:
            present = {types[t] for t in r["ner_tags"] if t != 0} - {"other"}
            k = rng.randint(2, 7)
            cand = rng.sample([t for t in pretty if t != "other"], k)
            names = [style(rng.choice(pretty[t])) for t in cand]
            q = rng.choice(["Which kinds of entities are mentioned in the text?", "Select every entity type that appears.", "entity types present"])
            out.append(ex("multi_entities", "any", q, names, [i for i, t in enumerate(cand) if t in present], " ".join(r["tokens"])))
        return out[:N(18000)]
    add("multi_entities", entity_types)

    def supported_claims():
        groups = {}
        for r in load("stanfordnlp/snli", split="validation").to_list() + load("stanfordnlp/snli", split="test").to_list() \
                + load("stanfordnlp/snli", n=N(120000)).to_list()[-N(90000):]:
            if r["label"] in (0, 1, 2):
                groups.setdefault(r["premise"], []).append((r["hypothesis"], r["label"]))
        out = []
        qs = ["Which of these statements are supported by the text?", "Select every claim that follows from the input.",
              "Which of the following are true according to the text?"]
        premises = [p for p, h in groups.items() if len(h) >= 3]
        for p in premises:
            hyps = groups[p][:]
            state = p
            if rng.random() < 0.5:  # merge two scenes, which yields more simultaneous positives
                p2 = rng.choice(premises)
                if p2 != p:
                    state = f"{p} {p2}" if rng.random() < 0.5 else f"{p2} {p}"
                    hyps = hyps + groups[p2]
            rng.shuffle(hyps)
            hyps = hyps[:rng.randint(2, 8)]
            out.append(ex("multi_claims", "any", rng.choice(qs), [h for h, _ in hyps], [i for i, (_, y) in enumerate(hyps) if y == 0], state))
        rng.shuffle(out)
        return out[:N(30000)]
    add("multi_claims", supported_claims)

    def mixed_messages():
        pools = []
        ds = load("community-datasets/yahoo_answers_topics", split="test", n=N(16000))
        names = ds.features["topic"].names
        pools.append(("multi_mix_yahoo", [((r["question_title"] + " " + r["question_content"]).strip(), names[r["topic"]]) for r in ds], names,
                      ["Which of these topics come up in the text?", "Select all topics covered."]))
        ds = load("clinc/clinc_oos", "plus", split="validation")
        inames = ds.features["intent"].names
        pools.append(("multi_mix_clinc", [(r["text"], inames[r["intent"]]) for r in ds if inames[r["intent"]] != "oos"], [n for n in inames if n != "oos"],
                      ["Which intents does the user express?", "The user may ask for several things. Select all that apply."]))
        ds = load("bitext/Bitext-customer-support-llm-chatbot-training-dataset", n=N(12000))
        cats = sorted(set(ds["category"]))
        pools.append(("multi_mix_bitext", [(fill_placeholders(r["instruction"])[0], r["category"]) for r in ds], cats,
                      ["Which departments does this message concern?", "Select every issue category mentioned."]))
        ds = load("dair-ai/emotion", "split", split="validation").to_list() + load("dair-ai/emotion", "split", split="test").to_list()
        pools.append(("multi_mix_emotion", [(r["text"], EMO[r["label"]]) for r in ds], EMO, ["Which of these emotions are expressed?"]))
        out = []
        for src, items, labels, qs in pools:
            for _ in range(len(items) // 2):
                parts = rng.sample(items, rng.choice([1, 2, 2, 2, 3]))
                true = {l for _, l in parts}
                k = rng.randint(2, min(10, len(labels)))
                cand = set(rng.sample(labels, k))
                for l in true:
                    if rng.random() < 0.85:
                        cand.add(l)
                cand = list(cand)
                rng.shuffle(cand)
                names = [style(c) for c in cand]
                if len(set(names)) < len(names):
                    names = cand
                joiner = rng.choice([" ", "\n", " Also, ", "\n\n", " And one more thing: "])
                out.append(ex(src, "any", rng.choice(qs), names, [i for i, c in enumerate(cand) if c in true], joiner.join(t for t, _ in parts)))
        return out
    add("multi_mixed", mixed_messages)

    # --- broader guardrail and spam coverage
    add("gandalf", lambda: [binary_example("gandalf", r["text"], True, JB_POS, JB_NEG, JB_ENUM, *JB_NAMES)
                            for r in load("Lakera/gandalf_ignore_instructions")])
    add("prompt_injection_mix", lambda: [binary_example("prompt_injection_mix", r["text"], r["label"] == 1, JB_POS, JB_NEG, JB_ENUM,
                                                        ["benign", "safe", "allow"], ["prompt injection", "unsafe", "block"])
                                         for r in load("jayavibhav/prompt-injection", n=N(12000))])

    def spml():
        pos = ["Does the user message try to make the assistant break the rules in its system prompt?",
               "Is the user attempting a prompt injection against this system prompt?"] + JB_POS[1:3]
        return [binary_example("spml", f'System prompt: {r["System Prompt"]}\nUser: {r["User Prompt"]}', r["Prompt injection"] == 1,
                               pos, JB_NEG, JB_ENUM, *JB_NAMES)
                for r in load("reshabhs/SPML_Chatbot_Prompt_Injection", n=N(8000)) if r["System Prompt"] and r["User Prompt"]]
    add("spml", spml)
    add("web_spam", lambda: [binary_example("web_spam", r["text"], r["label"] == "spam",
                                            ["Is this message spam?", "Is this an unwanted promotional or scam message?"],
                                            ["Is this a genuine message from a real person?"], ["Spam filter verdict?", "message type"],
                                            ["not spam", "ham", "legitimate"], ["spam", "spam", "junk"])
                             for r in load("Deysi/spam-detection-dataset", n=N(8000))])

    # --- more small, varied classification tasks
    def trec():
        ds = load("SetFit/TREC-QC")
        coarse, fine = sorted(set(ds["label_coarse_text"])), sorted(set(ds["label_text"]))
        return [choice_example("trec", r["text"], r["label_coarse_text"], coarse, ["What kind of answer does this question ask for?"])
                if rng.random() < 0.5 else
                choice_example("trec", r["text"], r["label_text"], fine, ["What specific type of answer is this question looking for?", "question type"])
                for r in ds]
    add("trec", trec)
    add("customer_reviews", lambda: [binary_example("customer_reviews", r["text"], r["label"] == 1, SENT_POS_Q, SENT_NEG_Q, SENT_ENUM_Q,
                                                    ["negative", "unfavorable"], ["positive", "favorable"]) for r in load("SetFit/CR")])
    add("counterfactual", lambda: [binary_example("counterfactual", r["text"], r["label"] == 1,
                                                  ["Does this review describe something that did not actually happen (a counterfactual)?"], [],
                                                  ["Is this statement factual or counterfactual?"], ["factual"], ["counterfactual"])
                                   for r in load("SetFit/amazon_counterfactual_en")])
    add("tweet_sentiment2", lambda: [choice_example("tweet_sentiment2", r["text"].strip(), r["label_text"], ["negative", "neutral", "positive"],
                                                    SENT_ENUM_Q, other_ok=False)
                                     for r in load("mteb/tweet_sentiment_extraction", n=N(12000)) if r["text"].strip()])
    add("tasksource", lambda: tasksource(N(1500)))

    add("synthetic_fields", lambda: synthetic_fields(bitext_rows, N(30000)))
    return tasks



# ----------------------------------------------------------------------------- 150 extra tasks, calibration tasks

# Never train on these tasksource tasks: they overlap the held-out evaluation sets...
TS_HELD_OUT = {"imdb", "counterfactually-augmented-imdb", "ag_news", "banking77", "sms_spam", "financial_phrasebank/sentences_allagree",
               "amazon_polarity/amazon_polarity"}
# ...these are reserved for calibration (the model must not see them, so that they behave like a user's brand new task)...
TS_CALIBRATION = {
    "crowdflower/airline-sentiment": "What is the sentiment of this tweet about an airline?",
    "emo/emo2019": "What emotion does the last speaker in this chat express?",
    "scicite": "What is the purpose of this citation in the paper?",
    "rotten_tomatoes": "Is this movie review positive or negative?",
    "health_fact": "How accurate is this health claim?",
    "insincere-questions": "Is this question sincere or insincere?",
    "lex_glue/ledgar": "What type of contract clause is this?",
    "logical-fallacy": "Which logical fallacy does this argument commit?",
    "silicone/dyda_e": "What emotion is expressed in this utterance?",
    "subjectivity": "Is this sentence subjective or objective?",
}
# ...and these are already in the mixture from their original sources, or have labels that carry no meaning.
TS_SKIP = {"glue/sst2", "paws/labeled_final", "paws/labeled_swap", "tweet_eval/offensive", "tweet_eval/sentiment", "tweet_eval/hate",
           "tweet_eval/irony", "tweet_eval/emoji", "yelp_review_full/yelp_review_full", "dbpedia_14/dbpedia_14", "hate_speech_offensive",
           "yahoo_answers_topics", "super_glue/boolq", "v1/gen_train234_test2to10", "blog_authorship_corpus/horoscope", "trec",
           "lexical_relation_classification/K&H+N", "lexical_relation_classification/BLESS", "lexical_relation_classification/ROOT09",
           "lexical_relation_classification/EVALution", "lexical_relation_classification/CogALexV", "discovery/discovery"}

_ts_cache = {}


def _tasksource_rows():
    if not _ts_cache:
        ds = load_dataset("tasksource/zero-shot-label-nli", split="train").shuffle(seed=0)
        vocab, rows = {}, {}
        for r in ds:
            m = re.match(r"This example is (.*)\.$", r["hypothesis"], flags=re.S)
            if not m or r["labels"] == 1:
                continue
            vocab.setdefault(r["task"], set()).add(m.group(1))
            rows.setdefault(r["task"], []).append((r["premise"], m.group(1), r["labels"] == 0))
        _ts_cache.update(vocab=vocab, rows=rows)
    return _ts_cache["vocab"], _ts_cache["rows"]


def tasksource(per_task):
    vocab, rows = _tasksource_rows()
    out = []
    for task, items in rows.items():
        if task in TS_HELD_OUT or task in TS_CALIBRATION or task in TS_SKIP or len(vocab[task]) < 2:
            continue
        labels, short, n = sorted(vocab[task]), task.split("/")[-1].replace("_", " ").replace("-", " "), 0
        for text, label, is_true in items:
            if n >= per_task:
                break
            if is_true and rng.random() < 0.75:
                q = [f"Task: {short}. Which label applies?", f"Classify this example ({short}).", f"{short}: pick the correct label."]
                out.append(choice_example(f"ts/{task}", text, label, labels, q, other_ok=len(labels) > 5))
            elif is_true or rng.random() < 0.5:
                q = rng.choice([f'Task: {short}. Does the label "{label}" apply?', f"Is this example {label}? ({short})"])
                out.append(ex(f"ts/{task}", "one", q, YES_NO, [0 if is_true else 1], text))
            else:
                continue
            n += 1
    return out


def calibration_rows(per_task=400):
    """Tasks the model never trains on, phrased the way a user would phrase a new task. Used only to fit temperatures."""
    vocab, rows = _tasksource_rows()
    out = []
    for task, q in TS_CALIBRATION.items():
        labels = sorted(vocab[task])
        pretty = [re.sub(r"[_\-]+", " ", l).strip().lower() for l in labels]
        if len(set(pretty)) < len(pretty):
            pretty = labels
        for text, label, is_true in [it for it in rows[task] if it[2]][:per_task]:
            out.append({"src": f"calib/{task}", "mode": "one", "q": q, "options": pretty, "labels": [labels.index(label)],
                        "state": text, "span": None})
    for name, col, q in (("benayas/snips", "category", "What does the user want to do?"),
                         ("tuetschek/atis", "intent", "What is the traveler asking about?")):
        ds = load(name, n=per_task)
        labels = sorted(set(ds[col]))
        pretty = [re.sub(r"(?<=[a-z])(?=[A-Z])|[_+]+", " ", l).strip().lower() for l in labels]
        for r in ds:
            out.append({"src": f"calib/{name}", "mode": "one", "q": q, "options": pretty, "labels": [labels.index(r[col])],
                        "state": r["text"], "span": None})
    return out


def _norm(t: str) -> str:
    return re.sub(r"\W+", "", t.lower())[:300]


def held_out_texts() -> set[str]:
    """Normalized text of every held-out evaluation example, for decontaminating the training set."""
    seen = set()
    for name, cfg, splits, col in (("stanfordnlp/imdb", None, ["test"], "text"), ("fancyzhx/ag_news", None, ["test"], "text"),
                                   ("mteb/banking77", None, ["test"], "text"), ("ucirvine/sms_spam", None, ["train"], "sms"),
                                   ("deepset/prompt-injections", None, ["train", "test"], "text"),
                                   ("zeroshot/twitter-financial-news-sentiment", None, ["validation"], "text")):
        for sp in splits:
            seen.update(_norm(t) for t in load_dataset(name, cfg, split=sp)[col])
    seen.discard("")
    return seen

# ----------------------------------------------------------------------------- synthetic business fields

FIRST = "James Maria Wei Aisha Carlos Priya Olga Kenji Fatima Liam Noah Emma Sofia Lucas Amara Yuki Ivan Chloe Omar Hannah Diego Mei Tariq Elena".split()
LAST = "Smith Garcia Chen Khan Rossi Patel Ivanov Tanaka Hassan Murphy Johnson Silva Kim Nguyen Okafor Sato Petrov Martin Ali Schmidt Lopez Wang Haddad Popescu".split()
CITIES = "London Berlin Austin Toronto Mumbai Osaka Lagos Madrid Seattle Dublin Denver Lyon Porto Seoul Cairo Lima Oslo Perth".split()
COUNTRIES = "Canada Germany Japan Brazil India France Spain Kenya Norway Mexico Italy Australia Ireland Portugal".split()
DOMAINS = "gmail.com outlook.com yahoo.com proton.me example.org acme.io company.co fastmail.com".split()


def _digits(n): return "".join(rng.choice(string.digits) for _ in range(n))
def _alnum(n): return "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(n))


def gen_order(): return rng.choice([f"#{_digits(rng.randint(5, 9))}", f"ORD-{_digits(6)}", f"{_alnum(3)}-{_digits(5)}", _digits(rng.randint(6, 10)), f"SO{_digits(7)}"])
def gen_invoice(): return rng.choice([f"INV-{_digits(5)}", f"INV{_digits(7)}", f"#{_digits(6)}", f"{rng.randint(2019, 2026)}-{_digits(4)}"])
def gen_name(): return f"{rng.choice(FIRST)} {rng.choice(LAST)}"
def gen_amount(): return rng.choice(["${:,.2f}", "{:.2f} EUR", "€{:.0f}", "${:.0f}", "{:,.2f} USD", "£{:.2f}"]).format(rng.uniform(3, 4000))
def gen_email():
    f, l = rng.choice(FIRST).lower(), rng.choice(LAST).lower()
    return rng.choice([f"{f}.{l}", f"{f}{rng.randint(1, 99)}", f"{f[0]}{l}", f"{l}_{f}"]) + "@" + rng.choice(DOMAINS)
def gen_phone(): return rng.choice([f"+1 ({_digits(3)}) {_digits(3)}-{_digits(4)}", f"{_digits(3)}-{_digits(3)}-{_digits(4)}", f"+44 {_digits(4)} {_digits(6)}"])
def gen_date():
    m = rng.choice("January February March April May June July August September October November December".split())
    d, y = rng.randint(1, 28), rng.randint(2022, 2026)
    return rng.choice([f"{m} {d}, {y}", f"{d} {m} {y}", f"{y}-{rng.randint(1, 12):02d}-{d:02d}", f"{rng.randint(1, 12)}/{d}/{y}", f"{m[:3]} {d}"])
def gen_tracking(): return rng.choice([f"1Z{_alnum(16)}", f"{_digits(12)}", f"{_alnum(2)}{_digits(9)}{_alnum(2)}"])


PLACEHOLDERS = {"Order Number": gen_order, "Invoice Number": gen_invoice, "Person Name": gen_name, "Refund Amount": gen_amount,
                "Delivery City": lambda: rng.choice(CITIES), "Delivery Country": lambda: rng.choice(COUNTRIES),
                "Account Type": lambda: rng.choice(["Premium", "Free", "Pro", "Enterprise", "Standard", "Gold", "Business"]),
                "Account Category": lambda: rng.choice(["Premium", "Free", "Pro", "Enterprise", "Standard", "Platinum"]),
                "Currency Symbol": lambda: rng.choice(["$", "€", "£"])}

# field -> (generator, questions, carrier sentences)
EXTRA_FIELDS = {
    "email": (gen_email, ["What is the customer's email address?", "Extract the email address.", "email", "contact email"],
              ["You can reach me at {v}.", "My email is {v}", "please reply to {v} thanks", "Contact: {v}"]),
    "phone": (gen_phone, ["What is the phone number?", "Extract the callback number.", "phone"],
              ["Call me on {v}.", "my number is {v}", "Phone: {v}"]),
    "date": (gen_date, ["What date is mentioned?", "When did this happen?", "Extract the date."],
             ["This happened on {v}.", "I placed it on {v}", "It was due {v} and nothing came.", "Date: {v}"]),
    "tracking": (gen_tracking, ["What is the tracking number?", "Extract the shipment tracking code.", "tracking_number"],
                 ["The tracking number is {v}.", "tracking {v} shows no updates", "Tracking: {v}"]),
    "name": (gen_name, ["What is the customer's name?", "Who wrote this message?", "customer_name"],
             ["Thanks, {v}", "My name is {v}.", "Regards,\n{v}", "This is {v} writing."]),
    "amount": (gen_amount, ["How much money is mentioned?", "What amount was charged?", "Extract the amount."],
               ["I was charged {v}.", "The total came to {v}", "you took {v} from my card"]),
    "order": (gen_order, ["What is the order number?", "Extract the order ID.", "order_id"],
              ["My order number is {v}.", "order {v}", "Ref: {v}", "This is about order {v}."]),
}
PLACEHOLDER_Q = {"Order Number": ["What is the order number?", "Extract the order ID.", "order_id", "Which order is this about?"],
                 "Invoice Number": ["What is the invoice number?", "invoice_id"], "Person Name": ["Which person is named?", "person name"],
                 "Refund Amount": ["What refund amount is mentioned?", "How much money does the customer want back?"],
                 "Delivery City": ["Which city is the delivery going to?", "delivery city"],
                 "Delivery Country": ["Which country is the delivery going to?", "destination country"],
                 "Account Type": ["What type of account is mentioned?", "account tier"], "Account Category": ["What account category is mentioned?"]}


def fill_placeholders(template: str):
    """Fills {{Placeholders}} with synthetic values; returns (text, {placeholder: (start, end)})."""
    out, spans, pos = "", {}, 0
    for m in re.finditer(r"\{\{(.*?)\}\}", template):
        out += template[pos:m.start()]
        gen = PLACEHOLDERS.get(m.group(1))
        val = gen() if gen else "it"
        if gen and m.group(1) not in spans:
            spans[m.group(1)] = (len(out), len(out) + len(val))
        elif m.group(1) in spans:
            spans[m.group(1)] = None  # appears twice: ambiguous, do not use as a target
        out += val
        pos = m.end()
    return out + template[pos:], spans


def synthetic_fields(templates, n):
    out = []
    if not templates:
        return out
    while len(out) < n:
        text, spans = fill_placeholders(rng.choice(templates))
        text = text[0].upper() + text[1:] if rng.random() < 0.5 else text
        present = {}
        for name in rng.sample(list(EXTRA_FIELDS), rng.randint(0, 3)):  # add carrier sentences around the text
            if name == "order" and "Order Number" in spans or name == "name" and "Person Name" in spans \
                    or name == "amount" and "Refund Amount" in spans:
                continue
            gen, _, carriers = EXTRA_FIELDS[name]
            val = gen()
            sent = rng.choice(carriers).format(v=val)
            if rng.random() < 0.7:
                off = len(text) + 1 + sent.index(val)
                text = text + " " + sent
            else:
                shift = len(sent) + 1
                spans = {k: (v[0] + shift, v[1] + shift) if v else None for k, v in spans.items()}
                present = {k: (v[0] + shift, v[1] + shift) for k, v in present.items()}
                off = sent.index(val)
                text = sent + " " + text
            present[name] = (off, off + len(val))
        targets = [(PLACEHOLDER_Q[k], v) for k, v in spans.items() if v and k in PLACEHOLDER_Q]
        targets += [(EXTRA_FIELDS[k][1], v) for k, v in present.items()]
        has_order = "Order Number" in spans or "order" in present
        has_name = "Person Name" in spans or "name" in present
        has_amount = "Refund Amount" in spans or "amount" in present
        absent = [EXTRA_FIELDS[k][1] for k in EXTRA_FIELDS if k not in present
                  and not (k == "order" and has_order) and not (k == "name" and has_name) and not (k == "amount" and has_amount)]
        if targets and rng.random() < 0.7:
            qs, (s, e) = rng.choice(targets)
            out.append(ex("synthetic_fields", "span", rng.choice(qs), [], [], text, span=[s, e]))
        elif absent:
            out.append(ex("synthetic_fields", "span", rng.choice(rng.choice(absent)), [], [], text, span=[-1, -1]))
    return out


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0, help="multiplies every per-dataset sample budget")
    ap.add_argument("--out", default="data")
    ap.add_argument("--val-per-task", type=int, default=250)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    print("building tasks:")
    tasks = build_all(args.scale)
    banned = held_out_texts()
    removed = 0
    for name in list(tasks):
        kept = [r for r in tasks[name] if _norm(r["state"]) not in banned]
        removed += len(tasks[name]) - len(kept)
        tasks[name] = kept
    print(f"decontamination: removed {removed} training examples whose text also appears in a held-out evaluation set")

    train, val = [], []
    for name, rows in tasks.items():
        k = min(args.val_per_task, len(rows) // 10)
        val += rows[:k]
        train += rows[k:]
    rng.shuffle(train)
    calib = [r for r in calibration_rows() if _norm(r["state"]) not in banned]
    for split, rows in (("train", train), ("val", val), ("calib", calib)):
        with open(out / f"{split}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    modes = {}
    for r in train:
        modes[r["mode"]] = modes.get(r["mode"], 0) + 1
    print(f"train={len(train)} val={len(val)} calib={len(calib)} modes={modes}")


if __name__ == "__main__":
    main()
