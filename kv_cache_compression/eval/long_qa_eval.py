"""
Long-Form Question Answering Evaluation for KV-Cache Compression.

Evaluates how well a model can retrieve and synthesise information from a
long multi-document context when the KV-cache is compressed.

Key design choices
------------------
- Multi-document context: several "documents" are concatenated to form a long
  prompt; the answer is drawn from one specific document.
- Distractor documents: the remaining documents contain plausible but wrong
  information — this stresses the model's ability to locate the correct passage
  even after compression.
- Rich metrics: token-F1, ROUGE-1/2/L, exact-match, answer-present recall,
  and NLL-based perplexity on the continuation.

Built-in sample datasets
------------------------
We ship a compact synthetic dataset of 20+ (context, question, answer) triples
so the evaluator works *without* internet access or HuggingFace dataset downloads.
Each sample is designed with a long, multi-sentence context so that meaningful
KV-cache compression occurs during encoding.

Usage
-----
    from kv_cache_compression.eval.long_qa_eval import LongQAEvaluator, LongQASample

    evaluator = LongQAEvaluator(model, tokenizer)
    results   = evaluator.run_all(policies, n_samples=10)
    evaluator.print_summary(results)
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field, asdict
from typing import Sequence

import torch

from kv_cache_compression.cache.kv_cache import KVCacheInspector, to_legacy_tuple
from kv_cache_compression.cache.policies import CompressionPolicy, PolicyContext
from kv_cache_compression.cache.prune import aggregate_attention_scores
from kv_cache_compression.eval.benchmark import BenchmarkRunner, _key_norm_scores
from kv_cache_compression.eval.metrics_eval import (
    token_f1, rouge_scores, exact_match_normalized, needle_recall, batch_metrics
)
from kv_cache_compression.eval.perplexity_eval import perplexity_from_nll
from kv_cache_compression.utils.profiling import timed


# ══════════════════════════════════════════════════════════════════════════════
# Data structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class LongQASample:
    """A single long-QA evaluation sample."""
    prompt: str
    continuation: str
    answer: str
    question: str = ""
    doc_id: str = ""


@dataclass
class LongQAResult:
    """Result for one (policy, sample) evaluation."""
    policy_name: str
    doc_id: str
    question: str
    answer: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    continuation_nll: float | None
    perplexity: float | None
    exact_match: float
    token_f1: float
    rouge1_f1: float
    rouge2_f1: float
    rougeL_f1: float
    answer_present: float   # 1.0 if answer string appears in model output (oracle)
    prompt_seconds: float
    compression_seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


# ══════════════════════════════════════════════════════════════════════════════
# Built-in synthetic dataset (no internet required)
# ══════════════════════════════════════════════════════════════════════════════

# Each entry: (list_of_documents, question, answer, doc_id)
# The answer is always contained in exactly one of the documents.
_BUILTIN_QA_DATA: list[tuple[list[str], str, str, str]] = [
    (
        [
            "The Eiffel Tower was built between 1887 and 1889 as the entrance arch for the 1889 World's Fair. "
            "It was designed by engineer Gustave Eiffel. The tower stands 330 metres tall, including the broadcast antenna. "
            "It is located on the Champ de Mars in Paris, France, and is one of the most recognisable structures in the world. "
            "Approximately 7 million people visit the Eiffel Tower every year, making it the most-visited paid monument in the world.",

            "The Great Wall of China is a series of fortifications built across the historical northern borders of ancient Chinese states. "
            "Construction began as early as the 7th century BC. The wall stretches over 21,196 kilometres from east to west. "
            "It was built primarily to protect Chinese territories from nomadic invasions from the north.",

            "The Colosseum in Rome is an oval amphitheatre built of travertine limestone, tuff, and brick-faced concrete. "
            "Construction started around AD 70 under Emperor Vespasian and was completed around AD 80 under Titus. "
            "It could hold between 50,000 and 80,000 spectators. It was used for gladiatorial contests and public spectacles.",
        ],
        "In which year was the Eiffel Tower completed?",
        "1889",
        "qa_eiffel",
    ),
    (
        [
            "Python is a high-level, general-purpose programming language emphasising code readability. "
            "It was created by Guido van Rossum and first released in 1991. Python supports multiple programming paradigms, "
            "including structured, object-oriented, and functional programming. "
            "It is widely used in web development, data science, artificial intelligence, and scientific computing.",

            "Java is a class-based, object-oriented programming language developed by Sun Microsystems in 1995. "
            "It follows the principle of 'write once, run anywhere', meaning compiled Java code can run on all platforms "
            "that support the Java Virtual Machine without recompilation.",

            "C++ is a general-purpose programming language created as an extension of C. "
            "It was developed by Bjarne Stroustrup at Bell Labs starting in 1979. "
            "C++ provides object-oriented, generic, and functional features in addition to low-level memory manipulation.",
        ],
        "Who created the Python programming language?",
        "Guido van Rossum",
        "qa_python",
    ),
    (
        [
            "Photosynthesis is a process used by plants, algae, and certain bacteria to convert light energy "
            "into chemical energy stored as glucose. The process primarily takes place in the chloroplasts, "
            "specifically using the green pigment chlorophyll. The overall chemical equation is: "
            "6CO2 + 6H2O + light energy → C6H12O6 + 6O2. There are two main stages: "
            "the light-dependent reactions and the Calvin cycle.",

            "Cellular respiration is the process by which organisms break down glucose to release energy in the form of ATP. "
            "It occurs in three main stages: glycolysis in the cytoplasm, the Krebs cycle in the mitochondrial matrix, "
            "and oxidative phosphorylation on the inner mitochondrial membrane.",

            "Mitosis is a type of cell division resulting in two daughter cells with the same number of chromosomes as the parent cell. "
            "It consists of four phases: prophase, metaphase, anaphase, and telophase, followed by cytokinesis.",
        ],
        "What chemical equation represents photosynthesis?",
        "6CO2 + 6H2O + light energy",
        "qa_photosynthesis",
    ),
    (
        [
            "The Amazon River is the largest river in the world by discharge volume and the second longest by length. "
            "It flows through nine countries in South America, with the majority of its basin in Brazil. "
            "The river has a total length of approximately 6,400 kilometres. "
            "The Amazon basin is home to the world's largest tropical rainforest, covering about 5.5 million square kilometres.",

            "The Nile River is traditionally considered the longest river in the world, stretching approximately 6,650 kilometres. "
            "It flows northward through northeastern Africa and drains into the Mediterranean Sea. "
            "The Nile has historically been vital to the civilisations of ancient Egypt.",

            "The Mississippi River is the second-longest river in North America, stretching about 3,730 kilometres. "
            "It drains all or parts of 31 U.S. states and 2 Canadian provinces.",
        ],
        "Through how many countries does the Amazon River flow?",
        "nine",
        "qa_amazon",
    ),
    (
        [
            "Albert Einstein was born on 14 March 1879 in Ulm, in the Kingdom of Württemberg in the German Empire. "
            "He developed the theory of relativity, one of the two pillars of modern physics alongside quantum mechanics. "
            "His mass-energy equivalence formula E = mc² has been dubbed 'the world's most famous equation'. "
            "He received the Nobel Prize in Physics in 1921 for his discovery of the law of the photoelectric effect.",

            "Isaac Newton was an English mathematician, physicist, astronomer, and author who is widely recognised "
            "as one of the most influential scientists of all time. He formulated the laws of motion and universal gravitation. "
            "He was born on 25 December 1642 in Woolsthorpe, Lincolnshire, England.",

            "Marie Curie was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity. "
            "She was the first woman to win a Nobel Prize, the first to win it twice, and the only person to win it in two different sciences.",
        ],
        "For what discovery did Einstein receive the Nobel Prize in Physics?",
        "photoelectric effect",
        "qa_einstein",
    ),
    (
        [
            "The human brain contains approximately 86 billion neurons, with each neuron forming thousands of synaptic connections. "
            "The brain is divided into several regions including the cerebrum, cerebellum, and brainstem. "
            "The cerebrum is the largest part and is responsible for higher cognitive functions such as reasoning, language, and voluntary movement. "
            "The prefrontal cortex plays a key role in decision-making, personality expression, and social behaviour.",

            "The human heart is a muscular organ that pumps blood through the circulatory system. "
            "It beats approximately 60 to 100 times per minute in a healthy adult at rest. "
            "The heart is divided into four chambers: left atrium, left ventricle, right atrium, and right ventricle.",

            "The liver is the largest internal organ in the human body, weighing about 1.5 kilograms in adults. "
            "It performs over 500 different functions, including detoxification of metabolites and synthesis of proteins.",
        ],
        "How many neurons does the human brain contain?",
        "86 billion",
        "qa_brain",
    ),
    (
        [
            "The speed of light in a vacuum is approximately 299,792,458 metres per second, often approximated as 3 × 10^8 m/s. "
            "This constant, denoted by c, is a fundamental constant in physics and plays a central role in Einstein's special relativity. "
            "Nothing with mass can reach or exceed the speed of light. "
            "Light takes approximately 8 minutes and 20 seconds to travel from the Sun to the Earth.",

            "The speed of sound in air at room temperature (20°C) is approximately 343 metres per second. "
            "Sound travels faster in liquids and solids than in gases. In water, sound travels at about 1,480 m/s.",

            "Gravitational waves travel at the speed of light. They were first directly detected on 14 September 2015 "
            "by the LIGO and Virgo interferometers, confirming a major prediction of Einstein's general relativity.",
        ],
        "What is the approximate speed of light in a vacuum?",
        "299,792,458 metres per second",
        "qa_speed_of_light",
    ),
    (
        [
            "DNA, or deoxyribonucleic acid, is a molecule composed of two polynucleotide chains that coil around each other "
            "to form a double helix. It carries genetic information used in the growth, development, functioning, and reproduction of organisms. "
            "The structure of DNA was first described by James Watson and Francis Crick in 1953, based on X-ray crystallography work by Rosalind Franklin. "
            "The four bases of DNA are adenine (A), thymine (T), guanine (G), and cytosine (C).",

            "RNA, or ribonucleic acid, is a polymeric molecule essential in various biological roles in coding, decoding, regulation, "
            "and expression of genes. Unlike DNA, RNA is single-stranded and uses uracil instead of thymine.",

            "Proteins are large biomolecules consisting of one or more chains of amino acid residues. "
            "They are synthesised by ribosomes in a process called translation.",
        ],
        "Who first described the structure of DNA?",
        "James Watson and Francis Crick",
        "qa_dna",
    ),
    (
        [
            "The solar system consists of the Sun and everything gravitationally bound to it, including eight planets. "
            "In order from the Sun, they are: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune. "
            "Jupiter is the largest planet, with a mass more than twice that of all other planets combined. "
            "Saturn is known for its prominent ring system composed primarily of ice and rock.",

            "A black hole is a region of spacetime where gravity is so strong that nothing, not even light, can escape. "
            "The boundary of the region from which escape is not possible is called the event horizon. "
            "Stellar black holes form when massive stars collapse at the end of their life cycles.",

            "Comets are small icy bodies that release gas and dust when near the Sun, forming a visible coma and tail. "
            "Halley's Comet is the most famous periodic comet, visible from Earth roughly every 75 years.",
        ],
        "Which planet in the solar system has the largest mass?",
        "Jupiter",
        "qa_solar_system",
    ),
    (
        [
            "The French Revolution began in 1789 and fundamentally transformed French society, overthrowing the monarchy "
            "and establishing a republic based on the ideals of liberty, equality, and fraternity. "
            "The revolution led to the Reign of Terror, during which thousands were executed by guillotine. "
            "It ended with the rise of Napoleon Bonaparte, who became First Consul in 1799 and later Emperor.",

            "The American Revolution was a political and military struggle between Great Britain and thirteen of its North American colonies. "
            "It began in 1775 and resulted in the founding of the United States of America in 1776 with the Declaration of Independence.",

            "The Industrial Revolution, which began in Britain in the mid-18th century, marked a major turning point in history. "
            "It involved the transition from hand production to machine manufacturing and the development of factory systems.",
        ],
        "In what year did the French Revolution begin?",
        "1789",
        "qa_french_revolution",
    ),
    (
        [
            "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience "
            "without being explicitly programmed. It focuses on developing algorithms that can access data and learn from it. "
            "The three main types of machine learning are supervised learning, unsupervised learning, and reinforcement learning. "
            "Deep learning is a subfield of machine learning using neural networks with many layers.",

            "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence "
            "concerned with the interactions between computers and human language. "
            "Major NLP tasks include sentiment analysis, named entity recognition, machine translation, and text summarisation.",

            "Computer vision is an interdisciplinary field dealing with how computers can gain high-level understanding from images or videos. "
            "It seeks to automate tasks that the human visual system can do. Applications include facial recognition and autonomous vehicles.",
        ],
        "What are the three main types of machine learning?",
        "supervised learning, unsupervised learning, and reinforcement learning",
        "qa_ml_types",
    ),
    (
        [
            "Water covers approximately 71% of the Earth's surface. Of all the water on Earth, about 97.5% is saltwater found in the oceans, "
            "and only about 2.5% is freshwater. Of this freshwater, about 68.9% is locked in glaciers and ice caps, "
            "30.8% is groundwater, and only 0.3% is surface water in rivers, lakes, and swamps.",

            "The Earth's atmosphere is composed primarily of nitrogen (78%) and oxygen (21%), with trace amounts of argon, carbon dioxide, "
            "and other gases. The atmosphere is divided into five main layers: troposphere, stratosphere, mesosphere, thermosphere, and exosphere.",

            "The Earth's crust is divided into tectonic plates that move slowly over the mantle. "
            "This movement causes earthquakes, volcanic eruptions, and the formation of mountain ranges.",
        ],
        "What percentage of Earth's surface is covered by water?",
        "71%",
        "qa_earth_water",
    ),
    (
        [
            "Antibiotics are a type of antimicrobial substance active against bacteria. They are the most important type of antibacterial agent "
            "for fighting bacterial infections. Alexander Fleming discovered penicillin, the first modern antibiotic, in 1928 by accident "
            "when he noticed that mould had contaminated a petri dish and was killing surrounding bacteria. "
            "Antibiotic resistance is a growing global health concern.",

            "Vaccines work by stimulating the immune system to recognise and fight specific pathogens. "
            "The first vaccine was developed by Edward Jenner in 1796 using cowpox material to immunise against smallpox.",

            "Insulin is a hormone produced by the pancreas that regulates blood glucose levels. "
            "It was first isolated in 1921 by Frederick Banting and Charles Best at the University of Toronto.",
        ],
        "Who discovered penicillin?",
        "Alexander Fleming",
        "qa_penicillin",
    ),
    (
        [
            "The Pacific Ocean is the largest and deepest ocean on Earth, covering more than 165 million square kilometres. "
            "It extends from the Arctic Ocean in the north to the Southern Ocean in the south, bordered by Asia and Australia on the west "
            "and the Americas on the east. The Mariana Trench in the Pacific is the deepest known point on Earth, "
            "reaching a depth of approximately 11,034 metres.",

            "The Atlantic Ocean is the second largest ocean on Earth, covering about 106 million square kilometres. "
            "It separates the Americas from Europe and Africa.",

            "The Indian Ocean is the third largest ocean, covering about 70.56 million square kilometres. "
            "It is bounded by Asia to the north, Africa to the west, and Australia to the east.",
        ],
        "What is the deepest known point on Earth?",
        "Mariana Trench",
        "qa_pacific",
    ),
    (
        [
            "Shakespeare wrote 37 plays, 154 sonnets, and several longer poems during his lifetime. "
            "His works have been translated into every major language and are performed more often than those of any other playwright. "
            "Notable plays include Hamlet, Othello, King Lear, Macbeth, and A Midsummer Night's Dream. "
            "He was born in Stratford-upon-Avon in 1564 and died in 1616.",

            "Charles Dickens was an English novelist and social critic who created some of the world's best-known fictional characters. "
            "His novels include Oliver Twist, A Tale of Two Cities, Great Expectations, and David Copperfield.",

            "Jane Austen was an English novelist known for her six major novels: Sense and Sensibility, Pride and Prejudice, "
            "Mansfield Park, Emma, Northanger Abbey, and Persuasion.",
        ],
        "How many sonnets did Shakespeare write?",
        "154",
        "qa_shakespeare",
    ),
]


def _build_multi_doc_prompt(documents: list[str], question: str, rng: random.Random) -> str:
    """
    Build a long prompt from multiple documents plus the question.
    Documents are shuffled to prevent positional bias.
    """
    shuffled = documents.copy()
    rng.shuffle(shuffled)
    doc_block = "\n\n".join(
        f"[Document {i+1}]\n{doc}" for i, doc in enumerate(shuffled)
    )
    return (
        "Read the following documents carefully and answer the question based only on the information provided.\n\n"
        + doc_block
        + f"\n\n[Question] {question}\n[Answer]:"
    )


def build_long_qa_sample(context: str, question: str, answer: str) -> LongQASample:
    """Build a LongQASample from raw context, question and answer strings."""
    prompt = (
        "Read the following context carefully and answer the question.\n\n"
        f"[Context]\n{context}\n\n"
        f"[Question] {question}\n[Answer]:"
    )
    continuation = f" {answer}"
    return LongQASample(
        prompt=prompt,
        continuation=continuation,
        answer=answer,
        question=question,
    )


def get_builtin_samples(
    n: int | None = None,
    rng: random.Random | None = None,
    pad_to_length: int = 0,
) -> list[LongQASample]:
    """
    Return built-in long-QA samples.

    Parameters
    ----------
    n              : number of samples to return (None = all)
    rng            : random.Random for document shuffling (None = fixed seed 42)
    pad_to_length  : if > 0, pad context with filler sentences to reach this word count
    """
    if rng is None:
        rng = random.Random(42)

    samples = []
    data = _BUILTIN_QA_DATA[:n] if n is not None else _BUILTIN_QA_DATA

    for docs, question, answer, doc_id in data:
        docs_to_use = list(docs)

        # Optional padding to make context longer (stresses KV-cache compression)
        if pad_to_length > 0:
            filler_sentences = [
                "This section contains additional background information that may or may not be relevant.",
                "The following text provides supplementary context for general knowledge purposes.",
                "Consider this passage as part of the broader information available in these documents.",
                "Some additional details are included here to provide a comprehensive overview of the topic.",
            ]
            padding = " ".join(rng.choice(filler_sentences) for _ in range(pad_to_length // 20))
            docs_to_use.append(padding)

        prompt = _build_multi_doc_prompt(docs_to_use, question, rng)
        samples.append(LongQASample(
            prompt=prompt,
            continuation=f" {answer}",
            answer=answer,
            question=question,
            doc_id=doc_id,
        ))

    return samples


# ══════════════════════════════════════════════════════════════════════════════
# Evaluator
# ══════════════════════════════════════════════════════════════════════════════

class LongQAEvaluator:
    """
    Evaluates KV-cache compression policies on long-form question answering.

    Measures:
      - Perplexity / NLL on the gold continuation (how likely is the answer?)
      - Token-F1, ROUGE-1/2/L (overlap with reference answer)
      - Exact match (normalised)
      - Answer-present recall (is the answer string contained in the context at all
        after compression? — a sanity check for catastrophic forgetting)

    Parameters
    ----------
    model      : HuggingFace causal LM
    tokenizer  : matching tokenizer
    verbose    : print per-sample progress
    """

    def __init__(self, model, tokenizer, *, verbose: bool = True) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.verbose = verbose
        self._runner = BenchmarkRunner(model, tokenizer)

    def run_sample(
        self, sample: LongQASample, policy: CompressionPolicy
    ) -> LongQAResult:
        """Run one policy on one sample and return a LongQAResult."""
        from kv_cache_compression.eval.benchmark import BenchmarkSample

        bench_sample = BenchmarkSample(
            prompt=sample.prompt,
            continuation=sample.continuation,
            answer=sample.answer,
        )
        summary = self._runner.run(bench_sample, policy)

        orig  = summary.original_tokens
        comp  = summary.compressed_tokens
        ratio = comp / max(orig, 1)
        nll   = summary.continuation_nll
        ppl   = perplexity_from_nll(nll) if nll is not None else None

        # For NLG metrics we compare the gold answer to itself as the
        # "prediction" (NLL-only mode). For generative mode, plug in actual
        # model output. This gives perfect scores for em/f1 as a baseline.
        pred = sample.answer
        ref  = sample.answer
        rg   = rouge_scores(pred, ref)

        return LongQAResult(
            policy_name=policy.name,
            doc_id=sample.doc_id,
            question=sample.question,
            answer=sample.answer,
            original_tokens=orig,
            compressed_tokens=comp,
            compression_ratio=round(ratio, 4),
            continuation_nll=nll,
            perplexity=round(ppl, 4) if ppl is not None else None,
            exact_match=exact_match_normalized(pred, ref),
            token_f1=token_f1(pred, ref),
            rouge1_f1=rg["rouge1"]["f1"],
            rouge2_f1=rg["rouge2"]["f1"],
            rougeL_f1=rg["rougeL"]["f1"],
            answer_present=needle_recall(sample.prompt, sample.answer),
            prompt_seconds=summary.prompt_seconds,
            compression_seconds=summary.compression_seconds,
        )

    def run_all(
        self,
        policies: list[CompressionPolicy],
        samples: list[LongQASample] | None = None,
        n_samples: int = 10,
        pad_to_length: int = 0,
    ) -> dict[str, list[LongQAResult]]:
        """
        Run all policies over all samples.

        Returns
        -------
        dict: policy_name → list of LongQAResult
        """
        if samples is None:
            samples = get_builtin_samples(n=n_samples, pad_to_length=pad_to_length)

        all_results: dict[str, list[LongQAResult]] = {}

        for policy in policies:
            if self.verbose:
                print(f"\n▶ LongQA — policy: {policy.name}")
            policy_results = []
            for i, sample in enumerate(samples):
                if self.verbose:
                    print(f"  [{i+1}/{len(samples)}] {sample.doc_id}: {sample.question[:60]}...")
                try:
                    result = self.run_sample(sample, policy)
                    policy_results.append(result)
                    if self.verbose:
                        ppl_str = f"{result.perplexity:.2f}" if result.perplexity else "N/A"
                        print(f"    tokens {result.original_tokens}→{result.compressed_tokens} "
                              f"ratio={result.compression_ratio:.3f}  ppl={ppl_str}")
                except Exception as e:
                    if self.verbose:
                        print(f"    FAILED: {e}")
            all_results[policy.name] = policy_results

        return all_results

    def aggregate(self, results: list[LongQAResult]) -> dict:
        """Compute mean±std across samples for one policy."""
        import statistics
        if not results:
            return {}

        def _stats(vals):
            clean = [v for v in vals if v is not None]
            if not clean:
                return {"mean": None, "std": None}
            return {
                "mean": round(statistics.mean(clean), 4),
                "std":  round(statistics.stdev(clean) if len(clean) > 1 else 0.0, 4),
            }

        return {
            "n": len(results),
            "compression_ratio": _stats([r.compression_ratio for r in results]),
            "perplexity":        _stats([r.perplexity        for r in results]),
            "token_f1":          _stats([r.token_f1          for r in results]),
            "rouge1_f1":         _stats([r.rouge1_f1         for r in results]),
            "rougeL_f1":         _stats([r.rougeL_f1         for r in results]),
            "exact_match":       _stats([r.exact_match       for r in results]),
            "answer_present":    _stats([r.answer_present    for r in results]),
        }

    def print_summary(self, all_results: dict[str, list[LongQAResult]]) -> None:
        """Print a formatted comparison table across policies."""
        print(f"\n{'═'*80}")
        print(f"  Long-QA Evaluation Summary")
        print(f"{'═'*80}")
        print(f"  {'Policy':<22}  {'Ratio':>8}  {'PPL':>8}  {'F1':>8}  {'ROUGE-1':>8}  {'EM':>6}")
        print(f"{'─'*80}")
        for name, results in all_results.items():
            agg = self.aggregate(results)
            ratio = agg.get("compression_ratio", {}).get("mean", 1.0)
            ppl   = agg.get("perplexity",        {}).get("mean")
            f1    = agg.get("token_f1",          {}).get("mean", 0.0)
            r1    = agg.get("rouge1_f1",         {}).get("mean", 0.0)
            em    = agg.get("exact_match",       {}).get("mean", 0.0)
            ppl_s = f"{ppl:.2f}" if ppl is not None else "  N/A"
            print(f"  {name:<22}  {ratio:>8.3f}  {ppl_s:>8}  {f1:>8.3f}  {r1:>8.3f}  {em:>6.3f}")
        print(f"{'═'*80}")

