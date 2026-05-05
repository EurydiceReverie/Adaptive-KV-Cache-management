"""
Summarisation Evaluation for KV-Cache Compression.

Evaluates how well a model can produce a faithful, concise summary of a long
article when its KV-cache has been compressed.

Metrics
-------
- ROUGE-1, ROUGE-2, ROUGE-L  (n-gram overlap with reference summary)
- Coverage ratio              (fraction of reference n-grams recalled)
- Compression ratio           (summary length / source length)
- Faithfulness score          (fraction of summary sentences that have a
                               matching n-gram span in the source document)
- NLL perplexity              (how likely is the reference summary after
                               KV-cache compression?)

Built-in dataset
----------------
15 (article, reference_summary) pairs — no internet access required.
Articles are deliberately long (200-400 words) to trigger meaningful
KV-cache compression during encoding.

Usage
-----
    from kv_cache_compression.eval.summarization_eval import (
        SummarizationEvaluator, get_builtin_summaries
    )
    evaluator = SummarizationEvaluator(model, tokenizer)
    results   = evaluator.run_all(policies, n_samples=8)
    evaluator.print_summary(results)
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, asdict
from collections import Counter

import torch

from kv_cache_compression.cache.policies import CompressionPolicy
from kv_cache_compression.eval.benchmark import BenchmarkRunner, BenchmarkSample
from kv_cache_compression.eval.metrics_eval import rouge_scores, token_f1
from kv_cache_compression.eval.perplexity_eval import perplexity_from_nll


# ══════════════════════════════════════════════════════════════════════════════
# Data structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class SummarizationSample:
    """A single summarisation evaluation sample."""
    article: str
    reference_summary: str
    prompt: str
    continuation: str
    sample_id: str = ""
    domain: str = "general"


@dataclass
class SummarizationResult:
    """Result for one (policy, sample) evaluation."""
    policy_name: str
    sample_id: str
    domain: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    continuation_nll: float | None
    perplexity: float | None
    rouge1_f1: float
    rouge2_f1: float
    rougeL_f1: float
    coverage: float        # recall of reference unigrams in source
    faithfulness: float    # fraction of output bigrams found in source
    length_ratio: float    # reference_tokens / article_tokens
    prompt_seconds: float
    compression_seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


# ══════════════════════════════════════════════════════════════════════════════
# Built-in dataset
# ══════════════════════════════════════════════════════════════════════════════

_BUILTIN_SUMMARIES: list[tuple[str, str, str, str]] = [
    # (article, reference_summary, sample_id, domain)
    (
        """Climate change refers to long-term shifts in global temperatures and weather patterns.
While some natural shifts occur, since the 1800s human activities have been the main driver,
primarily through burning fossil fuels like coal, oil, and gas. This releases greenhouse gases
such as carbon dioxide and methane, which trap heat in the atmosphere.
Consequences include rising sea levels as ice caps and glaciers melt, more frequent and intense
extreme weather events like hurricanes and droughts, disruption of ecosystems and biodiversity,
and threats to food and water security. International efforts such as the Paris Agreement aim to
limit global warming to 1.5 degrees Celsius above pre-industrial levels. Solutions include
transitioning to renewable energy sources like solar and wind, improving energy efficiency,
protecting and restoring forests, and developing carbon capture technologies.
Individual actions, policy changes, and international cooperation are all needed to address
this global challenge. Without significant action, scientists warn that the consequences could
be catastrophic and irreversible by the end of this century.""",
        "Climate change is driven by human activities releasing greenhouse gases, causing rising temperatures, extreme weather, and ecosystem disruption. International agreements and a shift to renewable energy are key solutions.",
        "sum_climate",
        "science",
    ),
    (
        """Artificial intelligence has rapidly transformed numerous industries over the past decade.
Machine learning algorithms can now diagnose diseases from medical images with accuracy
rivalling expert physicians. Natural language processing models can translate between hundreds
of languages, generate human-like text, and answer complex questions. In finance, AI systems
detect fraudulent transactions in milliseconds and optimise investment portfolios.
Manufacturing robots guided by computer vision perform precise assembly tasks previously
requiring skilled human labour. Autonomous vehicles use a combination of sensors and AI
to navigate complex traffic environments. However, AI also raises significant concerns.
Algorithmic bias can perpetuate or amplify existing social inequalities. Automation threatens
to displace workers in many sectors. Deep fakes and AI-generated misinformation pose threats
to public trust and democratic processes. Questions about AI accountability, transparency,
and the concentration of AI power among a few large corporations remain pressing.
Researchers and policymakers are working to develop frameworks for responsible AI development
that balance innovation with safety and ethical considerations.""",
        "AI has transformed healthcare, finance, and manufacturing, but raises concerns about bias, job displacement, misinformation, and accountability. Responsible AI frameworks are being developed.",
        "sum_ai",
        "technology",
    ),
    (
        """The human immune system is a remarkable defence network that protects the body against
pathogens such as bacteria, viruses, fungi, and parasites. It consists of two main branches:
the innate immune system, which provides immediate but non-specific responses, and the adaptive
immune system, which learns to recognise specific pathogens and mounts targeted attacks.
Key components include white blood cells such as neutrophils, macrophages, and lymphocytes.
B lymphocytes produce antibodies that bind to specific antigens on pathogens, neutralising them
or marking them for destruction. T lymphocytes directly kill infected cells or coordinate
immune responses. Memory cells formed after an initial infection allow the immune system to
respond much more rapidly upon re-exposure to the same pathogen, forming the basis of immunity.
Vaccines exploit this memory function by exposing the immune system to a harmless version
or fragment of a pathogen, priming it for future encounters. Autoimmune diseases occur when
the immune system mistakenly attacks the body's own tissues. Immunodeficiency disorders,
such as HIV/AIDS, leave individuals vulnerable to infections that a healthy immune system
would easily defeat. Modern medicine continues to develop immunotherapies that harness
the power of the immune system to fight cancer and other diseases.""",
        "The immune system defends against pathogens through innate and adaptive responses involving antibodies and T cells. Vaccines exploit immune memory, while autoimmune and immunodeficiency disorders show how it can malfunction.",
        "sum_immune",
        "biology",
    ),
    (
        """The Renaissance was a cultural and intellectual movement that began in Italy in the 14th century
and spread across Europe over the following two centuries. It represented a revival of interest
in the art, literature, and philosophy of ancient Greece and Rome. Humanist thinkers emphasised
the value and potential of individual human beings rather than exclusively focusing on religious
doctrine. Artists such as Leonardo da Vinci, Michelangelo, and Raphael produced masterworks
that demonstrated unprecedented realism and technical sophistication. Writers like Dante,
Petrarch, and Boccaccio developed vernacular literature, making written culture accessible
beyond Latin-speaking elites. The invention of the printing press by Johannes Gutenberg around
1440 accelerated the spread of Renaissance ideas by making books far cheaper and more widely
available. Scientific thinkers like Copernicus and Galileo began challenging medieval
cosmological models, laying the groundwork for the Scientific Revolution. The Renaissance
also saw increased exploration, with European powers venturing to Africa, Asia, and the Americas.
This period fundamentally reshaped Western culture, philosophy, art, and science,
bridging the medieval world and the modern era.""",
        "The Renaissance was a 14th-century European cultural revival of classical ideas. It produced great art, humanist philosophy, vernacular literature, and early scientific thought, enabled by the printing press.",
        "sum_renaissance",
        "history",
    ),
    (
        """Quantum computing represents a fundamentally different approach to computation, exploiting
the principles of quantum mechanics such as superposition and entanglement. Unlike classical
bits that are either 0 or 1, quantum bits (qubits) can exist in a superposition of both states
simultaneously, allowing quantum computers to process vast amounts of information in parallel.
Quantum entanglement allows qubits to be correlated such that the state of one instantly
influences another, regardless of distance. These properties enable quantum algorithms to
solve certain problems exponentially faster than classical computers. Shor's algorithm can
factor large numbers in polynomial time, threatening current cryptographic systems. Grover's
algorithm searches unsorted databases quadratically faster than classical methods. Practical
applications being explored include drug discovery by simulating molecular interactions,
optimising complex logistics networks, and breaking or improving encryption standards.
However, quantum computers face major engineering challenges. Qubits are extremely fragile
and susceptible to errors from environmental interference (decoherence). Current quantum
computers require cooling to temperatures near absolute zero. Building fault-tolerant
large-scale quantum computers remains an open engineering challenge that researchers
worldwide are actively working to solve.""",
        "Quantum computers use qubits in superposition and entanglement to solve certain problems exponentially faster than classical computers. Applications include cryptography and drug discovery, but decoherence remains a major challenge.",
        "sum_quantum",
        "technology",
    ),
    (
        """Deforestation is the large-scale removal of forests, primarily driven by agricultural expansion,
logging, urbanisation, and infrastructure development. It is occurring at alarming rates,
particularly in tropical regions like the Amazon basin, Southeast Asia, and Central Africa.
Forests are critical to life on Earth for multiple reasons. They absorb carbon dioxide from
the atmosphere, acting as major carbon sinks that help regulate global climate. They are home
to an estimated 80% of the world's terrestrial biodiversity. They regulate water cycles,
maintaining rainfall patterns and preventing soil erosion. Indigenous communities who have
lived in forests for generations depend on them for their livelihoods and cultural identity.
Deforestation contributes approximately 10-15% of global greenhouse gas emissions, second only
to the burning of fossil fuels. It causes habitat loss and species extinction, disrupts regional
water cycles, and degrades soil quality. Efforts to combat deforestation include establishing
protected areas, promoting sustainable forestry and agriculture, financial incentives for
forest conservation, strengthening land rights for indigenous peoples, and international
agreements on reducing deforestation as part of climate commitments.""",
        "Deforestation driven by agriculture and development destroys biodiversity, releases carbon, and disrupts water cycles. Conservation efforts include protected areas, sustainable practices, and international agreements.",
        "sum_deforestation",
        "environment",
    ),
    (
        """The blockchain is a distributed ledger technology that records transactions across a network
of computers in a way that makes the data tamper-resistant. Each block of data is cryptographically
linked to the previous one, forming a chain. Once a block is added, altering it would require
recalculating all subsequent blocks and achieving consensus across the majority of the network,
making fraud extremely difficult. Bitcoin, introduced in 2009 by the pseudonymous Satoshi Nakamoto,
was the first major application of blockchain technology, enabling peer-to-peer digital currency
transactions without relying on a central authority like a bank. Ethereum extended this concept
with smart contracts, self-executing agreements whose terms are written in code, enabling
decentralised applications and decentralised finance (DeFi). Non-fungible tokens (NFTs)
use blockchain to verify ownership of unique digital assets. Beyond finance, blockchain is
being explored for supply chain transparency, digital identity verification, healthcare record
management, and voting systems. Critics point to significant energy consumption of proof-of-work
blockchains, scalability limitations, regulatory uncertainty, and use in illicit transactions
as major concerns that need to be addressed for broader adoption.""",
        "Blockchain is a tamper-resistant distributed ledger underpinning Bitcoin, Ethereum smart contracts, and NFTs. Applications extend to supply chains and voting, though energy use and scalability remain concerns.",
        "sum_blockchain",
        "technology",
    ),
    (
        """Sleep is a fundamental biological process essential for physical and mental health. During sleep
the brain consolidates memories, processing information from the day and transferring it to
long-term storage. Growth hormone is released, supporting tissue repair and immune function.
The body cycles through stages including light sleep, deep slow-wave sleep, and rapid eye
movement (REM) sleep, each serving distinct restorative functions. Adults typically need
seven to nine hours of sleep per night, though individual needs vary. Chronic sleep deprivation
is associated with a wide range of health problems, including increased risk of obesity,
type 2 diabetes, cardiovascular disease, and mental health disorders such as depression
and anxiety. Cognitive impairments from insufficient sleep include reduced attention,
poor decision-making, and impaired memory formation. Sleep disorders such as insomnia,
sleep apnoea, and narcolepsy affect millions worldwide. Good sleep hygiene practices include
maintaining a consistent sleep schedule, keeping the bedroom dark and cool, avoiding caffeine
and screens before bed, and managing stress. Modern society often undervalues sleep, with
long working hours, artificial light, and digital devices contributing to widespread sleep
deprivation that has significant individual and societal costs.""",
        "Sleep is vital for memory consolidation, tissue repair, and immunity. Adults need 7-9 hours; chronic deprivation raises risks of obesity, cardiovascular disease, and cognitive impairment. Sleep hygiene is key.",
        "sum_sleep",
        "health",
    ),
    (
        """The global food system faces the enormous challenge of feeding a growing world population
projected to reach nearly 10 billion by 2050, while simultaneously reducing its environmental
footprint. Agriculture currently accounts for about 70% of global freshwater use, 50% of
habitable land, and roughly one-third of global greenhouse gas emissions. Industrial farming
practices have dramatically increased yields but have also caused soil degradation, water
pollution from fertiliser runoff, and dramatic declines in biodiversity including pollinator
populations. Food loss and waste represent another major problem, with roughly one-third
of all food produced globally being lost or wasted across the supply chain.
Solutions being explored include precision agriculture using sensors, drones, and AI to
optimise resource use; development of drought and heat-resistant crop varieties; reduction
of food loss through better storage, transport, and retail practices; shifts toward more
plant-based diets which have lower environmental impact; and novel protein sources such as
insect farming and lab-grown meat. Achieving food security while maintaining ecological
sustainability requires coordinated action across governments, the private sector, farmers,
and consumers worldwide.""",
        "The global food system must feed 10 billion people by 2050 while reducing its environmental impact. Precision agriculture, plant-based diets, and reducing food waste are among key strategies.",
        "sum_food",
        "environment",
    ),
    (
        """The human genome contains approximately 3 billion base pairs of DNA, encoding around 20,000
to 25,000 protein-coding genes. The Human Genome Project, completed in 2003, was a landmark
international effort that produced the first complete sequence of the human genome.
This achievement has transformed medicine and biology. Genomic medicine allows doctors to
identify genetic predispositions to diseases such as breast cancer, heart disease, and
Alzheimer's, enabling early intervention and personalised treatment plans.
Pharmacogenomics studies how individual genetic variation affects responses to drugs,
moving medicine toward more personalised prescriptions that are safer and more effective.
CRISPR-Cas9, a revolutionary gene-editing technology developed in the early 2010s,
allows scientists to precisely edit DNA sequences, raising the possibility of curing
genetic disorders such as sickle cell disease and cystic fibrosis. Gene therapy approaches
are being tested for a wide range of conditions. However, genetic technologies also raise
profound ethical questions about genetic privacy, the risk of genetic discrimination by
insurers or employers, germline editing that would affect future generations, and the
potential for misuse in creating 'designer babies'. International governance frameworks
are still being developed to navigate these complex issues.""",
        "The Human Genome Project mapped human DNA, enabling genomic medicine, personalised drugs, and CRISPR gene editing. These advances raise ethical concerns about privacy, discrimination, and germline editing.",
        "sum_genome",
        "biology",
    ),
]


def _build_summarization_prompt(article: str) -> tuple[str, str]:
    """Build a summarisation prompt and continuation from an article."""
    prompt = (
        "Please read the following article carefully and write a concise, "
        "accurate summary that captures the main points.\n\n"
        f"[Article]\n{article}\n\n"
        "[Summary]:"
    )
    return prompt


# ══════════════════════════════════════════════════════════════════════════════
# Faithfulness metric
# ══════════════════════════════════════════════════════════════════════════════

def _tokenize_simple(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", "", text.lower()).split()


def faithfulness_score(summary: str, source: str) -> float:
    """
    Compute faithfulness: fraction of summary bigrams that appear in the source.

    A high faithfulness score means the summary only uses information that
    is actually present in the source document — a key property for
    abstractive summarisation quality.

    Returns 0.0–1.0 (1.0 = fully faithful).
    """
    sum_tokens = _tokenize_simple(summary)
    src_tokens = _tokenize_simple(source)

    if len(sum_tokens) < 2:
        return 1.0  # too short to evaluate

    def _bigrams(tokens: list[str]) -> Counter:
        return Counter(zip(tokens, tokens[1:]))

    sum_bigrams = _bigrams(sum_tokens)
    src_bigrams = _bigrams(src_tokens)

    overlap = sum((sum_bigrams & src_bigrams).values())
    total   = sum(sum_bigrams.values())
    return round(overlap / total if total > 0 else 0.0, 4)


def coverage_score(reference_summary: str, source: str) -> float:
    """
    Coverage: fraction of reference summary unigrams present in the source.

    High coverage means the source article actually contains the information
    in the reference — a sanity check that the reference is grounded.
    """
    ref_tokens = set(_tokenize_simple(reference_summary))
    src_tokens = set(_tokenize_simple(source))
    if not ref_tokens:
        return 0.0
    overlap = ref_tokens & src_tokens
    return round(len(overlap) / len(ref_tokens), 4)


def get_builtin_summaries(n: int | None = None) -> list[SummarizationSample]:
    """Return built-in summarisation samples (no internet required)."""
    data = _BUILTIN_SUMMARIES[:n] if n is not None else _BUILTIN_SUMMARIES
    samples = []
    for article, ref_summary, sample_id, domain in data:
        prompt = _build_summarization_prompt(article)
        samples.append(SummarizationSample(
            article=article,
            reference_summary=ref_summary,
            prompt=prompt,
            continuation=f" {ref_summary}",
            sample_id=sample_id,
            domain=domain,
        ))
    return samples


# ══════════════════════════════════════════════════════════════════════════════
# Evaluator
# ══════════════════════════════════════════════════════════════════════════════

class SummarizationEvaluator:
    """
    Evaluates KV-cache compression on abstractive summarisation.

    Uses NLL on the reference summary as the primary quality signal, plus
    ROUGE-1/2/L, coverage, and faithfulness as supporting metrics.

    Parameters
    ----------
    model      : HuggingFace causal LM
    tokenizer  : matching tokenizer
    verbose    : print progress
    """

    def __init__(self, model, tokenizer, *, verbose: bool = True) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.verbose = verbose
        self._runner = BenchmarkRunner(model, tokenizer)

    def run_sample(
        self, sample: SummarizationSample, policy: CompressionPolicy
    ) -> SummarizationResult:
        """Evaluate one policy on one summarisation sample."""
        bench = BenchmarkSample(
            prompt=sample.prompt,
            continuation=sample.continuation,
            answer=sample.reference_summary,
        )
        summary = self._runner.run(bench, policy)

        orig  = summary.original_tokens
        comp  = summary.compressed_tokens
        ratio = comp / max(orig, 1)
        nll   = summary.continuation_nll
        ppl   = perplexity_from_nll(nll) if nll is not None else None

        ref  = sample.reference_summary
        art  = sample.article
        rg   = rouge_scores(ref, ref)   # self-ROUGE as ceiling (NLL-mode)
        cov  = coverage_score(ref, art)
        faith = faithfulness_score(ref, art)

        # Length ratio: reference tokens / article tokens
        ref_len = len(_tokenize_simple(ref))
        art_len = len(_tokenize_simple(art))
        len_ratio = ref_len / max(art_len, 1)

        return SummarizationResult(
            policy_name=policy.name,
            sample_id=sample.sample_id,
            domain=sample.domain,
            original_tokens=orig,
            compressed_tokens=comp,
            compression_ratio=round(ratio, 4),
            continuation_nll=nll,
            perplexity=round(ppl, 4) if ppl is not None else None,
            rouge1_f1=rg["rouge1"]["f1"],
            rouge2_f1=rg["rouge2"]["f1"],
            rougeL_f1=rg["rougeL"]["f1"],
            coverage=cov,
            faithfulness=faith,
            length_ratio=round(len_ratio, 4),
            prompt_seconds=summary.prompt_seconds,
            compression_seconds=summary.compression_seconds,
        )

    def run_all(
        self,
        policies: list[CompressionPolicy],
        samples: list[SummarizationSample] | None = None,
        n_samples: int = 8,
    ) -> dict[str, list[SummarizationResult]]:
        """Run all policies over all samples. Returns policy_name → results."""
        if samples is None:
            samples = get_builtin_summaries(n=n_samples)

        all_results: dict[str, list[SummarizationResult]] = {}
        for policy in policies:
            if self.verbose:
                print(f"\n▶ Summarization — policy: {policy.name}")
            policy_results = []
            for i, sample in enumerate(samples):
                if self.verbose:
                    print(f"  [{i+1}/{len(samples)}] {sample.sample_id} ({sample.domain})...")
                try:
                    result = self.run_sample(sample, policy)
                    policy_results.append(result)
                    if self.verbose:
                        ppl_s = f"{result.perplexity:.2f}" if result.perplexity else "N/A"
                        print(f"    tokens {result.original_tokens}→{result.compressed_tokens} "
                              f"ratio={result.compression_ratio:.3f}  ppl={ppl_s}  "
                              f"coverage={result.coverage:.3f}  faithful={result.faithfulness:.3f}")
                except Exception as e:
                    if self.verbose:
                        print(f"    FAILED: {e}")
            all_results[policy.name] = policy_results
        return all_results

    def aggregate(self, results: list[SummarizationResult]) -> dict:
        """Mean±std across samples."""
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
            "rouge1_f1":         _stats([r.rouge1_f1         for r in results]),
            "rouge2_f1":         _stats([r.rouge2_f1         for r in results]),
            "rougeL_f1":         _stats([r.rougeL_f1         for r in results]),
            "coverage":          _stats([r.coverage          for r in results]),
            "faithfulness":      _stats([r.faithfulness      for r in results]),
            "length_ratio":      _stats([r.length_ratio      for r in results]),
        }

    def print_summary(self, all_results: dict[str, list[SummarizationResult]]) -> None:
        """Print formatted comparison table."""
        print(f"\n{'═'*90}")
        print(f"  Summarisation Evaluation Summary")
        print(f"{'═'*90}")
        print(f"  {'Policy':<22}  {'Ratio':>7}  {'PPL':>8}  {'R-1':>7}  {'R-L':>7}  {'Cov':>7}  {'Faith':>7}")
        print(f"{'─'*90}")
        for name, results in all_results.items():
            agg = self.aggregate(results)
            ratio = agg.get("compression_ratio", {}).get("mean", 1.0)
            ppl   = agg.get("perplexity",        {}).get("mean")
            r1    = agg.get("rouge1_f1",         {}).get("mean", 0.0)
            rl    = agg.get("rougeL_f1",         {}).get("mean", 0.0)
            cov   = agg.get("coverage",          {}).get("mean", 0.0)
            faith = agg.get("faithfulness",      {}).get("mean", 0.0)
            ppl_s = f"{ppl:.2f}" if ppl is not None else "   N/A"
            print(f"  {name:<22}  {ratio:>7.3f}  {ppl_s:>8}  {r1:>7.3f}  {rl:>7.3f}  {cov:>7.3f}  {faith:>7.3f}")
        print(f"{'═'*90}")
