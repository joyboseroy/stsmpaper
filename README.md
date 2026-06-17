# Spiking Sequence Machines and Transformers

Code for the paper:

**Spiking Sequence Machines and Transformers**
Joy Bose · arXiv:2605.00662 [cs.NE] · May 2026
[https://arxiv.org/abs/2605.00662](https://arxiv.org/abs/2605.00662)

---

## What this paper is about (plain English)

In 2007, I built a brain-inspired "sequence machine" using spiking neurons, which are
artificial neurons that fire in a specific time order to remember and predict
sequences, like notes in a melody or words in a sentence. This was my PhD
research at the University of Manchester.

In 2017, Google researchers invented the Transformer, the architecture
behind ChatGPT and every modern LLM.

**This paper asks: are these two things secretly doing the same thing?**

The answer is yes. Despite being invented independently, a decade apart,
by completely different research communities, both systems solve sequence
learning using the same five steps:

1. **Encode** the input into a high-dimensional representation
2. **Maintain context** across time steps
3. **Retrieve** similar stored patterns using cosine similarity
4. **Store** the result
5. **Decode** the output

Both use cosine similarity as the core retrieval mechanism. The paper
proves this formally and shows the mathematical connection between:

- **Spike timing** in spiking neural networks
- **Positional encoding** in transformers

They are two different ways of saying "this token came at position N."

**The practical finding:** You don't need sine waves for positional
encoding. A simpler rank-based approach works just as well, because
what actually matters is that the system can tell position 3 apart from
position 7, not the specific mathematical form used to express that.

---

## Technical Summary

- Formal equivalence between spiking SDM sequence machines and transformers
  across five functional operations
- **Phase-Latency Isomorphism**: spike timing and sinusoidal positional
  phase are linearly related
- **Lemma 1**: dot-product attention is invariant to this mapping up to
  a global scale factor
- Empirical result: frequency-compressed positional encoding fails on
  copy task; rank-based learned embedding matches or exceeds sinusoidal
- Conclusion: distance discriminability under dot-product similarity is
  what matters for positional representation, not sinusoidal form

---

## Follow-up: Length generalization (added June 2026)

The original experiment above tests all three encodings only at a single
fixed sequence length. A natural follow-up question: what happens if the
model is evaluated at lengths longer than it was trained on?

`pe_length_generalization.py` extends the comparison with RoPE (the
positional scheme used in most current open-weight models: Llama, Gemma,
Qwen, Mistral, Phi) and trains once at length 64, then evaluates without
retraining at lengths up to 192.

Result, on the same toy copy task:

| PE type | L=64 (trained) | L=96 | L=128 | L=192 |
|---|---|---|---|---|
| sinusoidal | 0.0003 | 0.0005 | 0.0008 | 0.0012 |
| STPE | 0.0003 | 0.0002 | 0.0002 | 0.0002 |
| rank (learned) | 0.0003 | N/A | N/A | N/A |
| RoPE | 0.0003 | 0.0003 | 0.0003 | 0.0003 |

The rank-based embedding, which performed best in the original
in-distribution experiment, cannot be evaluated past its training
length at all, since it is a lookup table sized to that length. STPE
holds flat under length extrapolation, indistinguishable from RoPE
within noise, despite coming from a completely different derivation.

This is still a toy synthetic task, so treat it as a first signal, not
a finished result. The next step, if useful, is repeating this on a
small real text corpus before drawing firmer conclusions.

---

## Code in this Repository

| File | What it does |
|---|---|
| `positional_encoding.py` | Standard sinusoidal positional encoding |
| `pe_vs_stpe.py` | Comparison: sinusoidal PE vs spike-timing PE |
| `frequency_compression.py` | Frequency-compressed PE — shows failure on copy task |
| `similarity_positional.py` | Cosine similarity analysis of positional representations |

---

## Requirements

```bash
pip install torch numpy matplotlib
```

---

## Citation

```bibtex
@article{bose2026spiking,
  title={Spiking Sequence Machines and Transformers},
  author={Bose, Joy},
  journal={arXiv preprint arXiv:2605.00662},
  year={2026}
}
```

---

## Background and Related Work

This paper connects my 2007 PhD thesis directly to modern transformer
architecture, 19 years later.

- **PhD Thesis (2007)**: Engineering a Sequence Machine Through Spiking
  Neurons Employing Rank-Order Codes.
  [ResearchGate](https://www.researchgate.net/publication/215990961_Engineering_a_sequence_machine_through_spiking_neurons_employing_rank-order_codes)
- **Furber, Brown, Bose et al. (2007)**: Sparse Distributed Memory Using
  Rank-Order Neural Codes. IEEE Transactions on Neural Networks.
  [DOI](https://doi.org/10.1109/TNN.2006.890804)
- **Vaswani et al. (2017)**: Attention Is All You Need. NeurIPS 2017.

---

## Author

**Dr. Joy Bose** — Senior Data Scientist & AI Architect, Ericsson Global

[LinkedIn](https://linkedin.com/in/joyboseroy) ·
[Google Scholar](https://scholar.google.com/citations?user=1E0YgA4AAAAJ) ·
[arXiv author page](https://arxiv.org/search/?searchtype=author&query=Bose+J) ·
[Personal site](https://joyboseroy.github.io)
