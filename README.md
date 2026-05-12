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

## Background & Related Work

This paper connects my 2007 PhD thesis directly to modern transformer
architecture, 19 years later.

- **PhD Thesis (2007)**: Engineering a Sequence Machine Through Spiking
  Neurons Employing Rank-Order Codes.
  [British Library EThOS](https://ethos.bl.uk/OrderDetails.do?uin=uk.bl.ethos.789385)
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
