# STSM: Positional Encoding is About Order, Not Sinusoids

## The core idea

Transformers need to know where each token sits in a sequence. The standard approach uses sine and cosine waves at different frequencies. This repo asks a simpler question: do transformers actually need sine waves, or just a reliable sense of order?

The answer turns out to matter. Two encoding formulas that look almost identical behave very differently in practice, and understanding why reveals something fundamental about what positional encoding is actually doing.

---

## The key result

These two expressions look similar but are not equivalent:

```python
# Fails to learn
sin((pos / L) * omega)

# Works correctly
(1 / L) * sin(pos * omega)
```

One structural difference causes the model to fail completely. The first formula compresses frequencies in a way that destroys distance information, even though it preserves ordering. The second preserves both.

---

## What this means

Transformers do not fundamentally require sinusoidal encoding. What they require is a representation whose distance structure is preserved under dot-product similarity. A pure rank-based encoding, with no sine waves at all, satisfies this condition and learns just as well as the standard approach.

---

## Experiments

Three positional encodings are compared on a sequence copy task:

| Encoding | Description | Learns? |
|---|---|---|
| Sinusoidal PE | Standard transformer encoding | Yes |
| Frequency-compressed | sin((pos/L) times omega) | No |
| Rank-normalised (STPE) | (1/L) times sin(pos times omega) | Yes |

**Figure 1a** shows that STPE and standard PE have a perfectly linear similarity relationship, confirming they preserve the same distance structure.

![Figure 1a: Similarity structure](positional_encodings.png)

**Figure 1b** shows dot product versus positional distance for PE and STPE rescaled to the same units. The two overlap almost exactly.

![Figure 1b: Distance structure](fig1b_distance.png)

**Figure 1c** shows that frequency-compressed encoding preserves ordering (Spearman rho = 0.965) but destroys metric structure (Pearson r = 0.706), which is why attention fails.

![Figure 1c: Frequency compression](fig1c_freqcomp.png)

**Figure 2** shows learning curves for all three encodings on the copy task.

![Figure 2: Learning curves](stsmfig2.png)

---

## Running the experiments

```bash
python stsm1a.py    # similarity structure comparison
python stsm1b.py    # distance vs dot product
python stsm1c.py    # frequency compression analysis
python train_copy_task.py    # learning curves
```

Requirements: Python 3.8+, PyTorch, matplotlib, numpy

---

## Connection to spiking neural networks

This question connects directly to timing-based representations in spiking neural networks, which formed the basis of my doctoral research. Both transformers and spiking networks use similarity over ordered sequences to retrieve information. The structural requirement turns out to be the same in both cases.

---

## Limitations

Small synthetic task, single seed, minimal model. This is a mechanistic insight, not a benchmark result. The claim is about what positional encoding structurally requires, not about performance at scale.

---

## Paper

Preprint submitted to arXiv. Link will be added on acceptance.

---

## Author

Joy Bose  
PhD in Computer Science, University of Manchester  
Data Scientist and ML Researcher  
[joyboseroy@gmail.com](mailto:joyboseroy@gmail.com) | [LinkedIn](https://linkedin.com/in/joyboseroy) | [Google Scholar](https://scholar.google.com/citations?user=1E0YgA4AAAAJ&hl=en)
