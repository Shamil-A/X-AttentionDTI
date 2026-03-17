# KIBAp2 — Drug-Target Affinity Prediction Pipeline

> **Goal:** Given a drug molecule (SMILES) and a protein sequence (amino acids), predict how strongly the drug binds to the protein — expressed as a **KIBA affinity score**.

---

## Table of Contents

1. [Overview](#overview)
2. [Step 0 — Raw Input](#step-0--raw-input)
3. [Step 1 — Drug → Hypergraph](#step-1--drug--hypergraph)
   - [Why Not SMILES?](#why-not-smiles)
   - [Atom Feature Vector (49-dim)](#atom-feature-vector-49-dim)
   - [Hyperedge Types](#hyperedge-types)
4. [Step 2 — Drug Encoder (HypergraphConv)](#step-2--drug-encoder-hypergraphconv)
5. [Step 3 — Protein → ESM-2 Tokens](#step-3--protein--esm-2-tokens)
6. [Step 4 — Teacher CLS Embedding](#step-4--teacher-cls-embedding)
7. [Step 5 — Student Protein Encoder (ESM-2 35M)](#step-5--student-protein-encoder-esm-2-35m)
8. [Step 6 — Teacher-Student Fusion](#step-6--teacher-student-fusion)
9. [Step 7 — Bidirectional Cross-Attention](#step-7--bidirectional-cross-attention)
10. [Step 8 — MLP Predictor](#step-8--mlp-predictor)
11. [Step 9 — Loss Function](#step-9--loss-function)
12. [Final Result](#final-result)

---

## Overview

```
Drug SMILES  ──► Hypergraph ──► HypergraphConv ──► drug_global [512]
                                                         │
                                                    Cross-Attention ──► MLP ──► KIBA score
                                                         │
Protein AA   ──► ESM-2 35M ──► Linear(480→512)──► protein_seq [1024, 512]
                     ▲
                     │ distillation blend (λ=0.5)
ESM-2 650M   ──► teacher_cls [1280] (precomputed, frozen)
```

---

## Step 0 — Raw Input

| Field | Value |
|---|---|
| Drug SMILES | `Cn1nc(C(F)(F)F)c2c(=O)c3cc(Cl)ccc3n(O)c21` |
| Protein length | 1000 amino acids |
| True affinity (raw KIBA) | **11.100** |
| True affinity (log1p) | **2.4932** ← training target |

> The model is trained on **log1p(KIBA)**, not raw KIBA, to compress the large value range.

---

## Step 1 — Drug → Hypergraph

### Why Not SMILES?

A SMILES string is a **1-D linear sequence** representation of a molecule:

```
Cn1nc(C(F)(F)F)c2c(=O)c3cc(Cl)ccc3n(O)c21
```

**Problems with SMILES directly:**

| Issue | Detail |
|---|---|
| Non-unique | The same molecule can have many valid SMILES strings |
| No native ring support | Rings must be inferred by a model; they aren't structural primitives |
| No multi-atom relationships | Can't represent a 6-membered aromatic ring as a single unit |
| String-order dependent | CNNs/LSTMs must figure out 2-D topology from 1-D characters |

**A hypergraph solves this** by building directly from the RDKit molecular graph:

- Each **atom** is a node → same regardless of which SMILES string you write
- **Hyperedges** can connect 2, 3, or more atoms at once → rings become first-class citizens

---

### Atom Feature Vector (49-dim)

Every atom is represented by a **49-dimensional feature vector**. Each group of dimensions encodes a specific chemical property:

| Group | Indices | Dimensions | Encoding | Values / Meaning |
|---|---|---|---|---|
| **Atom Type** | 0 – 15 | 16 | One-hot | C, N, O, F, P, S, Cl, Br, I, Si, B, Na, K, Ca, Fe, unknown |
| **Degree** | 16 – 20 | 5 | One-hot | 0, 1, 2, 3, 4+ neighbours |
| **Num Implicit H** | 21 – 25 | 5 | One-hot | 0, 1, 2, 3, 4+ implicit hydrogens |
| **Hybridization** | 26 – 30 | 5 | One-hot | SP, SP2, SP3, SP3D, SP3D2 |
| **Formal Charge** | 31 – 38 | 8 | One-hot | −3, −2, −1, 0, +1, +2, +3, other |
| **Radical Electrons** | 39 – 41 | 3 | One-hot | 0, 1, 2+ unpaired electrons |
| **Ring & Aromaticity** | 42 – 45 | 4 | Binary flags | `in_ring`, `is_aromatic`, `chiral_CW`, `chiral_CCW` |
| **Total Hs** | 46 | 1 | Integer | 0–4 total H atoms (implicit + explicit) |
| **Atomic Mass** | 47 | 1 | Float (÷100) | ~0.01 (H) to ~2.07 (Pb) |
| **Ring Membership** | 48 | 1 | Binary | 1 if atom is in any ring, else 0 |

**Total: 16 + 5 + 5 + 5 + 8 + 3 + 4 + 1 + 1 + 1 = 49 ✓**

**Example — Atom 0 (Nitrogen, N-CH₃ group):**

```
Index:  0   1   2  ...  15  16  17  ...  20  21  ...  25  26  ...  30  31  ...  38  39  40  41  42  43  44  45  46     47       48
Value:  1   0   0  ...   0   1   0  ...   0   0  ...   0   1  ...   0   1  ...   0   1   0   0   1   0   0   0   4   0.120    0.0
        └── N ──┘        └degree=1┘       └─ 0 H ─┘      └─ SP2 ─┘      └ charge=0 ┘    └0 rad┘  ↑ring ↑arom       total_H  mass  ring?
```

---

### Hyperedge Types

From the 21-atom, 23-bond molecule we construct **47 hyperedges** of three kinds:

| Type | Count | Description | Atoms connected |
|---|---|---|---|
| **Bond** | 23 | One per covalent bond | 2 atoms (one per bond) |
| **Ring** | 3 | One per ring system | All atoms in the ring (3–7) |
| **Environment** | 21 | One per atom | Atom + all 1-hop neighbours |

```
Hyperedge #0  type=bond  atoms=[0, 1]        ← N–N single bond
Hyperedge #23 type=ring  atoms=[0,1,2,7,9,10] ← Pyrazole ring (6 atoms)
Hyperedge #24 type=ring  atoms=[7,8,11,12]    ← Imidazole ring (4 atoms)
Hyperedge #25 type=ring  atoms=[12,13,14,15]  ← Benzene ring (4 atoms shown)
Hyperedge #26 type=env   atoms=[0,...]        ← Atom 0 + its neighbours
```

**Why three types?**
- Bond edges → local connectivity (like a normal graph)
- Ring edges → capture aromaticity and ring strain in one step
- Env edges → each atom "sees" its whole neighbourhood simultaneously

---

## Step 2 — Drug Encoder (HypergraphConv)

**Architecture:** `Linear(49 → 512)` → 3× `HypergraphConv` layers → `GlobalMeanPool`

Each HypergraphConv layer:
```
Node → Hyperedge aggregation (mean/sum over member nodes)
     → Node update (hyperedge messages → node)
     → Residual connection
     → LayerNorm
     → GELU activation
```

| Stage | Shape | Mean | Max |
|---|---|---|---|
| Input projection `Linear(49→512)` | [21, 512] | 0.0083 | 1.296 |
| After Layer 1 | [21, 512] | 0.0861 | 3.781 |
| After Layer 2 | [21, 512] | 0.1002 | 4.222 |
| After Layer 3 | [21, 512] | 0.0428 | 4.438 |
| Global mean pool | [1, 512] | −0.021 | 1.246 |

**Outputs:**
- `drug_global` [1, 512] — single summary vector for the whole molecule
- `node_embs` [21, 512] — per-atom vectors (used later in P→D attention)

---

## Step 3 — Protein → ESM-2 Tokens

The 1000-amino-acid sequence is tokenized by ESM-2's tokenizer:

```
Real AA:  1000 amino acids
+ BOS/EOS tokens: +2
= 1002 real tokens → padded to 1024
```

| Tensor | Shape | Description |
|---|---|---|
| `input_ids` | [1, 1024] | Integer token IDs (0–23) |
| `attention_mask` | [1, 1024] | 1 = real token, 0 = padding |

- The 22 padding positions are **masked out** — the model ignores them.
- Token 0 is the **CLS token**, whose output becomes the global protein summary.

---

## Step 4 — Teacher CLS Embedding

| Property | Value |
|---|---|
| Model | ESM-2 650M (facebook/esm2_t33_650M_UR50D) |
| Output dim | 1280 |
| Status | **Completely frozen** — never updated |
| When computed | Once per protein, saved to disk |

```
teacher_cls [1, 1280]   first 3 values: [0.0348, 0.0095, 0.1464, ...]
```

The 650M model has seen vastly more protein data than the smaller student. Its CLS embedding captures rich evolutionary and structural context — we "distil" this knowledge into the smaller student at training time.

---

## Step 5 — Student Protein Encoder (ESM-2 35M)

| Property | Value |
|---|---|
| Model | ESM-2 35M (facebook/esm2_t12_35M_UR50D) |
| Layers | 12 transformer encoder layers |
| Hidden dim | 480 per token |
| Frozen layers | 0–9 (10 layers) |
| Fine-tuned layers | 10–11 (last 2 layers) |

**Processing:**
```
[input_ids, attention_mask] → ESM-2 35M → last_hidden_state [1, 1024, 480]
                                        → Linear(480 → 512)
                                        → protein_seq [1, 1024, 512]   ← all positions
                                        → student_cls [1, 512]          ← CLS token only
```

---

## Step 6 — Teacher-Student Fusion

**Why blend teacher + student?**  
The student (35M) is fast and fine-tunable. The teacher (650M) is rich but expensive. Blending their CLS embeddings at λ=0.5 gives the cross-attention access to 650M-level protein understanding at 35M inference cost.

```
s = align_student(student_cls)   Linear(512 → 512)  →  [1, 512]
t = align_teacher(teacher_cls)   Linear(1280 → 512) →  [1, 512]

protein_fused = λ × s + (1 − λ) × t
              = 0.5 × s + 0.5 × t     →  [1, 512]
```

`protein_fused` is then **injected at position 0** (the CLS slot) of `protein_seq`, replacing the raw student CLS:

```
protein_seq_fused [1, 1024, 512]  ← same as protein_seq but pos-0 = protein_fused
```

> After alignment, cosine similarity between s and t = **0.999998** — the student has learned to closely mimic the teacher (L_distill = 0.000002).

---

## Step 7 — Bidirectional Cross-Attention

Two separate attention heads let drug and protein query each other:

### D→P — Drug attends to Protein

*"Which protein positions are most relevant to this drug?"*

```
Query = drug_global        [1,    1, 512]
Key   = protein_seq_fused  [1, 1024, 512]
Value = protein_seq_fused  [1, 1024, 512]
                ↓
Output: d2p_out [1, 512]   ← drug embedding enriched by protein context
```

**Top-5 protein positions the drug focused on:**

| Rank | Position | Attention Weight |
|---|---|---|
| 1 | 173 | 24.7% |
| 2 | 153 | 18.1% |
| 3 | 171 | 7.6% |
| 4 | 39 | 7.4% |
| 5 | 168 | 6.7% |

### P→D — Protein attends to Drug Atoms

*"Which drug atoms does each protein position care about?"*

```
Query = protein_seq_fused  [1, 1024, 512]
Key   = drug_nodes          [1,   21, 512]
Value = drug_nodes          [1,   21, 512]
                ↓
p2d_out [1, 1024, 512]  → masked mean pool (ignore padding) → p2d_global [1, 512]
```

Max attention weight across drug atoms = **0.594** (one atom strongly dominates).

---

## Step 8 — MLP Predictor

Three 512-dim vectors are concatenated and fed through a 3-layer MLP:

```
drug_global [512]  ⊕  d2p_out [512]  ⊕  p2d_global [512]  →  combined [1536]
```

| Layer | Input | Output | Activation | Note |
|---|---|---|---|---|
| Linear | 1536 | 1536 | GELU | mean → −0.936 (before act) |
| Dropout(0.3) | 1536 | 1536 | — | no-op in eval mode |
| Linear | 1536 | 512 | GELU | mean → −1.870 (before act) |
| Dropout(0.3) | 512 | 512 | — | no-op in eval mode |
| Linear | 512 | 1 | — | scalar output = **2.4837** |

---

## Step 9 — Loss Function

The total loss combines three components:

```
total_loss = w_reg × L_reg  +  (w_rank × β) × L_rank  +  α × L_distill
```

### Regression Loss — SmoothL1 (Huber)

```
|pred − target| = |2.4837 − 2.4932| = 0.0095  < 1.0

L_reg = 0.5 × 0.0095²  =  0.000045     (weight = 1.0)
```

### Ranking Loss — Pairwise Hinge

```
L_rank = mean( max(0, margin − (pred_j − pred_i)) )
       for all pairs (i,j) where true_j > true_i + 0.05

= 0.0   (only 1 sample → no pairs to compare)        (weight = 0.5 × β=0.22 = 0.110)
```

### Distillation Loss — Cosine Similarity

```
L_distill = 1 − cos(student_cls_aligned, teacher_cls_aligned)
          = 1 − 0.999998
          = 0.000002                                  (weight = α=0.03)
```

### Total

```
total_loss = 1.0 × 0.000045  +  0.110 × 0.0  +  0.03 × 0.000002
           = 0.000045
```

---

## Final Result

| Metric | log1p scale | Raw KIBA |
|---|---|---|
| **Predicted** | 2.4837 | **10.986** |
| **Ground Truth** | 2.4932 | **11.100** |
| **Absolute Error** | 0.0095 | **0.114** |

The model predicts the binding affinity with an error of only **0.114 raw KIBA units** on this test sample.

---

## Model Architecture Summary

```
HypergraphDrugEncoder
  └─ Linear(49 → 512)
  └─ HypergraphConv × 3  (residual + LayerNorm + GELU each)
  └─ GlobalMeanPool  →  drug_global [512]

StudentProteinEncoder  (ESM-2 35M, layers 10-11 fine-tuned)
  └─ ESM-2 35M  →  last_hidden_state [1024, 480]
  └─ Linear(480 → 512)  →  protein_seq [1024, 512], student_cls [512]

TeacherStudentFusion  (λ=0.5)
  └─ align_student: Linear(512 → 512)
  └─ align_teacher: Linear(1280 → 512)
  └─ fused = λ·s + (1-λ)·t  →  injected at CLS position

CrossAttention (bidirectional)
  └─ D→P: drug_global queries protein_seq  →  d2p_out [512]
  └─ P→D: protein_seq queries drug_nodes  →  p2d_global [512]

MLPPredictor
  └─ Linear(1536→1536) → GELU → Dropout(0.3)
  └─ Linear(1536→512)  → GELU → Dropout(0.3)
  └─ Linear(512→1)     → scalar KIBA prediction
```

---

*Pipeline trace captured on test row 0 of KIBA dataset.*
