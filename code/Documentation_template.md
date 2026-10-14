# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** The Epoch Warriors  
**Team Members:** Rajveer Gupta, Manas Tiwari, Hrutuparna Bedekar, Sutikshan Upman  
**Submission Date:** 29 September 2026

---

## 1. Summary

For every business in the reference table (Source 1, S1), we find all records of the same real-world business in two other sources (S2 and S3). Our pipeline first shrinks the search space: rule-based cleaning, then two complementary retrievers (shared keywords and a multilingual embedding model), then a learned re-ranker. That leaves 5.58 candidates per S1 on validation. A stack of models then decides which candidates are true matches: gradient-boosted trees over hand-built similarity features, a multilingual cross-encoder, and a small multilingual LLM reranker used only on uncertain pairs. A final step enforces that each record belongs to at most one business, and keeps a match only when the model is confident.

| headline result (final submission) | value |
|---|---|
| validation macro F0.5 (100,000 held-out train S1, test-like distribution) | **0.99158** |
| macro precision | 0.99803 |
| macro recall | 0.97776 |
| F0.5 on singletons (S1 with no true match) | 0.99517 |
| public leaderboard F0.5 | **0.985978** |

*The public leaderboard value is from our submission history.*

**Three key ideas**

1. **Train on a test-like distribution.** The test set contains far more unmatched records than train. We remove 20 % of train S1 ("ghost S1") so that their records become unmatched noise, and we fit every model and threshold on that distribution.
2. **One record, one business.** A record can match at most one S1. At every stage, each candidate pair therefore also sees how strongly *other* S1 claim the same record, and the final assignment gives each record to at most one S1.
3. **A multilingual LLM reranker where the other models are unsure.** `Qwen/Qwen3-Reranker-0.6B`, fine-tuned on our training pairs, scores only uncertain pairs. It gave the largest leaderboard step: 0.9834 → 0.985619.

**Version mapping.** Internally the runs were called Test 1, Test 2, Test 3, Stage A and Stage B. In this document:

| document name | internal run | notebook / driver |
|---|---|---|
| v1 | first pipeline | `MLChallengeV1.ipynb` |
| v2 | v2 base run | `MLChallenge_v2.ipynb`, `run_all.py` |
| v3a | Test 1 | `run_test1.py` |
| v3b | Test 2 | `run_test2.py` |
| v4a | Test 3 | `qwen_ce` + `collective` + `write_final` |
| **v4b (final)** | **Stage A** | same, with the wider Qwen band |
| not selected | Stage B | `chain_final.sh` |

v3 is v3a + v3b. v4 is v4a + v4b.

---

## 2. Methodology

### 2.1 Problem Analysis

**Task.**
- **S1** is the reference business table: 1,732,544 test S1 across France, India and the US.
- **S2 and S3** are two further sources of business records: 9,969,589 test records.
- For each S1 we output the list of S2/S3 ids that describe the same business, possibly empty.
- A **distractor** is an S2/S3 record that matches no S1.

**Metric.**
- For each S1 we compute F0.5 between the predicted and the true set. The macro average over all S1 is the score.
- F0.5 weighs precision about twice as much as recall.
- An S1 with no true match (a **singleton**) scores 1 only if we predict an empty list, and 0 otherwise. Singletons are 5.58 % of train S1, so one wrong record on a singleton costs a full point for that S1.

**Structure.**
- In train, every S2/S3 record belongs to at most one S1 (verified on the ground truth).
- We never propose a match across countries: every blocking key and every dense search is restricted to one country.
- Train S1 have on average 3.46 true matches (at most 11).

**Noise types we observed.**

| noise | example / evidence |
|---|---|
| Script and transliteration | India S2 names in Indic scripts: 23.51 % (train), 23.64 % (test) |
| Legal suffixes and honorifics | "Pvt Ltd" vs "Private Limited" vs "प्राइवेट लिमिटेड", "M/s", "Smt" |
| Domain-style names, phone numbers, junk | "b7n.com" for "B 7 N Churchill" |
| Address abbreviations and reordering | "Road/Rd", "Rue/R.", components in a different order |
| House-number changes | zero-padded or ranged ("006/1", "2805-2807") |
| Missing addresses | 2.28 %–3.68 % of S2/S3 records; S1 addresses are always present |
| Shared names | 29.5 %–44.2 % of S1 share their lower-cased raw name with another S1 of the same country (train and test) |
| Unseen country | France has no training labels |

### 2.2 Solution Strategy

**Approach type:** a hybrid pipeline. Rule-based normalisation, then multi-source retrieval, then a learned two-pass re-ranker, then feature-based gradient boosting stacked with transformer scores, then global exclusive assignment.

```
 S1 + S2/S3 records (per country)
          |
 [1] Normalisation (rules; anyascii transliteration)
          |
          +--> [2a] Sparse keys: 11 families, IDF-weighted, top-50 per S1
          +--> [2b] Dense bi-encoder multilingual-e5-small (118M):
          |         S1->record top-40, record->S1 top-3
          v
 [3] Union  .............................................  ~86.7 candidates/S1 (val)
          |
 [4] Two-pass LightGBM re-ranker + cut  .................   5.58 candidates/S1 (val)
          |                                                  6.12 candidates/S1 (test)
          |                                                  = candidate_pairs.tsv
 [5] 107 pair features -> level-1 LightGBM -> p1 (+ 10 competition features)
 [6] Cross-encoder multilingual-e5-base (278M) -> logit (+ 9 derived features)
 [7] LightGBM stacker (126 features) -> p
 [8] Qwen3-Reranker-0.6B (596M) on uncertain pairs (v3a stacker p in [0.02, 0.98])
 [9] Collective LightGBM (27 features) -> final p
          |
 [10] Exclusive assignment + p >= 0.70  .................  ~3.38 matches/S1 (test)
          = matching_results.tsv
```

**Core innovations.**
1. **Ghost-S1 simulation of the test distractor rate.** Every model and threshold is fitted on it (§2.2.1).
2. **Complementary retrieval.** A dense bi-encoder that reads raw scripts, alongside IDF-weighted keys. Together they reach 0.9986 pair recall on validation.
3. **Record-side competition features at every learned stage** (re-ranker, level-1, stacker, collective pass), plus exclusive assignment at the end.
4. **An LLM reranker applied only to uncertain pairs,** entering the final model as one feature.
5. **Precision-first decisions.** One global threshold chosen for the F0.5 metric; we abstain on undecidable pairs.

#### 2.2.1 EDA Findings and How We Used Them

| Finding | Evidence (number, train/test) | What we did about it |
|---|---|---|
| Test has more distractors | Train: 26.0 % of S2/S3 records match no S1. Test: ≈ 39.8 % estimated from record/S1 ratios (5.531–5.824 in test vs 4.674–4.68 in train), if test keeps 3.46 true matches per S1. | **Ghost-S1 removal.** Train S1 are stratified by country × min(#true matches, 6). Within each stratum, the 20 % with the lowest random rank are removed from the S1 table. Their S2/S3 records stay as unlabelled distractors. The resulting ghost train has 1,765,456 S1 and 40.79 % distractors. The model subsets (`rr_train`, `rr_tune`, `val`, `stack_train`, `ce_train`, `bienc`, `rest`) are disjoint and drawn with the same stratification, so every model and threshold is fitted on a test-like distribution. |
| France is unseen in training | 259,452 of 1,732,544 test S1 (14.98 %); no France rows in train | Country is an open set (no per-country models or thresholds). French legal forms and street terms are in the normalisation. ZIP/state evidence is blanked for 15 % of train S1 so that France-like records (no ZIP, no known state) are in-distribution. We use multilingual encoders and the Qwen reranker. |
| Indic scripts | India S2 names in Indic script: 23.51 % (train), 23.64 % (test) | anyascii transliteration for string features; the raw script is kept for the multilingual transformers; a phonetic consonant skeleton survives transliteration differences. |
| Missing addresses | 2.28 %–3.68 % of S2/S3 records, both splits | Missing-indicator features (address features set to −1). Sibling features compare an address-less record with the S1's confident records. |
| Shared names across S1 | 29.5 %–44.2 % of S1 share their lower-cased raw name within the country | Multiplicity features: how many S1 and how many records share this name, skeleton or address. |
| Each record belongs to ≤ 1 S1 | Verified on train ground truth | Record-side competition features, and exclusive assignment of each record to at most one S1. |
| Singletons 5.58 %; F0.5 break-even | Adding one uncertain record pays off only above 0.632–0.743 precision (n = 2–5, §4.4) | A single global, precision-first threshold (0.70) after exclusive assignment. |

The first row's numbers come from executed notebook cells: train and test counts in `MLChallenge_v4(Final).ipynb` cells 5–13, ghost statistics in `MLChallenge_v2.ipynb` cell 6.

**Use of test data.**
- No test labels exist, and none were used.
- Test inputs were used for two things only:
  - unsupervised inference-time statistics: IDF weights, name/address multiplicity counts, and the candidate-competition graph (each model's predictions among the S1 that compete for a test record);
  - the descriptive shift analysis above, which set the ghost fraction.
- No test record was labelled by hand. No pseudo-labels were created. Every model and every vectoriser (including the TF-IDF vocabulary) is fitted on train data only.
- No external data, APIs or lookup services are used, apart from the public pretrained weights of the three transformer models in §4.1.
- The normalisation dictionaries (legal forms, honorifics, street abbreviations, state names) are generic linguistic and postal conventions that we wrote.
- Leaderboard scores were used to compare submitted runs. No leaderboard feedback enters any model's training data.

---

## 3. Candidate Generation (Blocking)

### 3.1 Current Approach (in brief)

Two retrievers run within each country:
- IDF-weighted **sparse keys** keep the 50 best S1→record pairs per S1.
- A fine-tuned multilingual **dense bi-encoder** keeps the top 40 records per S1 and the top 3 S1 per record.

Their union covers 99.86 % of true pairs. A two-pass LightGBM re-ranker then cuts the union to 5.58 candidates per S1 while keeping 99.43 % of true pairs.

Two terms used in the table below:
- **Pair recall** is the share of true (S1, record) pairs that are in the candidate set.
- **Oracle F0.5** is the macro F0.5 that a perfect matcher would reach when restricted to these candidates: the ceiling that blocking leaves for the matcher.

| candidate set (val, 100,000 S1) | cand/S1 | pair recall | oracle F0.5 |
|---|---|---|---|
| sparse keys, top-50 | 49.771 | 0.9575 | 0.98418 |
| dense bi-encoder, top-40 | 40.0 | 0.9977 | 0.99932 |
| union (input to the re-ranker) | 86.711 | 0.9986 | 0.99959 |
| **two-pass re-ranker cut (final)** | **5.58** | **0.9943** | **0.99846** |

*The first three rows are executed output (`MLChallenge_v2.ipynb` cell 8).*

On test, the final candidate file holds 10,594,837 pairs (6.1152 per S1); 5,354 S1 have no candidate.

### 3.2 Details

**Sparse keys.**
- Every record emits hashed keys of the form *country | family | value*.
- A key with more than 300 postings among the country's S2+S3 records is dropped as uninformative.
- Each remaining key has weight idf = ln((n_r + 1) / (df + 1)), where n_r is the country's record count and df the key's posting count.
- A pair's score is the sum of the IDF weights of its shared keys. We keep the top 50 per S1.

| family | key content |
|---|---|
| t | name tokens ≥ 3 characters (first 6) |
| n | sorted name-token pairs (adjacent and skip-one) |
| q | whole squashed name (≥ 4 characters) |
| p | 6-character prefix of the squashed name |
| a | alphabetic address tokens ≥ 4 characters (first 8) |
| b | address token bigrams (first 10) |
| z | ZIP / PIN code |
| m | address numbers with ≥ 3 digits, leading zeros removed (first 4) |
| k | phonetic consonant skeleton of the name |
| c | name token (or squashed name) × locality token |
| h | first address number × street or locality token |

**Dense bi-encoder.**
- **Model:** `intfloat/multilingual-e5-small`.
- **Input:** "name | address" in the raw script.
- **Fine-tuning:**
  - data: true pairs of the `bienc` subset (300,000 S1, at most 1,000,000 pairs);
  - loss: symmetric in-batch contrastive (InfoNCE, temperature 0.05);
  - batches: 1,024 pairs drawn from one country, so in-batch negatives are same-country businesses;
  - schedule: one epoch.
- **Search:** exact cosine top-k within country, in both directions: S1→record 40 and record→S1 3.

**Pass-1 re-ranker (v2 model).**
- A LightGBM on 39 cheap features: blocking evidence, dense cosine and ranks in both directions, fast string similarities, and record-side blocking competition.
- Fitted on `rr_train` (100,000 S1) and early-stopped on `rr_tune` (100,000 S1).

**Pass-2 re-ranker.** It adds 13 hints computed from the pass-1 score q1:
- **record-side competition:** the rank and gap of this S1 among *all* S1 claiming the record, and the number of claimants with q1 ≥ 0.5;
- **sibling evidence:** how many of the S1's confident records (q1 ≥ 0.9 and top claimant) share the record's name or skeleton, and the best name similarity to them;
- **acronym match** between a name and the other side's initials;
- **a no-address flag.**

Pass 2 is fitted on 200,000 S1 from the `rest` subset (not used by any other model) and early-stopped on 50,000 other `rest` S1.

**Cut rule.** A candidate (S1, record) with pass-2 probability p2 and rank r within its S1 is kept if any of these holds:

```
(r ≤ kmax and p2 ≥ floor)  or  p2 ≥ p_hi
   or  [grid option R]  S1 among the record's top-R claimants and p2 ≥ floor
   or  [grid option S]  record has no address, shares a confident sibling's name, p2 ≥ 0.001
   or  acronym match and p2 ≥ 0.001          (always on)
```

- The grid searched kmax ∈ {7, 8, 10, 12}, floor ∈ {0.003, 0.005, 0.01, 0.02}, p_hi ∈ {0.15, off}, R ∈ {0, 1, 2}, S ∈ {off, on}.
- On `rr_tune` we took the best oracle F0.5 under a budget of 6.0 candidates per S1, then the smallest set within 0.0003 of that best.
- The selected rule is kmax = 8, floor = 0.005, p_hi = 0.15, with the always-on acronym clause.

### 3.3 Version History of Blocking

| Version | Candidate sources | Re-ranker / cut rule | cand/S1 | pair recall | oracle F0.5 | Key change vs previous |
|---|---|---|---|---|---|---|
| v1 | 8 sparse families (t, n, q, p, a, b, z, m), top-50 | Logistic regression; rank ≤ 10 and p ≥ 0.05 | 4.79 | 0.9289 | 0.9715 | – |
| v2 | 11 sparse families ∪ dense e5-small bi-encoder | 39-feature LightGBM; rank ≤ 8 and p ≥ 0.01 | 5.353 | 0.9918 | 0.99792 | Dense retrieval (raw script) plus 3 new key families (skeleton, name × locality, number × street) |
| v3 | same union | Two-pass LightGBM with record-side, sibling and acronym hints; rule of §3.2 | 5.58 | 0.9943 | 0.99846 | Recovers 854 of the 2,346 true pairs the v2 cut dropped |
| v4 | same as v3 | same as v3 | 5.58 | 0.9943 | 0.99846 | No change |

The v1 row is executed output of `MLChallengeV1.ipynb` cell 50 on its own 100,000-S1 validation split, drawn from the original (not ghost) train distribution. The v2 row is `MLChallenge_v2.ipynb` cell 8.

---

## 4. Matching Model (Model Architecture and Feature Engineering)

### 4.1 Architecture Table

| Component | Exact model | Licence | Parameters | Inputs (feature count) | Training data | Applied to | val macro F0.5 |
|---|---|---|---|---|---|---|---|
| Dense bi-encoder (blocking) | `intfloat/multilingual-e5-small` | MIT | 118M | raw "name \| address" text | true pairs of `bienc` (300,000 S1, ≤ 1M pairs) | all S1 and records (retrieval) | – (dense top-40 pair recall 0.9977) |
| Level-1 | LightGBM | MIT | tree ensemble (num_leaves 127, ≤ 1,500 rounds) | 107 pair features | all ghost-train candidate pairs except S1 of `rr_train` / `bienc`; 3-fold cross-fit | every candidate pair | 0.98466 (standalone) |
| Cross-encoder | `intfloat/multilingual-e5-base` | MIT | 278M | both records as raw "name \| address" text | `ce_train` candidates: 2,137,708 pairs, positive rate 0.642 | every candidate pair | 0.98545 (standalone) |
| Stacker | LightGBM | MIT | tree ensemble (num_leaves 127, ≤ 4,000 rounds) | 126 (107 + 10 competition + 9 CE-derived) | `stack_train` (200,000 S1), 5-fold OOF | every candidate pair | 0.99093 (cumulative) |
| + Collective pass | LightGBM | MIT | tree ensemble (num_leaves 63, ≤ 3,000 rounds) | 26 without Qwen (stacker p, p1, CE logit + 23 collective features) | `stack_train`, 5-fold OOF | every candidate pair | 0.99098 (cumulative) |
| + LLM reranker | `Qwen/Qwen3-Reranker-0.6B` | Apache-2.0 | 596M | prompt with name \| address \| country of both records | 400,000 pairs of `ce_train` S1 outside the evaluation set | pairs with v3a stacker p in [0.05, 0.95] (v4a) | 0.99153 (cumulative, v4a) |
| **Final run (v4b)** | same components; collective with 27 features | – | – | Qwen score on the wider band | same | pairs with v3a stacker p in [0.02, 0.98] | **0.99158** |
| Oracle on our candidates | perfect matcher | – | – | – | – | – | 0.99846 |

*v2's executed values are in Appendix B.*

All models are MIT or Apache-2.0 licensed. The largest has 0.6B parameters (limit 8B).

### 4.2 Feature Engineering

Every feature is computed the same way on train and on test, from that split's own records. **No tabular feature uses the country label.** Country only scopes the blocking keys, the dense search and the frequency counts, so France is scored by exactly the same functions as India and the US. The Qwen prompt is the one exception: it includes the country name as text.

Features are built in layers:
1. String and address similarities are computed on normalised fields (§4.2.1).
2. Each later model adds features derived from the previous model's scores over the whole candidate graph (§4.2.4).

#### 4.2.1 Normalised fields (inputs to every string feature)

| field | how it is built (`normalize.py`) |
|---|---|
| `name_norm` | Non-ASCII text is NFKC-normalised and transliterated with anyascii, then lower-cased. Phone numbers and long digit runs are removed, and so is "M/s". Dotted legal forms are collapsed (L.L.C. → llc, S.A.R.L. → sarl). "&" and "+" become "and". Domain names are reduced to their stem ("b7n.com" → "b7n"). Legal words are canonicalised, including transliterated Indic spellings (praivet → pvt, limitedd → ltd, kampani → co). |
| `legal` | The sorted set of legal-form tokens found in the name (pvt, ltd, inc, llc, sarl, sas, gmbh, …) |
| `name_core` | `name_norm` without legal forms and honorifics (smt, sri, shri, mr, dr, messrs, the, …) |
| `name_sq` | `name_core` with spaces removed |
| `name_sk` | Phonetic consonant skeleton. Rewrite rules (ph→f, th→t, sh→s, ch→k, c→k/s, z→s, j→s, tion→shn, …); each token keeps its first letter and drops vowels and y/w/h/v; repeated letters are collapsed and tokens sorted. |
| `addr_norm` | Lower-cased; PO boxes and "#" removed. US and Indian state names become codes, per comma-separated component. Ordinals are stripped (1st → 1), digits are split from the letters that follow, and leading zeros are removed (006 → 6). US, Indian and French address words are canonicalised (street → st, nagar → ngr, taluka → tq, chemin → chem, …). Filler words (no, hno, door, bis, …) are dropped. |
| `loc` | Up to 6 locality tokens from the digit-free address components: ≥ 3 characters, not street words |
| `addr_nums` | All numbers in the address (runs of 8+ digits removed), leading zeros removed |
| `zip`, `state` | The last 5–6-digit number, and the last component that is a US/Indian state code |
| flags | `is_domain`, `name_indic` / `addr_indic` (Indic script in the raw text), `has_addr` |

Transliteration and the skeleton make native-script and Latin spellings meet. On a real training pair, "Anand Foods Private Limited" becomes `name_core` "anand foods" with skeleton "and fds". Its Bengali record "আনন্দ ফুডস প্রাইভেট লিমিটেড" becomes "annd phuds", also with skeleton "and fds" (executed, `MLChallenge_v4(Final).ipynb` cell 17).

#### 4.2.2 Feature groups

| group | count | what it measures |
|---|---|---|
| Stage-2 cheap features | 39 | Blocking evidence (IDF sum, key count, rank, per-family IDF); dense cosine and ranks both ways; fast name/address similarities; ZIP and house-number agreement; locality Jaccard; skeleton equality; record-side blocking competition |
| Stage-2 hints | 13 | Pass-1 score; record-side rank, gap and claimant count; sibling equality and similarity; acronym match; no-address flag |
| Stage-2 outputs | 2 | Pass-2 probability and its rank within the S1 |
| Full string features | 43 | Name token-sort / partial / Levenshtein / prefix / suffix / token-set similarity; character 2–4-gram TF-IDF cosine; Jaccard; lengths; legal-form agreement and conflict; domain and script flags; address similarities and 3-gram TF-IDF; number, state and presence features; exact raw-string equality |
| Multiplicity | 10 | How many S1 and how many records in the country share this pair's squashed name, skeleton or address, on each side |
| **Pair features (level-1 input)** | **107** | 39 + 13 + 2 + 43 + 10 |
| Level-1 competition | 10 | From level-1 p1 (listed below) |
| Cross-encoder derived | 9 | CE logit; its rank, gap, count > 0 and maximum within the S1; its rank, gap, best rival and count > 0 among the record's claimants |
| **Stacker input** | **126** | 107 + 10 + 9 |
| **Collective input (final run)** | **27** | Stacker p, p1, CE logit, 23 collective features, Qwen score |

The v2 pipeline used the same groups without the 13 stage-2 hints. That gave 94 pair features and a 109-feature stacker, with 5 CE features and no record-side CE features.

#### 4.2.3 The 107 pair features in detail

**Stage-2 cheap features (39)** (`stage2.py`)

| feature(s) | definition |
|---|---|
| `bscore`, `bkeys`, `brank` | log(1 + sum of IDF over shared blocking keys); number of shared keys; sparse rank within the S1 (99 if not retrieved by sparse) |
| `b_t` … `b_h` (11) | IDF sum per key family (§3.2) |
| `b_rel`, `s_n` | sparse score relative to the S1's best candidate; number of union candidates of the S1 |
| `r_n`, `r_rel`, `r_best`, `r_gap2` | record side of blocking: how many S1 lists contain the record; score relative to the record's best claimant; whether this S1 is that best claimant; score minus the record's second-best |
| `dcos`, `drank_s`, `drank_r` | bi-encoder cosine; rank of the record for the S1 and of the S1 for the record (99 if absent) |
| `d_gap_s`, `d_gap_r`, `r_dn` | cosine minus the S1's best; record-side dense margin (lead over the runner-up if best, deficit to the best otherwise); number of union S1 containing the record |
| `in_sparse`, `in_dense`, `src3` | which retriever found the pair; record comes from S3 |
| `n_ratio`, `n_tset`, `n_jw` | ratio and token-set similarity of `name_core`; Jaro–Winkler of `name_sq` |
| `n_sk_eq` | skeletons are equal |
| `a_tset`, `a_miss` | address token-set similarity; either address missing |
| `zip_eq`, `zip_miss` | ZIP equality; either ZIP missing |
| `loc_jacc`, `num1_eq` | locality-token Jaccard; first address number equal |

**Stage-2 hints (13)** (`stage2_v3.py`), computed from the pass-1 score q1

| feature(s) | definition |
|---|---|
| `q1`, `q1_rank_s`, `q1_gap_s` | pass-1 score; its rank and gap within the S1 |
| `q1r_rank`, `q1r_gap`, `q1r_n05` | rank and gap of this S1 among all S1 claiming the record; number of claimants with q1 ≥ 0.5 |
| `sib_n` | number of the S1's other confident records (q1 ≥ 0.9 and the S1 is the record's top claimant) |
| `sib_eq_core`, `sib_eq_sk` | how many of them have the same `name_core` / skeleton as this record |
| `sib_tset` | best token-set name similarity to them (computed for the S1's top 20 pass-1 candidates; −1 otherwise) |
| `acr_eq`, `acr_pre` | one side's squashed name (2–6 letters) equals, or is a prefix of, the other side's initials |
| `r_noaddr` | the record has no address |

**Stage-2 outputs (2):** `p_rr` (pass-2 probability) and `rr_rank` (its rank within the S1).

**Full string features (43)** (`features.py`)

| feature(s) | definition |
|---|---|
| `n_tsort`, `n_partial` | token-sort and partial-ratio similarity of `name_core` |
| `n_lev`, `n_prefix`, `n_postfix` | normalised Levenshtein, common-prefix and common-suffix similarity of `name_sq` |
| `n_full_tset` | token-set similarity of `name_norm` (legal words kept) |
| `n_tfidf` | cosine of character 2–4-gram TF-IDF vectors of `name_core` |
| `n_jacc` | name-token Jaccard |
| `n_len1`, `n_len2`, `n_ntok1`, `n_ntok2`, `n_len_diff` | name lengths and token counts |
| `legal_eq`, `legal_conf`, `legal1`, `legal2` | same legal form; both have one but they differ; presence on each side |
| `dom1`, `dom2`, `indic1`, `indic2` | domain-style name; Indic-script name, on each side |
| `a_tsort`, `a_ratio`, `a_partial` | address token-sort, ratio and partial-ratio similarity |
| `a_tfidf` | cosine of character 3-gram TF-IDF vectors of `addr_norm` |
| `a_jacc` | address-token Jaccard |
| `num_inter`, `num1`, `num2`, `num_jacc`, `num_first_eq` | shared address numbers; number counts; number Jaccard; first number equal |
| `state_eq`, `state_miss` | same state code; either state missing |
| `a_len1`, `a_len2`, `has_a1`, `has_a2`, `aindic1`, `aindic2` | address lengths, presence and Indic script on each side |
| `n_raw_eq`, `a_raw_eq` | lower-cased raw name / raw address identical |
| `p_rel`, `n_cand` | `p_rr` relative to the S1's best candidate; number of candidates of the S1 |

Both TF-IDF models are hashed character n-grams (2^20 buckets, sublinear term frequency). The IDF weights are fitted on 400,000 sampled train records from each source; no test text is used.

**Multiplicity (10)** (`features.py`), counted within the country

| feature(s) | definition |
|---|---|
| `m_s1_sq`, `m_r_sq` | how many S1 / how many S2+S3 records have the same squashed name as this S1 |
| `m_s1_sk`, `m_r_sk` | the same for the S1's skeleton |
| `m_s1_addr`, `m_r_addr` | the same for the S1's normalised address |
| `mr_s1_sq`, `mr_r_sq`, `mr_s1_addr`, `mr_r_addr` | the same counts for the record's squashed name and address |

A common name at many addresses is weak evidence; a unique name is strong evidence. These counts need no labels.

#### 4.2.4 Features derived from model scores

All are computed over the complete candidate graph of a split: every S1 and every record it proposes. For a pair score x, the same definitions are reused:

| definition | meaning |
|---|---|
| rank within the S1 (`*_rank_s`) | position of the pair among the S1's candidates |
| gap within the S1 (`*_gap_s`) | x minus the S1's runner-up if the pair is the S1's best, otherwise x minus the S1's best |
| rank among claimants (`*_rank_r`) | position of this S1 among all S1 whose candidates include the record |
| gap among claimants (`*_gap_r`) | the same gap, computed over the record's claimants |
| best rival (`*_max_other_r`, `riv_p`) | the highest score of any other S1 claiming the record |
| claimant counts (`n_claim_r`, `*_n05_r`) | number of claimants, and number with a score above 0.5 |

| group | count | features |
|---|---|---|
| Level-1 competition (from p1) | 10 | `p1`, `p1_rank_s`, `p1_gap_s`, `p1_n05_s`, `p1_sum_s`, `p1_max_other_r`, `p1_rank_r`, `p1_n05_r`, `n_claim_r`, `p1_best_r` |
| Cross-encoder derived (from the CE logit) | 9 | `ce_e5b` (the logit), `ce_rank_s`, `ce_gap_s`, `ce_n0_s` (candidates with logit > 0), `ce_max_s`, `ce_rank_r`, `ce_gap_r`, `ce_max_other_r`, `ce_n0_r` |
| Collective (from stacker p) | 23 | `p_rank_s`, `p_gap_s`, `p_n05_s`, `p_rank_r`, `p_gap_r`, `p_n05_r`, `n_claim_r`, `r_noaddr`, sibling features and rival features (below) |

The cross-encoder itself sees only the two records' text. Its within-S1 and record-side features are computed from its logits, and the stacker uses them.

**Sibling evidence in the collective pass.** A "confident" record of an S1 has stacker p ≥ 0.95 and is the record's top claimant. For each pair, the collective pass computes the following against the S1's other confident records:
- `sib_n`: how many there are;
- `sib_eq_core`, `sib_eq_sk`, `sib_eq_raw`: how many share this record's normalised name, skeleton or lower-cased raw name;
- `sib_tset`, `sib_addr`: the best name and address token-set similarity to them.

The same five values are computed for the record's best rival S1 (`riv_sib_*`), together with own − rival differences (`d_sib_eq_core`, `d_sib_tset`, `d_sib_addr`). This lets an address-less record be attached to the S1 whose confident records it resembles, rather than to a same-name rival.

**Final collective input (27):** the stacker probability `p`, `p1`, `ce_e5b`, the 23 collective features, and `qw`, the Qwen score on uncertain pairs.

#### 4.2.5 Missing values and augmentation

| situation | encoding |
|---|---|
| Either side lacks an address, ZIP, state or numbers | the corresponding similarities are set to −1, and explicit flags are set (`a_miss`, `zip_miss`, `state_miss`, `has_a1`/`has_a2`, `r_noaddr`) |
| Pair found by only one retriever | sparse rank or dense rank set to 99, and sparse score 0. The dense cosine of sparse-only pairs is computed from the stored embeddings, so it is never missing. |
| No confident sibling | sibling similarities −1, counts 0 |
| Pair outside the Qwen band | `qw` left missing (LightGBM's native missing-value handling) |

Values are never imputed as matches.

**Missing-field augmentation.** For a deterministic 15 % of train S1 (hashed on the S1 id), the ZIP and state evidence is blanked in the training features: ZIP IDF 0, agreement −1, missing flag 1. France has no ZIP and no state dictionary, so this makes France-like pairs look familiar instead of looking like weak pairs. The augmentation is applied to train features only.

#### 4.2.6 Text inputs of the transformer models

| model | text of each record | max length |
|---|---|---|
| Bi-encoder (e5-small) | "query: name \| address" in the raw script (name only if no address) | 64 tokens |
| Cross-encoder (e5-base) | the two records as a sentence pair, each "name \| address" in the raw script | 128 tokens |
| Qwen3-Reranker | "name \| address (or 'no address') \| country" for Query and Document, inside the reranker's yes/no prompt | 96 content tokens |

The transformers read the raw, untransliterated text. Their scores are therefore complementary to the normalised-string features.

#### 4.2.7 Which features the models rely on

**v2 stacker (executed, `MLChallenge_v2.ipynb` cell 12), share of LightGBM gain:**

| feature | gain share |
|---|---|
| `p1` | 0.509 |
| `ce_e5b` | 0.403 |
| `p1_n05_r` | 0.039 |
| `p1_max_other_r` | 0.013 |
| `ce_gap_s_e5b` | 0.008 |

**Final pipeline, top features per model:**

| model | top features by gain share |
|---|---|
| stage-2 pass 1 | `d_gap_r` 0.724, `drank_r` 0.082, `num1_eq` 0.051 |
| stage-2 pass 2 | `q1r_gap` 0.532, `q1` 0.410 |
| level-1 | `p_rr` 0.509, `p_rel` 0.164, `q1r_gap` 0.162 |
| stacker | `p1` 0.656, `ce_e5b` 0.233, `p1_n05_r` 0.041 |
| collective | `p` 0.721, `p_gap_r` 0.202, `p1` 0.061, `qw` 0.005 |

Record-side competition features (`d_gap_r`, `q1r_gap`, `p1_n05_r`, `p_gap_r`) rank near the top at every stage. This is the practical effect of the rule that each record belongs to at most one business.

### 4.3 Model Training Details

**Evaluation set and closure.**
- Stacker and collective data use the **evaluation set**: `stack_train` ∪ `val` plus their **closure**.
- The closure is every other S1 that claims one of their candidate records. With it, exclusive assignment on validation faces the same competitors it faces on test.
- **OOF** (out-of-fold) predictions are predictions for rows held out of the fold that trained the model.
- Thresholds are chosen on `stack_train` OOF. `val` (100,000 S1) is used only for reporting.

| model | cross-validation | key hyperparameters |
|---|---|---|
| Pass-1 re-ranker | fit `rr_train`, early stop on `rr_tune` | lr 0.08, 127 leaves, min 100 per leaf, ≤ 1,500 rounds (best 681) |
| Pass-2 re-ranker | fit 200,000 `rest` S1, early stop on 50,000 others | lr 0.06, 127 leaves, ≤ 2,000 rounds |
| Level-1 | 3-fold GroupKFold by S1 | lr 0.1, 127 leaves, 255 bins, min 200 per leaf, feature fraction 0.7, bagging 0.8, L2 2.0, ≤ 1,500 rounds, early stop 50 |
| Stacker | 5-fold GroupKFold by S1 on `stack_train` | lr 0.04, 127 leaves, min 200 per leaf, feature fraction 0.7, bagging 0.8, L2 2.0, ≤ 4,000 rounds, early stop 100 |
| Collective | 5-fold GroupKFold by S1 on `stack_train` | lr 0.04, 63 leaves, min 200 per leaf, feature fraction 0.8, bagging 0.8, L2 2.0, ≤ 3,000 rounds, early stop 100 |
| Cross-encoder | trained on the disjoint `ce_train` subset | BCE, 1 epoch, lr 3e-5, batch 128, max length 128, 6 % warm-up, AdamW (weight decay 0.01) |
| Bi-encoder | trained on the disjoint `bienc` subset | InfoNCE (temperature 0.05), 1 epoch, lr 5e-5, batch 1,024, max length 64, 5 % warm-up |
| Qwen reranker | trained on `ce_train` S1 outside the evaluation set | BCE, 1 epoch, lr 2e-5, batch 64, 96 content tokens, 5 % warm-up, AdamW (weight decay 0.01) |

For the tree models:
- Early stopping uses the held-out fold.
- The final model is refitted on all rows with 1.1 × the mean best iteration.
- On `stack_train` rows, the collective uses the stacker's OOF p, so no row is scored by a model that saw its label.

**Qwen reranker.**
- **Input:** we use the model's yes/no relevance prompt. The instruction reads "Is the Document the same real-world business as the Query?". Query and Document are each "name | address (or 'no address') | country".
- **Score:** logit("yes") − logit("no").
- **Training mix:** 400,000 pairs, with at most 70 % "hard" pairs (stage-2 probability in [0.02, 0.98]) and the rest sampled outside that band.
- **Training S1:** drawn from `ce_train`, excluding every S1 in the evaluation set, so validation and test scores are out-of-sample.
- **Scoring band:**
  - In the final run (v4b), Qwen scores every pair whose **v3a stacker** probability lies in [0.02, 0.98]. v4a used [0.05, 0.95]; its scores are reused.
  - Elsewhere the Qwen feature is missing, and the collective model learns to handle that. Qwen scores only uncertain pairs, which keeps its inference cost small.
  - The band is defined on the v3a stacker, not on the v3b stacker that feeds the collective pass.

### 4.4 Decision Rule

**Exclusive assignment.** Each S2/S3 record is kept only for the S1 that gives it the highest final probability. The submitted file assigns each of its 5,852,231 matched records to exactly one S1.

**Threshold.** A remaining pair is kept if p ≥ **0.70**.
- The collective step evaluates global thresholds on `stack_train` OOF over a grid from 0.55 to 0.85 in steps of 0.03. The final threshold t = 0.70 is a fixed value passed to the final assignment step. It is supported by our OOF grid search (the executed v2 selection gave 0.69; 0.70 sits on the same grid). OOF estimates may be slightly optimistic because early stopping uses the held-out fold.
- In the one selection preserved as executed notebook output (v2, `MLChallenge_v2.ipynb` cell 12), the OOF-optimal threshold on a 0.01 grid was 0.69.

**Why a high threshold.** Take an S1 with n true records of which n − 1 are already predicted. Adding one more uncertain record raises F0.5 only if that record is right with probability above the break-even value below (`MLChallenge_v4(Final).ipynb` cell 32, computed):

| true records n | 2 | 3 | 4 | 5 |
|---|---|---|---|---|
| break-even precision | 0.632 | 0.698 | 0.727 | 0.743 |

On a true singleton, a wrong record costs the full 1.0.

**Decision-rule study (v2, executed, `MLChallenge_v2.ipynb` cell 14).** All rules use exclusive assignment:

| rule | parameters | stack_train OOF F0.5 | val F0.5 | val singletons |
|---|---|---|---|---|
| A: global threshold (used) | t = 0.69 | 0.9899565 | 0.9901303 | 0.9940913 |
| B: per-country thresholds | India 0.69, US 0.73 | 0.9899704 | 0.9900524 | 0.9942704 |
| C: threshold + margin over the record's runner-up | t = 0.69, margin 0.3 | 0.9899864 | 0.9901745 | 0.9940913 |
| D: expected-F0.5 lists | – | 0.9899049 | 0.9902847 | 0.9903312 |

- The margin rule (C) was marginally better than the global threshold on both OOF (+0.00003) and validation (+0.00004).
- Expected-F lists (D) were better on validation but worse on OOF and much worse on singletons.
- Per-country thresholds were worse on validation, and cannot be tuned for France.
- We judged gains of this size too small to justify an extra parameter. We kept the global threshold for simplicity and singleton safety.

### 4.5 Version History of the Model

| Version | Architecture (models + params) | Problem observed | How we fixed it in the next version | val macro F0.5 | public LB |
|---|---|---|---|---|---|
| v1 | 8-family keys + logistic re-ranker; e5-small cross-encoder (118M); LightGBM stack | Trained on the raw train distribution (26 % distractors; test ≈ 39.8 %); blocking ceiling only 0.9715 | Ghost-S1 training; dense bi-encoder; e5-base cross-encoder; LightGBM level-1 + stacker | 0.96310 (raw distribution) | ~0.94 |
| v2 | 11-family keys ∪ e5-small bi-encoder (118M); 39-feature LightGBM re-ranker; level-1 LightGBM; e5-base CE (278M); LightGBM stacker | 2,346 true val pairs lost at the stage-2 cut (mostly address-less records with shared names, and S1 with > 8 true records); val→LB gap; France unseen | Two-pass re-ranker; level-1 on all rows; CE record-side features; collective pass | 0.99013 | 0.981874 |
| v3 | v2 + two-pass re-ranker, level-1 on all rows, CE record-side features, collective LightGBM | Acronyms, French abbreviations and region names, coined trade names: cases the e5-base cross-encoder handles poorly | Qwen3-Reranker-0.6B on uncertain pairs | 0.99098 (v3b) | 0.9834 (v3b) |
| v4 | v3 + Qwen3-Reranker-0.6B (596M) as a collective feature | Qwen coverage limited to [0.05, 0.95] (v4a) | v4b widened the band to [0.02, 0.98] | 0.99153 (v4a) / **0.99158 (v4b)** | 0.985619 (v4a) / **0.985978 (v4b)** |

*Public leaderboard values are from our submission history; v1's is approximate.*

**v1 → v2.** v1 scored 0.96310 on a validation split drawn from train, but about 0.94 on the leaderboard. The test set has more S2/S3 records per S1 (5.53–5.82 against 4.67–4.68 in train), so a model tuned on train over-predicts on test. v2 therefore trains and tunes everything on the ghost-S1 distribution, and adds a dense retriever that lifts the blocking ceiling from 0.9715 to 0.99792. The dense retriever reads the raw scripts; for example, it aligns Devanagari names with their English forms.

**v2 → v3.** The v2 cut discarded 2,346 true validation pairs. These were mostly address-less records whose name several S1 share, and records of S1 with more than 8 true matches. Such records look weak to every claimant when scored alone. v3 first scores all candidates, then gives the re-ranker and the later models each record's position among *all* its claimants, plus evidence from the S1's other confident records. This recovered 854 of the lost pairs. The leaderboard rose from 0.981874 to 0.9834 (v3b).

**v3 → v4.** The remaining errors involve knowledge that string features and a 278M encoder lack: acronyms such as "FBP" for "Fun Boutique Primaire SAS", French abbreviations and region names, and coined trade names. A multilingual LLM reranker, applied only where the stacker is uncertain, brings that knowledge at a manageable cost. It gave the largest leaderboard step, +0.0022 (0.9834 → 0.985619). Widening the scored band in v4b added a further +0.000359 (0.985978).

---

## 5. Results & Error Analysis

Note: v3/v4 values in this document were logged during our training runs; v2 values are reproduced by executed notebook output (MLChallenge_v2.ipynb).

**Runs.** All values are on the same 100,000 held-out ghost-train S1 (`val`).

| run | val macro F0.5 | stack_train OOF F0.5 | singletons | macro precision | macro recall | public LB |
|---|---|---|---|---|---|---|
| v2 | 0.99013 | 0.98996 | 0.99409 | 0.99739 | 0.97484 | 0.981874 |
| v3a | 0.99053 | 0.9901 | 0.99481 | 0.99768 | 0.97539 | – |
| v3b | 0.99098 | 0.99061 | 0.99517 | 0.99779 | 0.97664 | 0.9834 |
| v4a | 0.99153 | 0.99105 | 0.99499 | 0.99798 | 0.97772 | 0.985619 |
| **v4b (final submission)** | **0.99158** | 0.9911 | 0.99517 | 0.99803 | 0.97776 | **0.985978** |
| Stage B (not selected) | 0.9916 | 0.99111 | 0.99463 | 0.998 | 0.97781 | – |

Stage B continued Qwen fine-tuning with generic training-only noise: street abbreviations, case, accents, dropped or reordered address parts, acronyms. Its validation F0.5 was 0.00002 higher, but its singleton score was lower (0.99463 vs 0.99517), so we kept v4b. These three runs differ by less than single-split validation noise; Stage A was selected because it was also best on the public leaderboard (0.985978 vs 0.985619 for Test 3).

**Where the remaining F0.5 is lost (final run, validation).**

| error type | F0.5 points lost |
|---|---|
| true records missed (no wrong records) | 0.00527 |
| S1 with true records predicted empty | 0.00151 |
| extra wrong record (no misses) | 0.00123 |
| singleton given a match | 0.00027 |
| both missed and wrong records | 0.00014 |
| **total** (= 1 − 0.99158) | **0.00842** |

**Where true validation pairs are lost** (346,212 true pairs):

| stage | v2 | final pipeline |
|---|---|---|
| never retrieved by sparse or dense | 488 | 488 |
| removed by the stage-2 cut | 2,346 | 1,492 |
| rejected by the matcher | 6,150 | 5,864 |
| found | 337,228 | 338,368 |

**Common error types.** Most of the loss is recall: missed records and S1 left empty. Our diagnostics point to address-less records whose name several S1 share. When a record has no address, the name alone cannot tell the competing S1 apart. The diagnostic values below were logged from our analysis runs:

| logged diagnostic | value |
|---|---|
| precision of address-less candidate pairs with 0.40 ≤ p < 0.70 | 0.52 (below the 0.698 break-even) |
| S1 predicted empty that are true singletons | 97.4 % |
| address-less misses where the raw spelling is identical, so spelling cannot separate the claimants | 57.5 % |
| sibling evidence helpful vs misleading on those misses | 17 % vs 8 % |

Adding these records would lower F0.5. Rules that force a match for every S1 would hurt, because almost all empty predictions are true singletons. We therefore abstain on them.

**France.** France has no training labels, so its accuracy cannot be measured locally. Three observations on test:
- France S1 receive 3.321 predicted matches on average, with 5.75 % left empty, similar to India (3.39, 5.76 %) and the US (3.385, 5.82 %) (executed, `MLChallenge_v4(Final).ipynb` cell 42).
- Acronym-like record names (2–5 capitals) are 1.67 % of France records, against 0.19 % in India and 0.09 % in the US (executed, cell 13).
- 31.84 % of France records name a département, found with a keyword heuristic. That is a different administrative level from the région, which 33.29 % of records name (executed, cell 13).

The Qwen reranker targets exactly these cases. Our logged summary shows the share of France acronym-like records that receive a match rose from 72 % to 84 % when it was added (*values logged from our training runs*).

The gap between validation (0.99158) and public LB (0.985978) is expected: France (~15 % of test S1) has no labels and cannot be validated, the test distractor rate is estimated rather than known, and the public LB covers only a subset of test.

**Limitations.**
- **Address-less records with shared names** remain the main error source, and the text holds no signal to separate them.
- **France is unvalidated.** No label exists, and the leaderboard gives only one aggregate score.
- **Transformer training is not bit-exact.** GPU training is not bit-exact because the torch random state is not seeded; the tree models and all data splits use fixed seeds (42). A re-run reproduces the results within run-to-run noise, not to the last digit.

---

## 6. Conclusion

The biggest gains came from matching the test conditions rather than from larger models. Training on a test-like share of unmatched records, together with dense retrieval and a stronger cross-encoder, took the leaderboard from about 0.94 (v1) to 0.981874 (v2). Using the rule that a record belongs to at most one business, at every stage, recovered pairs that looked weak in isolation. A small multilingual LLM, applied only to uncertain pairs, handled acronyms and French names and gave the largest late step, to a final 0.985978. What remains is mostly address-less records whose names several businesses share. Under a precision-weighted metric, we leave those unmatched rather than guess.

---

## Appendix

### A. Code Artefacts

The package contains `code/business_entity_resolution/src/er_v2/` (pipeline modules), `src/utils/validate_submission.py` (official validator), `README.md` and `requirements.txt`. The modules are grouped below.

**Pipeline stages**

| module | role |
|---|---|
| `config.py` | paths, seeds (42) and cache version tags |
| `data.py` | integer id maps, ground truth, subsets |
| `normalize.py` | normalisation |
| `splits.py` | ghost-S1 removal and disjoint subsets |
| `blocking_sparse.py` | 11-family IDF key blocking |
| `dense.py` | bi-encoder fine-tuning and top-k search |
| `stage2.py` | union and pass-1 (v2) re-ranker |
| `stage2_v3.py` | two-pass re-ranker and final cut |
| `features.py` | pair features |
| `level1.py` | level-1 model and competition features |
| `cross_encoder.py` | cross-encoder training and scoring |
| `stack.py` | stacker and submission writer |
| `qwen_ce.py` | Qwen reranker (and the unselected Stage B) |
| `collective.py` | collective pass and threshold study |
| `write_final.py` | exclusive assignment + threshold → final TSVs |
| `metrics.py` | exact metric, candidate reports, decision rules |
| `tok_fast.py` | fast batched tokenisation |

**Drivers**

| file | role |
|---|---|
| `run_all.py` | the full v2 pipeline |
| `run_test1.py` | v3a |
| `run_test2.py` | v3b |
| `chain_final.sh` | v4b and the unselected Stage B |
| `chain1.sh`–`chain3.sh`, `run_bg.sh` | v2-era orchestration |

v4 has no single driver module; it runs as the commands below.

**Diagnostics and analysis** (not part of the submission path)

| module | role |
|---|---|
| `decide.py` | v2 decision-rule study |
| `summary.py` | per-run validation report |
| `diag_recall.py` | where true pairs are lost |
| `diag_gap.py` | val→leaderboard gap |
| `sibling_analysis.py` | address-less sibling resolvability |
| `diag_sibling_order.py` | file order / id order check |
| `report.py` | full v2 report |
| `probe.py` | leaderboard probe files |
| `acronym_rule.py` | optional rule, **not used** |

**Packaging and notebooks**

| file | role |
|---|---|
| `build_notebook.py`, `build_notebook_v3.py` | notebook builders |
| `package.py`, `build_package_local.py`, `make_transfer_bundle.py` | packaging |
| `MLChallengeV1.ipynb` | v1 |
| `MLChallenge_v2.ipynb` | executed v2 run |
| `MLChallenge_v4(Final).ipynb` | EDA and the validator on the final files |

Every stage caches versioned outputs and resumes after an interruption.

**Reproduction order.** Run from `src/`, with `ER_DATA_DIR` (the 7 TSVs) and `ER_ROOT` (work directory) set:

```bash
pip install -r requirements.txt
# v2 base stages
python -m er_v2.normalize && python -m er_v2.splits && python -m er_v2.blocking_sparse && python -m er_v2.dense
python -m er_v2.stage2
# v3a, v3b
ER_STAGE2=s3 ER_FEAT=f3 python -m er_v2.run_test1
ER_STAGE2=s3 ER_FEAT=f3 ER_L1=b ER_STACK=r python -m er_v2.run_test2
# v4a
ER_STAGE2=s3 ER_FEAT=f3 python -m er_v2.qwen_ce train
ER_STAGE2=s3 ER_FEAT=f3 python -m er_v2.qwen_ce score
ER_STAGE2=s3 ER_FEAT=f3 ER_L1=b ER_STACK=r ER_QW=1 python -m er_v2.collective
# v4b (final submission)
ER_STAGE2=s3 ER_FEAT=f3 ER_QW_BAND=0.02,0.98 ER_QW_MODEL=qwen_rr06 ER_QW_OUT=_wide ER_QW_REUSE= python -m er_v2.qwen_ce score
ER_STAGE2=s3 ER_FEAT=f3 ER_L1=b ER_STACK=r ER_QW=1 ER_QW_FILES=_wide:qw ER_QW_TAG=_qwA ER_RULE=global python -m er_v2.collective
ER_STAGE2=s3 ER_FEAT=f3 python -m er_v2.write_final test_p_f3br_qwA_coll_e5b.parquet 0.70 $ER_ROOT/../output
# validate
python utils/validate_submission.py --matching $ER_ROOT/../output/matching_results.tsv \
    --candidate $ER_ROOT/../output/candidate_pairs.tsv --test-dir $ER_DATA_DIR --check-ids
```

### B. Additional Results

**v2 executed results** (`MLChallenge_v2.ipynb` cells 8, 10, 12, 16)

| experiment | val macro F0.5 | oracle F0.5 | cand/S1 |
|---|---|---|---|
| level-1 LightGBM, features only | 0.982001 | 0.99792 | 5.353 |
| cross-encoder (e5-base) only | 0.985383 | 0.99792 | 5.353 |
| final v2 stack (threshold 0.69, exclusive) | 0.990130 | 0.99792 | 5.353 |

| v2 validation by group | macro F0.5 |
|---|---|
| all | 0.99013 |
| India (40,020 S1) | 0.99118 |
| US (59,980 S1) | 0.98943 |
| singletons | 0.99409 |

| v2 model statistic | value |
|---|---|
| stage-2 re-ranker best iteration | 681 |
| level-1 OOF AUC | 0.996784 |
| stacker OOF AUC | 0.99895 |
| v2 cut on `rr_tune` | kmax 8, floor 0.01 → 5.343 cand/S1, oracle 0.99796 |

**v2 retrieval ablation** (`rr_tune`, executed)

| candidate set | cand/S1 | pair recall | oracle F0.5 |
|---|---|---|---|
| sparse only, top-5 | 5.0 | 0.8405 | 0.95425 |
| sparse only, top-10 | 9.997 | 0.9257 | 0.97353 |
| sparse only, top-50 | 49.759 | 0.9574 | 0.9841 |
| dense only, top-5 | 5.0 | 0.9042 | 0.98344 |
| dense only, top-10 | 10.0 | 0.9902 | 0.99741 |
| dense only, top-40 | 40.0 | 0.9977 | 0.99934 |
| union | 86.747 | 0.9986 | 0.99959 |

**Matcher ablations, final architecture**

| matcher | val macro F0.5 |
|---|---|
| level-1 LightGBM, features only | 0.98466 |
| e5-base cross-encoder alone | 0.98545 |
| stacker | 0.99093 |
| + collective pass | 0.99098 |
| + Qwen3-Reranker-0.6B (v4a) | 0.99153 |
| final run (v4b, wider Qwen band) | 0.99158 |
| perfect matcher on our candidates | 0.99846 |

*The executed v2 counterparts are in the first table of this appendix.*

**Official validator on the final files** (`MLChallenge_v4(Final).ipynb` cell 42)
- **Result:** PASS.
- 1,732,544 rows in each file.
- `matching_results.tsv`: 100,129 empty and 1,632,415 non-empty rows.
- `candidate_pairs.tsv`: 5,354 empty and 1,727,190 non-empty rows.
- 9,969,589 valid S2/S3 ids checked.
- Every match lies inside its S1's candidate list, and no record is assigned to two S1.

**Key library versions** (`requirements.txt`): Python 3.12.11, polars 1.44.2, pyarrow 25.0.1, numpy 1.26.4, scikit-learn 1.3.2, scipy 1.11.4, lightgbm 4.7.0, rapidfuzz 3.14.6, anyascii 0.3.3, torch 2.8.0, transformers 5.17.0, tokenizers 0.23.2, safetensors 0.8.0, huggingface-hub 1.33.0.
