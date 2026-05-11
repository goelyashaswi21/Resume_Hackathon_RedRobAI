# Resume Matching Engine

A skill-based resume ranking system that matches candidate resumes against job descriptions using TF-IDF vectors and cosine similarity.

## What It Does

Takes 10 candidate resumes with noisy, inconsistently written skill data and 3 job descriptions, normalizes everything through a fixed skill alias map, computes TF-IDF vectors for each resume, builds binary vectors for each JD, and ranks the top 3 matching candidates per job using cosine similarity.

## How to Run

Make sure Python 3 is installed. No external libraries needed — only the built-in `math` module is used.

```bash
python resume.py
```

## Expected Output

```
JD-1 — Kakao (ML Engineer)
Sneha Patel(0.57), Karan Mehta(0.53), Arjun Sharma(0.40)

JD-2 — Naver (Backend Engineer)
Rahul Gupta(0.81), Ananya Krishnan(0.28), Deepika Rao(0.19)

JD-3 — Line (Frontend Engineer)
Aditya Kumar(0.67), Priya Nair(0.58), Ananya Krishnan(0.35)
```

## Logic Breakdown

### 1. Skill Normalization
Raw skill strings are split on commas, lowercased, and matched against a predefined `SKILL_ALIASES` map. Multi-word phrases like `"spring boot"` or `"feature engineering"` are matched before single tokens to avoid partial mismatches. Anything not in the alias map is discarded.

### 2. Deduplication
After normalization, duplicate canonical skills are removed from each resume. This matters in cases where two different raw inputs map to the same skill — for example, both `matplotlib` and `data-viz` resolve to `data_visualization`.

### 3. Vocabulary Construction
A shared vocabulary is built from all normalized and deduplicated resume skills, sorted alphabetically. This same ordering is used for both resume and JD vectors to keep cosine similarity consistent.

### 4. TF-IDF for Resumes
Since each skill appears exactly once per resume after deduplication:

```
TF  = 1 / N              where N = total unique skills in the resume
IDF = ln(10 / df)        where df = number of resumes containing that skill
TF-IDF = TF * IDF
```

No smoothing is applied. Natural log is used throughout.

### 5. JD Binary Vectors
JD skills are also passed through the same alias map and matched against the vocabulary. Each position in the vector is 1 if the skill is required or preferred by the JD, 0 otherwise.

### 6. Cosine Similarity and Ranking
```
cosine(A, B) = dot(A, B) / (|A| * |B|)
```
Where A is the resume TF-IDF vector and B is the JD binary vector. Candidates are ranked by score descending. Ties are broken alphabetically by name.

## Dataset

**10 Candidates:** Arjun Sharma, Priya Nair, Rahul Gupta, Sneha Patel, Vikram Singh, Ananya Krishnan, Karan Mehta, Deepika Rao, Aditya Kumar, Meera Iyer

**3 Job Descriptions:**
- JD-1: Kakao, Seoul — ML Engineer
- JD-2: Naver, Seongnam — Backend Engineer
- JD-3: Line, Seoul — Frontend Engineer

## Constraints

- No external libraries (numpy, pandas, scikit-learn, etc.)
- `SKILL_ALIASES` map is used exactly as provided — not modified
- TF-IDF computed only for resumes, not JDs
- Vocabulary built from resume skills only
