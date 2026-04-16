# Literature Review

## Scope

This literature review positions the project in the part of the chess-ML literature most relevant to the current system:
- human move prediction
- skill-aware chess modeling
- personalized behavior modeling
- style embeddings and behavioral stylometry
- human-AI alignment in chess

The key point is that this project is **not** primarily about building the strongest chess engine. It is about **modeling human decision-making and structural weakness in a rating-aware way**.

## 1. Foundational Chess AI Context

### Silver et al. (2018), *A General Reinforcement Learning Algorithm that Masters Chess, Shogi, and Go through Self-Play*
- Source: Science / DeepMind
- Link: https://www.science.org/doi/10.1126/science.aar6404

Why it matters:
- AlphaZero established the modern neural/self-play paradigm for top-level chess AI.
- It showed that deep neural approaches could learn strong chess play directly from self-play rather than hand-crafted evaluation.

Relevance to this project:
- This project uses chess as a deep learning domain, but with a different objective.
- AlphaZero is about **superhuman play**.
- Our project is about **human-aligned modeling** and **weakness diagnosis**.

Key difference:
- AlphaZero optimizes winning.
- This project optimizes alignment with human weakness structure and rating-conditioned behavior.

## 2. Maia: Human-Aligned Chess Modeling

### McIlroy-Young et al. (2020), *Aligning Superhuman AI with Human Behavior: Chess as a Model System*
- Source: arXiv / KDD 2020
- Link: https://arxiv.org/abs/2006.01855

What it introduced:
- Maia showed that chess models can be trained to predict **human moves**, not just strong engine moves.
- It made skill-aware chess prediction central: different models were trained to align with different rating levels.

Why it matters:
- This is the most important conceptual predecessor to the current project.
- It reframed chess AI from “play strongest move” to “predict the move a human at a given skill level would make.”

How this project builds on it:
- Maia models skill-aware behavior.
- Our project goes one step further and asks:
  - what is expected at a given rating?
  - what residual weakness remains after accounting for rating?

In short:
- Maia is the foundation for **skill-aware human modeling**.
- Our project extends that direction toward **rating-adjusted weakness modeling**.

## 3. Personalized Human Behavior Modeling

### McIlroy-Young et al. (2020), *Learning Models of Individual Behavior in Chess*
- Source: KDD 2020 / authors’ project line
- DOI/record: https://dblp.org/rec/conf/kdd/McIlroy-YoungWSKA20.html

What it introduced:
- This work moved from population-level human modeling to **individual-level modeling**.
- The main idea was that fine-tuning to specific players improves move prediction beyond aggregate Maia-style models.

Why it matters:
- It supports the idea that players have individually learnable decision signatures.
- This is directly relevant to our embedding and personalization goals.

How this project differs:
- That paper focuses on predicting an individual player’s future decisions.
- Our project focuses on learning a representation of **phase-level residual weakness**, not just next-move imitation.

Takeaway:
- This paper motivates personalization.
- Our work changes the target from “which move will they choose?” to “what structural weakness profile do they exhibit?”

## 4. Behavioral Stylometry in Chess

### McIlroy-Young et al. (2021), *Detecting Individual Decision-Making Style: Exploring Behavioral Stylometry in Chess*
- Source: NeurIPS 2021
- Link: https://www.microsoft.com/en-us/research/publication/detecting-individual-decision-making-style-exploring-behavioral-stylometry-in-chess/

What it introduced:
- This paper formalized **behavioral stylometry** in chess.
- It used transformer-based modeling to identify players from their moves alone.

Why it matters:
- It is the clearest prior evidence that chess decisions contain a stable, learnable style signal.
- It also supports the use of embeddings as a meaningful representation of player-specific behavior.

How this project builds on it:
- Stylometry asks whether players can be distinguished from one another.
- Our project asks whether player representations can be used for **rating-adjusted weakness diagnosis**.

Relation to our embedding results:
- The “embedding vs rating” evaluation in our project is conceptually aligned with stylometry.
- We are not just identifying players, but testing whether embeddings encode behavioral structure beyond rating alone.

## 5. Unified Human-AI Alignment Across Skill Levels

### Tang et al. (2024), *Maia-2: A Unified Model for Human-AI Alignment in Chess*
- Source: NeurIPS 2024
- Link: https://arxiv.org/abs/2409.20553
- Also summarized by Microsoft Research: https://www.microsoft.com/en-us/research/publication/maia-2-a-unified-model-for-human-ai-alignment-in-chess/

What it introduced:
- Maia-2 unified human-aligned modeling across skill levels instead of relying on separate skill-binned models.
- It proposed a skill-aware attention mechanism to better capture how human behavior changes with strength.

Why it matters:
- It is a direct precedent for stronger rating-aware modeling.
- It shows that rating should not be treated as a crude fixed bucket only; it can be integrated into the model more coherently.

How this project relates:
- Our current system conditions on rating bucket and predicts residual weakness relative to expected quality.
- Maia-2 suggests a stronger next step: tighter integration of rating into representation learning itself.

Takeaway:
- Maia-2 validates the broader direction of **coherent human-aligned chess modeling across skill levels**.
- Our project’s residual formulation is complementary: we explicitly model what remains after accounting for rating.

## 6. Efficient Individual Modeling with Less Data

### Tang et al. (2025), *Learning to Imitate with Less: Efficient Individual Behavior Modeling in Chess*
- Source: arXiv 2025
- Link: https://arxiv.org/abs/2507.21488
- Metadata record: https://dblp.org/rec/journals/corr/abs-2507-21488.html

What it introduced:
- This work addresses a practical weakness in personalized chess modeling: data efficiency.
- It studies how to model individual behavior with fewer games per player.

Why it matters:
- Personalization in our project is the weakest empirical component.
- One reason is that player-specific signal is sparse and noisy.
- This paper suggests that better individual modeling may be possible with more data-efficient techniques.

How this project relates:
- Our current system learns phase-level weakness profiles from windows and only modestly improves personalization.
- This line of work suggests that stronger personalization may require a more direct individual-behavior modeling approach.

## 7. Where the Current Project Fits

The literature above supports three main observations:

### A. Human chess behavior is learnable
- Maia and related work show that human moves can be modeled.
- Personalized Maia-style work shows that individual behavior can also be modeled.

### B. Player-specific style is real
- Stylometry work shows that players leave stable behavioral signatures in their decisions.
- This supports our use of embeddings and temporal stability evaluation.

### C. Rating matters, but it is not the whole story
- Maia and Maia-2 show that skill-aware modeling is critical.
- Our project adds a new layer: **residual weakness after accounting for rating-conditioned expectation**.

## 8. What Is Novel in This Project Relative to the Literature

The current project does **not** contribute a fundamentally new transformer architecture.

Its strongest novel elements are:

### 1. Rating-conditioned residual formulation
- Prior work mainly asks:
  - can we predict human moves?
  - can we model a player?
  - can we identify a player’s style?
- This project asks:
  - what weakness remains after subtracting expected performance at that rating?

### 2. Phase-level weakness target
- Raw move-level residuals were too noisy.
- The project found that **phase-aggregated residual weakness** is a learnable and more stable target.

### 3. Chronological evaluation of persistence
- The project explicitly evaluates:
  - temporal stability
  - beyond-rating embedding structure
  - personalization validity

This combination is the main project contribution.

## 9. Main Literature Review Conclusion

The literature suggests a clear progression:

1. superhuman chess engines
2. human move prediction
3. skill-aware alignment
4. personalized modeling
5. style embeddings

This project fits naturally as the next step in that sequence:

**rating-aware residual weakness modeling with player embeddings and temporal evaluation.**

That is the best way to present the project in relation to prior work.

## References

1. Silver, D. et al. (2018). *A General Reinforcement Learning Algorithm that Masters Chess, Shogi, and Go through Self-Play*. Science.
   Link: https://www.science.org/doi/10.1126/science.aar6404

2. McIlroy-Young, R., Sen, S., Kleinberg, J., Anderson, A. (2020). *Aligning Superhuman AI with Human Behavior: Chess as a Model System*.
   Link: https://arxiv.org/abs/2006.01855

3. McIlroy-Young, R. et al. (2020). *Learning Models of Individual Behavior in Chess*.
   Record: https://dblp.org/rec/conf/kdd/McIlroy-YoungWSKA20.html

4. McIlroy-Young, R., Wang, R., Kleinberg, J., Sen, S., Anderson, A. (2021). *Detecting Individual Decision-Making Style: Exploring Behavioral Stylometry in Chess*.
   Link: https://www.microsoft.com/en-us/research/publication/detecting-individual-decision-making-style-exploring-behavioral-stylometry-in-chess/

5. Tang, Z. et al. (2024). *Maia-2: A Unified Model for Human-AI Alignment in Chess*.
   Link: https://arxiv.org/abs/2409.20553

6. Tang, Z. et al. (2025). *Learning to Imitate with Less: Efficient Individual Behavior Modeling in Chess*.
   Link: https://arxiv.org/abs/2507.21488
