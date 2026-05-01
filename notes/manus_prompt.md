# Copy-Paste Prompt for Manus AI

Create a polished slide deck from the content in `manus_enhanced_slide_deck.md`.

Presentation context:
- Course: Deep Learning final project.
- Format: Zoom presentation with a short demo.
- Audience: instructor and classmates.
- Goal: convince the audience that the project is implemented, empirically evaluated, and honestly interpreted.

Design requirements:
- Use a clean academic/research style.
- Use blue, white, charcoal, and light gray.
- Use chessboard, chess-piece, pipeline, and model-diagram visuals where useful.
- Avoid cartoon graphics, excessive decoration, and marketing-style hero slides.
- Prioritize readable charts, compact tables, and clear evidence.
- Each slide should have short visible text and more detail in speaker notes.

Deck structure:
- Build the 16 main slides from `manus_enhanced_slide_deck.md`.
- Include the two backup slides at the end.
- Keep the exact metric values.
- Keep the cautious interpretation: research prototype, not production-ready.

Important metrics to preserve:
- Train phase residual RMSE: `46.62`
- Validation phase residual RMSE: `47.12`
- Test phase residual RMSE: `47.65`
- Rating heuristic validation RMSE: `67.77`
- Ridge validation RMSE: `58.85`
- Histogram gradient boosting validation RMSE: `49.63`
- Transformer validation RMSE: `47.12`
- Embedding-neighbor RMSE: `54.80`
- Rating-neighbor RMSE: `61.99`
- Embedding beats rating rate: `57.7%`
- Train -> validation temporal RMSE: `32.08` vs random `33.57`
- Train -> validation personalization RMSE: `45.36` vs random `46.16`

Terminology:
- Say RMSE, not accuracy.
- Say centipawns, not percent.
- Explain that positive residual means worse than expected and negative residual means better than expected.
- Do not call the model production-ready.

Speaker notes:
- Include speaker notes for each slide using the notes from `manus_enhanced_slide_deck.md`.
- Keep notes conversational and suitable for a live Zoom presentation.
- Add a short transition sentence between slides where helpful.

Demo:
- Include a slide explaining that full training is too slow to run live.
- The live demo should load the saved checkpoint and generate held-out test predictions.
- Mention these commands:

```bash
python3 src/eval/final_project_report.py
python3 scripts/demo_final_project.py --num-samples 5
```

Final message:
The deck should end with this claim:

“This project delivers a leak-free transformer-based system for rating-aware chess weakness modeling, with evidence that player embeddings capture meaningful structure beyond rating alone.”

