# Final Demo: Test Predictions

- checkpoint: `experiments/run2/model.pt`
- data: `data/processed/games_sample_25000_stockfish_trainonly.parquet`
- split: `test`

## Sample 1

- game_id: `222716`
- window: `0:65`
- time class: `Classical`
- result: `white_win`
- white rating: true `1405`, predicted `1471.33`
- black rating: true `1405`, predicted `1474.58`
- white predicted weakness: opening=-9.38, middlegame=-16.23, endgame=-34.69
- white observed weakness: opening=65.69, middlegame=8.44, endgame=77.50
- black predicted weakness: opening=-0.69, middlegame=9.42, endgame=25.29
- black observed weakness: opening=109.29, middlegame=28.30, endgame=7.16
- opening moves preview: `e2e4 e7e5 h2h3 d7d5 e4d5 d8d5 d1e2 b8c6 b1c3 d5d8 e2e4 g8f6`

## Sample 2

- game_id: `222712`
- window: `0:112`
- time class: `Rapid`
- result: `black_win`
- white rating: true `1471`, predicted `1407.39`
- black rating: true `1464`, predicted `1421.42`
- white predicted weakness: opening=-8.40, middlegame=-33.45, endgame=-24.51
- white observed weakness: opening=-32.31, middlegame=-54.72, endgame=-17.50
- black predicted weakness: opening=-9.03, middlegame=-34.94, endgame=-29.96
- black observed weakness: opening=-32.51, middlegame=-44.22, endgame=-37.84
- opening moves preview: `d2d4 d7d5 e2e3 e7e6 f1e2 a7a6 g1f3 h7h6 e1g1 c8d7 b2b3 g8f6`

## Sample 3

- game_id: `222711`
- window: `0:46`
- time class: `Rapid`
- result: `white_win`
- white rating: true `1394`, predicted `1300.22`
- black rating: true `1411`, predicted `1395.81`
- white predicted weakness: opening=-13.22, middlegame=-47.22, endgame=-48.50
- white observed weakness: opening=-12.93, middlegame=-65.12, endgame=-61.45
- black predicted weakness: opening=-7.34, middlegame=-35.86, endgame=-33.43
- black observed weakness: opening=2.49, middlegame=-47.91, endgame=-20.80
- opening moves preview: `e2e4 e7e5 f1c4 b8c6 c2c3 g8f6 d2d3 d7d5 e4d5 f6d5 d1h5 c8e6`

## Sample 4

- game_id: `222710`
- window: `0:76`
- time class: `Rapid`
- result: `black_win`
- white rating: true `1549`, predicted `1544.59`
- black rating: true `1575`, predicted `1551.97`
- white predicted weakness: opening=-4.94, middlegame=4.59, endgame=94.75
- white observed weakness: opening=2.61, middlegame=-18.68, endgame=45.36
- black predicted weakness: opening=-6.70, middlegame=-11.14, endgame=7.23
- black observed weakness: opening=-29.39, middlegame=-24.92, endgame=3.08
- opening moves preview: `e2e4 e7e5 d1f3 g8f6 f1c4 d7d6 b1c3 c7c6 a2a3 c8g4 f3g3 d8d7`

## Sample 5

- game_id: `222709`
- window: `0:112`
- time class: `Rapid`
- result: `white_win`
- white rating: true `1321`, predicted `1278.88`
- black rating: true `1247`, predicted `1242.40`
- white predicted weakness: opening=-12.44, middlegame=-38.23, endgame=-28.53
- white observed weakness: opening=-52.33, middlegame=0.44, endgame=16.32
- black predicted weakness: opening=-10.64, middlegame=-43.07, endgame=-23.70
- black observed weakness: opening=-25.33, middlegame=-48.06, endgame=48.95
- opening moves preview: `d2d4 d7d5 g1f3 g8f6 a2a3 b8c6 e2e3 e7e5 d4e5 c6e5 f3e5 f8c5`
