python cot_generation.py \
--test_split test \
--test_number -1 \
--n_shots 2 \
--prompt_format QCM-ALE \
--seed 3 \
--use_visual_clues \
--visual_clues_file data/visual_clues_with_chat.json \
--temperature 0.7 \
--n_paths 10