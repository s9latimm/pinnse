python -m src.nse -D cuda -PFGN 200000 -L 150:150:150 -E empty  --id "eval_5/1.1" --nu 0.02  --inlet .5 # --dry
python -m src.nse -D cuda -PFGN 200000 -L 150:150:150 -E empty  --id "eval_5/1.2" --nu 0.01  --inlet 1  # --dry
python -m src.nse -D cuda -PFGN 200000 -L 150:150:150 -E empty  --id "eval_5/1.3" --nu 0.02  --inlet 1  # --dry
python -m src.nse -D cuda -PFGN 200000 -L 150:150:150 -E empty  --id "eval_5/1.4" --nu 0.02  --inlet 2  # --dry
python -m src.nse -D cuda -PFGN 200000 -L 150:150:150 -E empty  --id "eval_5/1.5" --nu 0.04  --inlet 1  # --dry

python -m src.nse -D cuda -PFGN 200000 -L 150:150:150 -E step   --id "eval_5/2.1" --nu 0.01  --inlet .5 # --dry
#python -m src.nse -D cuda -PFGN 200000 -L 150:150:150 -E step   --id "eval_5/2.2" --nu 0.005 --inlet 1  # --dry
#python -m src.nse -D cuda -PFGN 200000 -L 150:150:150 -E step   --id "eval_5/2.3" --nu 0.01  --inlet 1  # --dry
#python -m src.nse -D cuda -PFGN 200000 -L 150:150:150 -E step   --id "eval_5/2.4" --nu 0.01  --inlet 2  # --dry
#python -m src.nse -D cuda -PFGN 200000 -L 150:150:150 -E step   --id "eval_5/2.5" --nu 0.02  --inlet 1  # --dry

python -m src.nse -D cuda -PFGN 200000 -L 150:150:150 -E slalom --id "eval_5/3.1" --nu 0.01  --inlet .5 # --dry
#python -m src.nse -D cuda -PFGN 200000 -L 150:150:150 -E slalom --id "eval_5/3.2" --nu 0.005 --inlet 1  # --dry
#python -m src.nse -D cuda -PFGN 200000 -L 150:150:150 -E slalom --id "eval_5/3.3" --nu 0.01  --inlet 1  # --dry
#python -m src.nse -D cuda -PFGN 200000 -L 150:150:150 -E slalom --id "eval_5/3.4" --nu 0.01  --inlet 2  # --dry
python -m src.nse -D cuda -PFGN 200000 -L 150:150:150 -E slalom --id "eval_5/3.5" --nu 0.02  --inlet 1  # --dry

python -m src.nse -D cuda -PFGN 200000 -L 150:150:150 -E block  --id "eval_5/4.1" --nu 0.02  --inlet .5 # --dry
python -m src.nse -D cuda -PFGN 200000 -L 150:150:150 -E block  --id "eval_5/4.2" --nu 0.01  --inlet 1  # --dry
python -m src.nse -D cuda -PFGN 200000 -L 150:150:150 -E block  --id "eval_5/4.3" --nu 0.02  --inlet 1  # --dry
python -m src.nse -D cuda -PFGN 200000 -L 150:150:150 -E block  --id "eval_5/4.4" --nu 0.02  --inlet 2  # --dry
python -m src.nse -D cuda -PFGN 200000 -L 150:150:150 -E block  --id "eval_5/4.5" --nu 0.04  --inlet 1  # --dry

python -m src.nse -D cuda -PFGN 200000 -L 150:150:150 -E slit   --id "eval_5/5.1" --nu 0.02  --inlet .5 # --dry
python -m src.nse -D cuda -PFGN 200000 -L 150:150:150 -E slit   --id "eval_5/5.2" --nu 0.01  --inlet 1  # --dry
python -m src.nse -D cuda -PFGN 200000 -L 150:150:150 -E slit   --id "eval_5/5.3" --nu 0.02  --inlet 1  # --dry
python -m src.nse -D cuda -PFGN 200000 -L 150:150:150 -E slit   --id "eval_5/5.4" --nu 0.02  --inlet 2  # --dry
python -m src.nse -D cuda -PFGN 200000 -L 150:150:150 -E slit   --id "eval_5/5.5" --nu 0.04  --inlet 1  # --dry
