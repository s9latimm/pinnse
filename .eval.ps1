#python -m src.nse -D cuda -PFGN 1000 -L 150:150:150 -E empty  --id "eval_6/1.1" --nu 0.02  --inlet .5 # --dry
#python -m src.nse -D cuda -PFGN 1000 -L 150:150:150 -E empty  --id "eval_6/1.2" --nu 0.01  --inlet 1  # --dry
#python -m src.nse -D cuda -PFGN 1000 -L 150:150:150 -E empty  --id "eval_6/1.3" --nu 0.02  --inlet 1  # --dry
#python -m src.nse -D cuda -PFGN 1000 -L 150:150:150 -E empty  --id "eval_6/1.4" --nu 0.02  --inlet 2  # --dry
#python -m src.nse -D cuda -PFGN 1000 -L 150:150:150 -E empty  --id "eval_6/1.5" --nu 0.04  --inlet 1  # --dry
#
#python -m src.nse -D cuda -PFGN 1000 -L 150:150:150 -E step   --id "eval_6/2.1" --nu 0.01  --inlet .5 # --dry
#python -m src.nse -D cuda -PFGN 1000 -L 150:150:150 -E step   --id "eval_6/2.2" --nu 0.005 --inlet 1  # --dry
#python -m src.nse -D cuda -PFGN 1000 -L 150:150:150 -E step   --id "eval_6/2.3" --nu 0.01  --inlet 1  # --dry
#python -m src.nse -D cuda -PFGN 1000 -L 150:150:150 -E step   --id "eval_6/2.4" --nu 0.01  --inlet 2  # --dry
#python -m src.nse -D cuda -PFGN 1000 -L 150:150:150 -E step   --id "eval_6/2.5" --nu 0.02  --inlet 1  # --dry
#
#python -m src.nse -D cuda -PFGN 1000 -L 150:150:150 -E slalom --id "eval_6/3.1" --nu 0.01  --inlet .5 # --dry
#python -m src.nse -D cuda -PFGN 1000 -L 150:150:150 -E slalom --id "eval_6/3.2" --nu 0.005 --inlet 1  # --dry
#python -m src.nse -D cuda -PFGN 1000 -L 150:150:150 -E slalom --id "eval_6/3.3" --nu 0.01  --inlet 1  # --dry
#python -m src.nse -D cuda -PFGN 1000 -L 150:150:150 -E slalom --id "eval_6/3.4" --nu 0.01  --inlet 2  # --dry
#python -m src.nse -D cuda -PFGN 1000 -L 150:150:150 -E slalom --id "eval_6/3.5" --nu 0.02  --inlet 1  # --dry
#
#python -m src.nse -D cuda -PFGN 1000 -L 150:150:150 -E block  --id "eval_6/4.1" --nu 0.02  --inlet .5 # --dry
#python -m src.nse -D cuda -PFGN 1000 -L 150:150:150 -E block  --id "eval_6/4.2" --nu 0.01  --inlet 1  # --dry
#python -m src.nse -D cuda -PFGN 1000 -L 150:150:150 -E block  --id "eval_6/4.3" --nu 0.02  --inlet 1  # --dry
#python -m src.nse -D cuda -PFGN 1000 -L 150:150:150 -E block  --id "eval_6/4.4" --nu 0.02  --inlet 2  # --dry
#python -m src.nse -D cuda -PFGN 1000 -L 150:150:150 -E block  --id "eval_6/4.5" --nu 0.04  --inlet 1  # --dry
#
#python -m src.nse -D cuda -PFGN 1000 -L 150:150:150 -E slit   --id "eval_6/5.1" --nu 0.02  --inlet .5 # --dry
#python -m src.nse -D cuda -PFGN 1000 -L 150:150:150 -E slit   --id "eval_6/5.2" --nu 0.01  --inlet 1  # --dry
#python -m src.nse -D cuda -PFGN 1000 -L 150:150:150 -E slit   --id "eval_6/5.3" --nu 0.02  --inlet 1  # --dry
#python -m src.nse -D cuda -PFGN 1000 -L 150:150:150 -E slit   --id "eval_6/5.4" --nu 0.02  --inlet 2  # --dry
#python -m src.nse -D cuda -PFGN 1000 -L 150:150:150 -E slit   --id "eval_6/5.5" --nu 0.04  --inlet 1  # --dry

python -m src.nse -D cuda -PFGN 1000000 -L 150:150:150 -E cylinder  --id "eval_7/0.1" --nu 0.02  --inlet 1  # --dry
python -m src.nse -D cuda -PFGN 1000000 -L 150:150:150 -E wing      --id "eval_7/0.2" --nu 0.02  --inlet 1  # --dry

python -m src.nse -D cuda -PFGN 1000000 -L 150:150:150 -E block  --id "eval_7/4.4" --nu 0.02  --inlet 2     # --dry
python -m src.nse -D cuda -PFGN 1000000 -L 150:150:150 -E block  --id "eval_7/4.5" --nu 0.04  --inlet 1     # --dry
