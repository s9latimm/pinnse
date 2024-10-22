python -m src.nse -D cuda -PFGN 200000 -L 150:150:150 -E empty  --id "eval_5/empty__0_100__0_020__0_500"  --nu 0.02 --inlet .5
python -m src.nse -D cuda -PFGN 200000 -L 150:150:150 -E empty  --id "eval_5/empty__0_100__0_010__1_000"  --nu 0.01 --inlet 1
python -m src.nse -D cuda -PFGN 200000 -L 150:150:150 -E empty  --id "eval_5/empty__0_100__0_020__1_000"  --nu 0.02 --inlet 1
python -m src.nse -D cuda -PFGN 200000 -L 150:150:150 -E empty  --id "eval_5/empty__0_100__0_020__2_000"  --nu 0.02 --inlet 2
python -m src.nse -D cuda -PFGN 200000 -L 150:150:150 -E empty  --id "eval_5/empty__0_100__0_040__1_000"  --nu 0.04 --inlet 1

python -m src.nse -D cuda -PFGN 200000 -L 150:150:150 -E step   --id "eval_5/step__0_100__0_010__0_500"   --nu 0.01  --inlet .5
python -m src.nse -D cuda -PFGN 200000 -L 150:150:150 -E step   --id "eval_5/step__0_100__0_005__1_000"   --nu 0.005 --inlet 1
python -m src.nse -D cuda -PFGN 200000 -L 150:150:150 -E step   --id "eval_5/step__0_100__0_010__1_000"   --nu 0.01  --inlet 1
python -m src.nse -D cuda -PFGN 200000 -L 150:150:150 -E step   --id "eval_5/step__0_100__0_010__2_000"   --nu 0.01  --inlet 2
python -m src.nse -D cuda -PFGN 200000 -L 150:150:150 -E step   --id "eval_5/step__0_100__0_020__1_000"   --nu 0.02  --inlet 1

python -m src.nse -D cuda -PFGN 200000 -L 150:150:150 -E slalom --id "eval_5/slalom__0_100__0_010__0_500" --nu 0.01  --inlet .5
python -m src.nse -D cuda -PFGN 200000 -L 150:150:150 -E slalom --id "eval_5/slalom__0_100__0_005__1_000" --nu 0.005 --inlet 1
python -m src.nse -D cuda -PFGN 200000 -L 150:150:150 -E slalom --id "eval_5/slalom__0_100__0_010__1_000" --nu 0.01  --inlet 1
python -m src.nse -D cuda -PFGN 200000 -L 150:150:150 -E slalom --id "eval_5/slalom__0_100__0_010__2_000" --nu 0.01  --inlet 2
python -m src.nse -D cuda -PFGN 200000 -L 150:150:150 -E slalom --id "eval_5/slalom__0_100__0_020__1_000" --nu 0.02  --inlet 1

python -m src.nse -D cuda -PFGN 200000 -L 150:150:150 -E block  --id "eval_5/block__0_100__0_020__0_500"  --nu 0.02 --inlet .5
python -m src.nse -D cuda -PFGN 200000 -L 150:150:150 -E block  --id "eval_5/block__0_100__0_010__1_000"  --nu 0.01 --inlet 1
python -m src.nse -D cuda -PFGN 200000 -L 150:150:150 -E block  --id "eval_5/block__0_100__0_020__1_000"  --nu 0.02 --inlet 1
python -m src.nse -D cuda -PFGN 200000 -L 150:150:150 -E block  --id "eval_5/block__0_100__0_020__2_000"  --nu 0.02 --inlet 2
python -m src.nse -D cuda -PFGN 200000 -L 150:150:150 -E block  --id "eval_5/block__0_100__0_040__1_000"  --nu 0.04 --inlet 1

python -m src.nse -D cuda -PFGN 200000 -L 150:150:150 -E slit   --id "eval_5/slit__0_100__0_020__0_500"   --nu 0.02 --inlet .5
python -m src.nse -D cuda -PFGN 200000 -L 150:150:150 -E slit   --id "eval_5/slit__0_100__0_010__1_000"   --nu 0.01 --inlet 1
python -m src.nse -D cuda -PFGN 200000 -L 150:150:150 -E slit   --id "eval_5/slit__0_100__0_020__1_000"   --nu 0.02 --inlet 1
python -m src.nse -D cuda -PFGN 200000 -L 150:150:150 -E slit   --id "eval_5/slit__0_100__0_020__2_000"   --nu 0.02 --inlet 2
python -m src.nse -D cuda -PFGN 200000 -L 150:150:150 -E slit   --id "eval_5/slit__0_100__0_040__1_000"   --nu 0.04 --inlet 1
