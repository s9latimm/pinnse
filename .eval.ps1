python -m src.nse -e slalom --id "eval/slalom_1" -l 20:20:20:20:20 -i 1 --nu .1 -d cuda -pr -fg -n 30000
python -m src.nse -e slalom --id "eval/slalom_2" -l 50:50:50:50 -i 1 --nu .1 -d cuda -pr -fg -n 30000
python -m src.nse -e slalom --id "eval/slalom_3" -l 80:80:80 -i 1 --nu .1 -d cuda -pr -fg -n 30000
python -m src.nse -e slalom --id "eval/slalom_4" -l 100:100:100 -i 1 --nu .1 -d cuda -pr -fg -n 30000
python -m src.nse -e slalom --id "eval/slalom_5" -l 100:100:100:100 -i 1 --nu .1 -d cuda -pr -fg -n 30000
python -m src.nse -e slalom --id "eval/slalom_6" -l 200:200:200 -i 1 --nu .1 -d cuda -pr -fg -n 30000
