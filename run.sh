# MovieLens
nohup python main.py --device 0 --search_method Random --full_data_name MovieLens > MovieLens+7+Random+nohup.txt & 
sleep 15
nohup python main.py --device 0 --search_method TPE_SMBO --full_data_name MovieLens > MovieLens+0+TPE_SMBO+nohup.txt & 
sleep 15
nohup python main.py --device 0 --search_method SMAC_SMBO --full_data_name MovieLens > MovieLens+0+SMAC_SMBO+nohup.txt & 
sleep 15
nohup python main.py --device 0 --search_method MONO_MAB_SMBO --full_data_name MovieLens > MovieLens+0+MONO_MAB_SMBO+nohup.txt & 
sleep 15
nohup python main.py --device 0 --search_method AutoCASH --full_data_name MovieLens > MovieLens+0+AutoCASH+nohup.txt & 
sleep 15

