nohup python src/main.py --gpu 1 -e grid_search/loss_mse_reg_0.1_neg_500 --loss mse --reg 0.1  --neg_sample_num 500 > /dev/null 2>&1 &
nohup python src/main.py --gpu 3 -e grid_search/loss_mse_reg_0.1_neg_1000 --loss mse --reg 0.1  --neg_sample_num 1000 > /dev/null 2>&1 &
nohup python src/main.py --gpu 1 -e grid_search/loss_mse_reg_0.1_neg_2000 --loss mse --reg 0.1  --neg_sample_num 2000 > /dev/null 2>&1 &
nohup python src/main.py --gpu 3 -e grid_search/loss_mse_reg_0.1_neg_4000 --loss mse --reg 0.1  --neg_sample_num 4000 > /dev/null 2>&1 &
nohup python src/main.py --gpu 1 -e grid_search/loss_mse_reg_0.2_neg_500 --loss mse --reg 0.2  --neg_sample_num 500 > /dev/null 2>&1 &
nohup python src/main.py --gpu 3 -e grid_search/loss_mse_reg_0.2_neg_1000 --loss mse --reg 0.2  --neg_sample_num 1000 > /dev/null 2>&1 

nohup python src/main.py --gpu 1 -e grid_search/loss_mse_reg_0.2_neg_2000 --loss mse --reg 0.2  --neg_sample_num 2000 > /dev/null 2>&1 &
nohup python src/main.py --gpu 3 -e grid_search/loss_mse_reg_0.2_neg_4000 --loss mse --reg 0.2  --neg_sample_num 4000 > /dev/null 2>&1 &
nohup python src/main.py --gpu 1 -e grid_search/loss_mse_reg_0.3_neg_500 --loss mse --reg 0.3  --neg_sample_num 500 > /dev/null 2>&1 &
nohup python src/main.py --gpu 3 -e grid_search/loss_mse_reg_0.3_neg_1000 --loss mse --reg 0.3  --neg_sample_num 1000 > /dev/null 2>&1 &
nohup python src/main.py --gpu 1 -e grid_search/loss_mse_reg_0.3_neg_2000 --loss mse --reg 0.3  --neg_sample_num 2000 > /dev/null 2>&1 &
nohup python src/main.py --gpu 3 -e grid_search/loss_mse_reg_0.3_neg_4000 --loss mse --reg 0.3  --neg_sample_num 4000 > /dev/null 2>&1 

nohup python src/main.py --gpu 1 -e grid_search/loss_mse_reg_0.4_neg_500 --loss mse --reg 0.4  --neg_sample_num 500 > /dev/null 2>&1 &
nohup python src/main.py --gpu 3 -e grid_search/loss_mse_reg_0.4_neg_1000 --loss mse --reg 0.4  --neg_sample_num 1000 > /dev/null 2>&1 &
nohup python src/main.py --gpu 1 -e grid_search/loss_mse_reg_0.4_neg_2000 --loss mse --reg 0.4  --neg_sample_num 2000 > /dev/null 2>&1 &
nohup python src/main.py --gpu 3 -e grid_search/loss_mse_reg_0.4_neg_4000 --loss mse --reg 0.4  --neg_sample_num 4000 > /dev/null 2>&1 &
nohup python src/main.py --gpu 1 -e grid_search/loss_tr_reg_0.1_neg_500 --loss tr --reg 0.1  --neg_sample_num 500 > /dev/null 2>&1 &
nohup python src/main.py --gpu 3 -e grid_search/loss_tr_reg_0.1_neg_1000 --loss tr --reg 0.1  --neg_sample_num 1000 > /dev/null 2>&1 

nohup python src/main.py --gpu 1 -e grid_search/loss_tr_reg_0.1_neg_2000 --loss tr --reg 0.1  --neg_sample_num 2000 > /dev/null 2>&1 &
nohup python src/main.py --gpu 3 -e grid_search/loss_tr_reg_0.1_neg_4000 --loss tr --reg 0.1  --neg_sample_num 4000 > /dev/null 2>&1 &
nohup python src/main.py --gpu 1 -e grid_search/loss_tr_reg_0.2_neg_500 --loss tr --reg 0.2  --neg_sample_num 500 > /dev/null 2>&1 &
nohup python src/main.py --gpu 3 -e grid_search/loss_tr_reg_0.2_neg_1000 --loss tr --reg 0.2  --neg_sample_num 1000 > /dev/null 2>&1 &
nohup python src/main.py --gpu 1 -e grid_search/loss_tr_reg_0.2_neg_2000 --loss tr --reg 0.2  --neg_sample_num 2000 > /dev/null 2>&1 &
nohup python src/main.py --gpu 3 -e grid_search/loss_tr_reg_0.2_neg_4000 --loss tr --reg 0.2  --neg_sample_num 4000 > /dev/null 2>&1 

nohup python src/main.py --gpu 1 -e grid_search/loss_tr_reg_0.3_neg_500 --loss tr --reg 0.3  --neg_sample_num 500 > /dev/null 2>&1 &
nohup python src/main.py --gpu 3 -e grid_search/loss_tr_reg_0.3_neg_1000 --loss tr --reg 0.3  --neg_sample_num 1000 > /dev/null 2>&1 &
nohup python src/main.py --gpu 1 -e grid_search/loss_tr_reg_0.3_neg_2000 --loss tr --reg 0.3  --neg_sample_num 2000 > /dev/null 2>&1 &
nohup python src/main.py --gpu 3 -e grid_search/loss_tr_reg_0.3_neg_4000 --loss tr --reg 0.3  --neg_sample_num 4000 > /dev/null 2>&1 &
nohup python src/main.py --gpu 1 -e grid_search/loss_tr_reg_0.4_neg_500 --loss tr --reg 0.4  --neg_sample_num 500 > /dev/null 2>&1 &
nohup python src/main.py --gpu 3 -e grid_search/loss_tr_reg_0.4_neg_1000 --loss tr --reg 0.4  --neg_sample_num 1000 > /dev/null 2>&1 

nohup python src/main.py --gpu 0 -e grid_search/loss_tr_reg_0.4_neg_2000 --loss tr --reg 0.4  --neg_sample_num 2000 > /dev/null 2>&1 &
nohup python src/main.py --gpu 1 -e grid_search/loss_tr_reg_0.4_neg_4000 --loss tr --reg 0.4  --neg_sample_num 4000 > /dev/null 2>&1 &
