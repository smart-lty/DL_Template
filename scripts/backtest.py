import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", "-e", type=str, default="test")
parser.add_argument("--gpu", type=int, default=0)
args = parser.parse_args()

exp_path = os.path.join("experiments", args.exp_name)

for i in range(1, 21):
    try:
        cmd = f"python src/main.py --gpu {args.gpu} --do_pre --init {exp_path}/epoch_{i}.pth"
        os.system(cmd)
        cmd = f"python ~/backtest/long100.risk.py {exp_path}/backtest_results/"
        os.system(cmd)
    except:
        print(f"Skip Epoch {i}")
