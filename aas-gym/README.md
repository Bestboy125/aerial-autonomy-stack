# aas-gym

Install Anaconda (https://docs.conda.io/projects/conda/en/stable/user-guide/install/linux.html)
```sh
wget https://repo.anaconda.com/archive/Anaconda3-2025.06-0-Linux-x86_64.sh
bash Anaconda3-2025.06-0-Linux-x86_64.sh
```

Install `aas-gym`
```sh
cd aerial-autonomy-stack/aas-gym/
conda create -n aas python=3.13
conda activate aas
pip3 install --upgrade pip
pip3 install -e .
```

Use
```sh
python3 test.py --mode step       # Manually step the simulation
python3 test.py --mode speed      # Check the simulation throughput
python3 test.py --mode learn      # Train and test a PPO agent
```

<!--

Clean up with:
docker stop $(docker ps -q) && docker container prune -f && docker network prune -f

-->
