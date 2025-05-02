Siapin
-wandb token
-huggingface API & username
-wallet coldkey(utama)

Start VPS & install bittensor-cli

wget https://raw.githubusercontent.com/rennzone/Auto-Install-Bittensor-Script/refs/heads/main/bittensor-cli.sh && bash bittensor-cli.sh
-

regen coldkey
create new wallet hotkey, simpen phrase

git clone https://github.com/rayonlabs/G.O.D.git
wget https://raw.githubusercontent.com/firzahdzm/bittensor/refs/heads/main/rtxsetup.sh && bash rtxsetup.sh
wget https://raw.githubusercontent.com/firzahdzm/bittensor/refs/heads/main/fixing56.sh && bash fixing56.sh
cd G.O.D
sudo -E ./bootstrap.sh
source $HOME/.bashrc
source $HOME/.venv/bin/activate
task install

python3 -m core.create_config --miner

masukin pilihan wallet, wandb token, Huggingface api & username

sudo reboot

tunggu reboot selesai

regist hotkey

btcli s register --netuid 56 --wallet.name [coldkey] --wallet.hotkey [hotkey]

lalu fiber post

fiber-post-ip --netuid 56 --subtensor.network finney --external_port 7999 --wallet.name [coldkey] --wallet.hotkey [hotkey] --external_ip [ip]

install screen

apt install screen

lalu open screen

screen -S [nama-screen]

udah masuk screen lalu run

cd G.O.D
task miner

keluar screen pakai CTRL+A+D

DONE
