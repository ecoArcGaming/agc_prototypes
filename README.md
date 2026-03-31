# agc_prototypes

Code structure:

1. `main.py`: responsible for the main event loop, checking and responding to button inputs
2. `vlm.py`: responsible for processing inputs and returning an output, called by main on events. 

Since we are likely short on RAM:

```
# 1. Disable the Ubuntu Desktop GUI (frees ~1.5GB of RAM)
sudo systemctl set-default multi-user.target
sudo reboot

# 2. (After reboot) Mount an NVMe SSD and create a 16GB Swap file
# DO NOT put the swap file on the SD card, it will ruin it.
sudo fallocate -l 16G /mnt/nvme/swapfile
sudo chmod 600 /mnt/nvme/swapfile
sudo mkswap /mnt/nvme/swapfile
sudo swapon /mnt/nvme/swapfile

```