#!/usr/bin/env bash
# z4-p104: container runtime setup. Run ONCE:  sudo bash ~/p104-bench/container/ROOT-SETUP.sh
# Idempotent (safe to re-run). Installs ONLY:
#   1. Docker Engine (docker-ce) from Docker's official apt repo (Ubuntu 24.04 "noble").
#      Why not docker.io: the official repo is current and ships buildx/compose; both work with the toolkit.
#   2. NVIDIA Container Toolkit from NVIDIA's official apt repo + `nvidia-ctk runtime configure --runtime=docker`.
#      Needs only the host driver (580 is far newer than the 525+ that CUDA 12.x containers require); the CUDA
#      toolkit is NOT needed on the host. sm_61 (P104) is supported by CUDA 12.x images (dropped only in CUDA 13).
#   3. json-file log rotation (max-size 50m, max-file 3) merged into /etc/docker/daemon.json.
#   4. Adds the invoking user (SUDO_USER, normally <user>) to group "docker".
#      WARNING: membership of group docker is ROOT-EQUIVALENT (can mount / into a container).
# Does NOT touch: GPU power cap, the P620, the llama-p104 user service, the serving registry, any model.
# Disk: images/layers go to /var/lib/docker on / (141 GB free). Move with "data-root" in daemon.json if ever needed.
# No images are pulled here. Group membership needs a NEW login (or `newgrp docker`) to take effect.
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
U="${SUDO_USER:-<user>}"
export DEBIAN_FRONTEND=noninteractive
CODENAME="$(. /etc/os-release && echo "$VERSION_CODENAME")"
ARCH="$(dpkg --print-architecture)"

apt-get update
apt-get install -y ca-certificates curl gnupg python3
install -d -m 0755 /etc/apt/keyrings

# --- Docker Engine (official repo)
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor --yes -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=$ARCH signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $CODENAME stable" \
  > /etc/apt/sources.list.d/docker.list

# --- NVIDIA Container Toolkit (official repo)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin nvidia-container-toolkit

# --- log rotation (merge, keep any existing keys)
install -d -m 0755 /etc/docker
python3 - <<'PY'
import json, os
p = "/etc/docker/daemon.json"
d = json.load(open(p)) if os.path.exists(p) and os.path.getsize(p) else {}
d["log-driver"] = "json-file"
d["log-opts"] = {"max-size": "50m", "max-file": "3"}
json.dump(d, open(p, "w"), indent=2)
PY

nvidia-ctk runtime configure --runtime=docker      # merges the nvidia runtime into daemon.json
systemctl enable docker
systemctl restart docker

getent group docker >/dev/null || groupadd docker
usermod -aG docker "$U"

echo "--- versions"
docker --version; nvidia-ctk --version | head -1; cat /etc/docker/daemon.json
echo
echo "OK. $U must log in again (or run: newgrp docker). Then, WITHOUT sudo, verify (pulls ~100 MB base image):"
echo "  docker run --rm --gpus device=<gpu-uuid> -e CUDA_DEVICE_ORDER=PCI_BUS_ID nvidia/cuda:12.9.1-base-ubuntu24.04 nvidia-smi -L"
echo "Expected: exactly one line, the P104-100 with UUID GPU-3c2a902a-...; the P620 must NOT be listed."
