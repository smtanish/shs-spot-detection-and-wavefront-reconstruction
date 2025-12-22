import os
import shutil

BASE_DIR = "perfectspots"
IP_DIR = os.path.join(BASE_DIR, "IP")
IA_DIR = os.path.join(BASE_DIR, "IA")

# ===================== IP =====================
# Duplicate IP -> IP_0001 ... IP_0050

ip_files = [f for f in os.listdir(IP_DIR) if os.path.splitext(f)[0] == "IP"]

if len(ip_files) != 1:
    raise RuntimeError("IP folder must contain exactly one file named 'IP'")

ip_src = ip_files[0]
_, ip_ext = os.path.splitext(ip_src)

for i in range(1, 51):
    dst_name = f"IP_{i:04d}{ip_ext}"
    shutil.copy(
        os.path.join(IP_DIR, ip_src),
        os.path.join(IP_DIR, dst_name)
    )

print("perfectspots/IP duplication complete.")


# ===================== IA =====================
# Duplicate IA_sph -> IA_0001 ... IA_0050

ia_files = [f for f in os.listdir(IA_DIR) if os.path.splitext(f)[0] == "IA_sph"]

if len(ia_files) != 1:
    raise RuntimeError("IA folder must contain exactly one file named 'IA_sph'")

ia_src = ia_files[0]
_, ia_ext = os.path.splitext(ia_src)

for i in range(1, 51):
    dst_name = f"IA_{i:04d}{ia_ext}"
    shutil.copy(
        os.path.join(IA_DIR, ia_src),
        os.path.join(IA_DIR, dst_name)
    )

print("perfectspots/IA duplication complete.")
