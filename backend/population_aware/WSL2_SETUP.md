# WSL2 Setup Guide for Population-Aware Project

## Step 1: Install WSL2

**Run in PowerShell (as Administrator):**

```powershell
wsl --install
```

This installs Ubuntu Linux as the default distribution.

**Then restart your computer** (required for WSL2 to activate).

---

## Step 2: First-Time Setup (After Restart)

1. Open "Ubuntu" from the Windows Start menu
2. Wait for installation to complete (~2 minutes)
3. Create a Linux username and password when prompted

**Important:** Remember this password - you'll need it for `sudo` commands.

---

## Step 3: Install Python Dependencies in WSL2

In the Ubuntu terminal:

```bash
# Update package lists
sudo apt update

# Install Python and pip
sudo apt install python3-pip python3-dev -y

# Install pysam (works perfectly in Linux!)
pip3 install pysam pandas openpyxl tqdm
```

---

## Step 4: Access Your Windows Files

Your Windows D: drive is accessible at `/mnt/d/` in WSL2:

```bash
# Navigate to your project
cd /mnt/d/project/biotech-evo2/backend/population_aware

# List files (verify you can see them)
ls
```

---

## Step 5: Restore Original pysam-based Code

The original `gnomad_client_local.py` will work perfectly in WSL2.

I'll restore it for you - it uses pysam for fast indexed access.

---

## Step 6: Run Day 1 Pipeline

```bash
# In WSL2 Ubuntu terminal
cd /mnt/d/project/biotech-evo2/backend/population_aware

# Run the script
python3 day1_add_gnomad.py
```

**Expected:**
- Runtime: 3-5 minutes
- Output: `results/brca1_with_gnomad.csv`
- Speed: ~17 variants/second (using index files!)

---

## Troubleshooting

### "wsl: command not found"
You need Windows 10 version 2004+ or Windows 11.
Update Windows if needed.

### "This operation returned because the timeout period expired"
Your antivirus might be blocking WSL2. Add an exception for `wsl.exe`.

### Files not visible in /mnt/d/
WSL2 automatically mounts Windows drives. If not visible:
```bash
ls /mnt/  # Should show c, d, etc.
```

---

## Benefits of WSL2

✅ **pysam works perfectly** (no compilation issues)  
✅ **All bioinformatics tools available** (bcftools, samtools, etc.)  
✅ **Access Windows files** via `/mnt/`  
✅ **Linux terminal** for all future projects  
✅ **No dual-boot needed** - runs inside Windows  

---

## Next Steps After Setup

1. Run `wsl --install` and restart
2. Open Ubuntu from Start menu
3. Run the commands in Step 3
4. Navigate to `/mnt/d/project/biotech-evo2/backend/population_aware`
5. Run `python3 day1_add_gnomad.py`

**I'll be here to help once WSL2 is installed!**
