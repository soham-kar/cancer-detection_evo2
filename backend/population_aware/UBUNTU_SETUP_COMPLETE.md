# Ubuntu WSL2 Setup - First Time Configuration

## ✅ Ubuntu is Installed!

The installation window is asking you to create a user account.

---

## Complete These Steps:

### 1. Create Ubuntu User Account

In the Ubuntu terminal that opened, you'll see:
```
Enter new UNIX username:
```

**Enter a username** (e.g., your Windows username or just "user")
**Press Enter**

Then it will ask:
```
New password:
```

**Enter a password** (you won't see it as you type - this is normal)
**Press Enter**

**Re-enter the same password** to confirm

**Remember this password!** You'll need it for `sudo` commands.

---

### 2. After Account Creation - Install Dependencies

Once you have a Ubuntu command prompt (looks like `username@computername:~$`), run:

```bash
# Update package lists
sudo apt update

# Install Python and pip
sudo apt install python3-pip python3-dev -y

# Install pysam and other dependencies
pip3 install pysam pandas openpyxl tqdm
```

This will take 2-3 minutes.

---

### 3. Navigate to Your Project

```bash
# Your Windows D: drive is at /mnt/d/
cd /mnt/d/project/biotech-evo2/backend/population_aware

# Verify you can see files
ls
```

---

### 4. Run Day 1 Pipeline!

```bash
python3 day1_add_gnomad.py
```

**Expected:**
- Runtime: 3-5 minutes
- Uses index files for FAST queries
- Output: `results/brca1_with_gnomad.csv` with 3,893 annotated variants

---

## If the Window Closed

If the Ubuntu window closed after installation:
1. Search for "Ubuntu" in Windows Start menu
2. Open it
3. Continue from Step 1 above

---

**I'll be here once you've completed the setup!** 🚀
