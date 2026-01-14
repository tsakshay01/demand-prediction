# How to Upload Your Project to GitHub

Since I have already prepared the code on your computer, you just need to connect it to the internet.

### Step 1: Create the Repository on GitHub
1. Open your browser and go to [GitHub.com/new](https://github.com/new).
2. **Repository name**: `demand-prediction-system`
3. **Description**: "Multimodal Demand Prediction System using Python (TensorFlow) and Node.js".
4. **Public/Private**: Choose "Public" (easier) or "Private".
5. **Do NOT check** "Initialize with README", "Add .gitignore", or "Choose a license". (I already did this for you).
6. Click **Create repository**.

### Step 2: Connect Your Computer
Once created, GitHub will show you a page with commands. Look for the section **"…or push an existing repository from the command line"**.

Copy the command that looks like this (but with your username):
```bash
git remote add origin https://github.com/YOUR_USERNAME/demand-prediction-system.git
```

### Step 3: Send the Code
1. Open your terminal in this folder.
2. Paste the command you copied above and press Enter.
3. Then run this final command:
```bash
git push -u origin master
```

### Done!
Refresh your GitHub page, and you will see all your files there.
