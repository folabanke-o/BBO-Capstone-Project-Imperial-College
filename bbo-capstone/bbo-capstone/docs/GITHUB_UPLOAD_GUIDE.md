# How to Upload This Repository to GitHub

This guide walks you through publishing the BBO capstone project as a public GitHub repository step by step. No prior GitHub experience is assumed.

---

## Part A: Create a GitHub Account (skip if you already have one)

1. Go to [https://github.com](https://github.com) in your browser.
2. Click **Sign up** in the top right corner.
3. Enter your email address, create a password, and choose a username.
4. Verify your email address when GitHub sends you a confirmation link.
5. Log in to your new account.

---

## Part B: Create a New Repository on GitHub

1. Once logged in, click the **+** icon in the top right corner of any GitHub page.
2. Select **New repository** from the dropdown.
3. Fill in the form as follows:
   - **Repository name:** `bbo-capstone` (or any name you prefer)
   - **Description:** `Bayesian Black-Box Optimisation Capstone Project — 8 functions, 10 rounds`
   - **Visibility:** select **Public** so peers and facilitators can view it
   - **Initialise this repository:** leave this unticked (you already have files ready)
4. Click **Create repository**.
5. GitHub will show you a page with setup instructions. Copy the repository URL shown at the top (it will look like `https://github.com/YourUsername/bbo-capstone.git`). You will need this in Step D.

---

## Part C: Prepare Your Computer

You need Git installed on your laptop. Check whether it is already installed:

1. Open a terminal (on Windows: search for **Command Prompt** or **PowerShell**; on Mac: search for **Terminal**).
2. Type the following and press Enter:
   ```
   git --version
   ```
3. If a version number appears (e.g. `git version 2.43.0`), Git is installed. Skip to Part D.
4. If you see an error, download and install Git from [https://git-scm.com/downloads](https://git-scm.com/downloads). Accept all default settings during installation.

---

## Part D: Upload Your Files

### Step 1 — Extract the ZIP

Download the `bbo-capstone.zip` file and extract it to a folder on your laptop. You should see a folder called `bbo-capstone` containing `README.md`, `notebooks/`, `docs/`, `results/` and so on.

### Step 2 — Open a terminal in the folder

- **Windows:** open the `bbo-capstone` folder in File Explorer, then right-click in an empty area and select **Open in Terminal** (or **Open PowerShell window here**).
- **Mac:** open Terminal, then type `cd ` (with a space), drag the `bbo-capstone` folder into the Terminal window, and press Enter.

### Step 3 — Set up Git (first time only)

Run these two commands, replacing the values with your own name and email:
```
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### Step 4 — Initialise and upload

Copy and paste these commands one at a time, pressing Enter after each:

```
git init
git add .
git commit -m "Initial upload: BBO capstone project weeks 1-10"
git branch -M main
git remote add origin https://github.com/YourUsername/bbo-capstone.git
git push -u origin main
```

Replace `YourUsername` and `bbo-capstone` with your actual GitHub username and repository name from Part B.

### Step 5 — Authenticate

When prompted for a username and password:
- **Username:** your GitHub username
- **Password:** use a Personal Access Token, not your GitHub password

To create a token: go to GitHub, click your profile picture, select **Settings**, scroll to **Developer settings**, click **Personal access tokens**, then **Tokens (classic)**, then **Generate new token**. Give it a name, set expiry to 90 days, tick the **repo** checkbox, and click **Generate token**. Copy the token and paste it as your password.

---

## Part E: Verify the Upload

1. Go to `https://github.com/YourUsername/bbo-capstone` in your browser.
2. You should see the README displayed on the main page with the full project description.
3. Click through the folders to confirm `notebooks/`, `docs/cards/`, `docs/weekly/`, `docs/reflections/`, and `results/` all contain the expected files.

---

## Part F: Add Files Later

Every time you want to add or update files:

1. Copy the new file into the correct folder on your laptop.
2. Open a terminal in the `bbo-capstone` folder.
3. Run:
   ```
   git add .
   git commit -m "Add week 10 results and final reflection"
   git push
   ```

---

## Part G: Share the Link

Once the repository is public, share the URL in the format:

```
https://github.com/YourUsername/bbo-capstone
```

Peers and facilitators can view all files, download individual documents, and read the README without needing a GitHub account.

---

## Folder Structure Reminder

When your repository is live, the structure visible on GitHub will look like this:

```
bbo-capstone/
├── README.md                          Main project overview
├── requirements.txt                   Library versions
├── notebooks/                         Python notebooks W2-W10
├── docs/
│   ├── cards/
│   │   ├── DATASHEET.md               Data sheet (Gebru et al. 2021)
│   │   ├── MODEL_CARD.md              Model card (Mitchell et al. 2019)
│   │   └── bbo_datasheet_modelcard.docx
│   ├── weekly/                        Per-week strategy notes
│   └── reflections/                   Discussion board submissions
└── results/
    └── performance_self_assessment_v2.xlsx
```

---

*Guide prepared for the BBO Capstone Project. For questions, open an issue in the GitHub repository.*
