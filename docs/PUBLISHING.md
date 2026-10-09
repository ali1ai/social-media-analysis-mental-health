# Putting this project on GitHub (no password needed)

GitHub no longer accepts your account password for git operations; it was retired in 2021. If anything asks you for one, use one of the routes below instead. Nothing in the notebook itself ever needs your GitHub details.

## Option A: website only (simplest)

1. Sign in at github.com, click **+ → New repository**, name it `socialmediamind`, choose **Public**, and leave "Add a README" **unticked**. Click **Create repository**.
2. On the empty repo page, click **uploading an existing file**.
3. Unzip `socialmediamind.zip` on your computer, open the `socialmediamind` folder, select **everything inside it**, and drag it onto the page. Then click **Commit changes**.
   - **macOS:** the `.github` folder (which holds the CI workflow) is hidden in Finder. Press **Cmd + Shift + .** to show it, then include it in the drag.
   - If the browser won't upload folders, use Option B.

## Option B: GitHub Desktop (no terminal)

1. Install [GitHub Desktop](https://desktop.github.com) and sign in. It opens your browser, so you never type a password into the app.
2. **File → Add local repository** → pick the unzipped `socialmediamind` folder → **create a repository** when prompted.
3. Click **Publish repository**, untick "Keep this code private" if you want it public, and confirm.

## Option C: command line (browser sign-in via the GitHub CLI)

```bash
# one-time: install the GitHub CLI (https://cli.github.com), then
gh auth login            # choose GitHub.com → HTTPS → "Login with a web browser"

cd socialmediamind
git init -b main
git add .
git commit -m "SocialMediaMind v0.2"
gh repo create socialmediamind --public --source=. --push
```

## After publishing (optional, about 2 minutes)

- **Check CI:** open the **Actions** tab. The workflow runs the tests and executes the notebook on synthetic data, so a green tick there is good portfolio evidence.
- **Add badges:** paste these under the title in `README.md`, replacing `YOUR-NAME` with your GitHub username:

  ```markdown
  [![CI](https://github.com/YOUR-NAME/socialmediamind/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR-NAME/socialmediamind/actions/workflows/ci.yml)
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-NAME/socialmediamind/blob/main/notebooks/SocialMediaMind.ipynb)
  ```

- **Add real results:** run the notebook in Colab, then download `reports/metrics.json`, `reports/MODEL_CARD.md` and `reports/figures/*.png` from Colab's file panel. Upload them to the repo's `reports/` folder and fill in the README results table.

## Never commit

- the dataset CSV (licence and sensitive content; already excluded by `.gitignore`)
- `kaggle.json`, tokens, or any password
- notebook outputs that contain raw post text (outputs are hidden by default, and the committed notebook has none)
