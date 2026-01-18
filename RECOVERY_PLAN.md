# Git Repository Recovery Plan

## Problem
The local repository has diverged from `origin/main` with several large commits (approx. 2GB) containing binary datasets/images. These commits cannot be pushed to GitHub due to size limits. The goal is to keep the project restructuring and code changes but **exclude** the large data files from the repository history, keeping them only locally.

## Prerequisite
**CRITICAL**: Before executing any commands, create a full backup of the `BP-face-recognition` folder to a separate location (e.g., an external drive or `../BP-face-recognition-BACKUP`).

## Execution Steps

### 1. Reset to Remote State
We will undo the local commits that introduced the bloat. This command resets the git index to match the remote `main` branch but **leaves all your files on disk untouched**.
```powershell
git reset --mixed origin/main
```
*Effect:* All your recent work (refactoring `src`, moving folders) will appear as "Unstaged Changes".

### 2. Configure Ignore Rules
Ensure `.gitignore` blocks the large data folders from being re-added.
*   Verify `.gitignore` contains:
    ```gitignore
    data/datasets/
    data/logs/
    *.jpg
    *.png
    *.h5
    *.keras
    wheels/
    ```

### 3. Selective Staging
Instead of `git add .`, we will add files explicitly to avoid picking up the data folder.

**Stage Configuration & Tools:**
```powershell
git add pyproject.toml uv.lock noxfile.py Makefile .gitignore
```

**Stage Source Code:**
```powershell
git add src/
```
*(Note: Since we are adding `src/`, verify no binary files are inside it using `git status`)*

**Stage Documentation:**
```powershell
git add README.md TODO.md PROGRESS.md IMPLEMENTATION_CONTEXT.md notebooks/README.md
```

**Stage Specific Data Files (Metadata only):**
```powershell
git add data/*.csv data/*.md
```

### 4. Verify Index
Check what is about to be committed.
```powershell
git status
```
*   **Green (Staged):** Should be python files, markdown, toml, etc.
*   **Red (Unstaged):** Should be the massive list of images in `data/datasets/`. **Do not add these.**

### 5. Commit
Create a clean commit.
```powershell
git commit -m "refactor: Modernize project structure and exclude large datasets

- Migrated source to src/bp_face_recognition package layout
- Configured uv and nox for development workflow
- Refactored notebooks to scripts
- Updated documentation
- NOTE: Datasets are now ignored and strictly local"
```

### 6. Push
Since we reset to `origin/main`, we are technically just adding a new commit on top of it. However, if `origin/main` has moved since our last fetch, we might need to pull first.
```powershell
git pull --rebase
git push origin main
```
