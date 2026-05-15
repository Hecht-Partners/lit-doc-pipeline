# Git Workflow with `gh` CLI

**Repo:** https://github.com/Hecht-Partners/lit-doc-pipeline
**Remote:** `origin` (HTTPS)
**Default branch:** `main`

## Setup

```bash
# Authenticate (one-time)
gh auth login

# Clone the repo
gh repo clone Hecht-Partners/lit-doc-pipeline
```

## Daily Workflow

```bash
# Check status
git status
git log --oneline -10

# Stage, commit, push
git add <files>
git commit -m "Description of changes"
git push origin main
```

## Branches & Pull Requests

```bash
# Create a feature branch
git checkout -b feature/my-feature

# Push branch and create PR
git push -u origin feature/my-feature
gh pr create --title "Add my feature" --body "Description of changes"

# List open PRs
gh pr list

# View a specific PR
gh pr view <number>

# Check out someone else's PR locally
gh pr checkout <number>

# Merge a PR
gh pr merge <number> --merge
```

## Issues

```bash
# Create an issue
gh issue create --title "Bug: description" --body "Details"

# List open issues
gh issue list

# View an issue
gh issue view <number>

# Close an issue
gh issue close <number>
```

## Repo Management

```bash
# View repo info
gh repo view

# Edit repo settings
gh repo edit --description "New description"
gh repo edit --visibility private  # or public

# View recent activity
gh repo view --web  # opens in browser
```

## Releases

```bash
# Create a release
gh release create v1.0.0 --title "v1.0.0" --notes "Release notes"

# List releases
gh release list

# Download release assets
gh release download v1.0.0
```

## Useful Shortcuts

```bash
# Open repo in browser
gh browse

# Open a specific file in browser
gh browse <file>

# View CI/workflow status
gh run list
gh run view <run-id>

# Sync fork with upstream (if applicable)
gh repo sync
```

## Notes

- This repo uses **HTTPS** for the remote (not SSH). The `gh` CLI handles auth automatically via `gh auth login`.
- All work currently happens on `main`. Create feature branches for larger changes.
- Run tests before pushing: `.venv/bin/python -m pytest tests/ -v`
