# Pre-Push Checklist

Run these checks **BEFORE** pushing to GitHub to ensure no sensitive data or large files are included.

## 1. Check for Large Files (>100MB)

GitHub will **reject** files larger than 100MB!

```bash
# Find files larger than 100MB
find . -type f -size +100M 2>/dev/null | grep -v ".git"
```

**Expected output**: None (all large files should be in .gitignore)

**If you see files**:
- Add them to `.gitignore`
- Run `git rm --cached <filename>` if already staged

## 2. Check for Database Files

Database files may contain sensitive prediction data!

```bash
# Find database files
find . -name "*.db" -o -name "*.sqlite" -o -name "*.sqlite3" 2>/dev/null | grep -v ".git"
```

**Expected output**: Files should exist but be in .gitignore

**Verify they're excluded**:
```bash
git status
# Should NOT see .db files listed
```

## 3. Check for Secrets/API Keys

```bash
# Search for potential secrets
grep -r "password\|secret\|api_key\|token\|private_key" --include="*.py" --include="*.env" --include="*.yml" . | grep -v ".git"
```

**Expected output**: None (or only in comments/variable names)

**If you find real secrets**:
- Remove them immediately
- Use environment variables instead
- Add files to .gitignore

## 4. Verify .gitignore is Working

```bash
# Check what will be committed
git status

# Check if large files are excluded
git ls-files | xargs du -h | sort -rh | head -20
```

**Files that SHOULD appear**:
- ✅ `models/scaler_no_indicators.pkl` (~3KB)
- ✅ `models/baseline_stats.json` (~28KB)
- ✅ `models/feature_names.json` (~1KB)
- ✅ Source code (`src/**/*.py`)
- ✅ Documentation (`*.md`)
- ✅ Config files (`Dockerfile`, `requirements.txt`)

**Files that should NOT appear**:
- ❌ `models/best_model_real_features.pkl` (206MB)
- ❌ `data/processed_no_indicators/*.csv` (>100MB)
- ❌ `predictions.db`
- ❌ `mlflow.db`
- ❌ `__pycache__/`
- ❌ `*.pyc`

## 5. Check Commit Size

```bash
# See what will be committed (with sizes)
git add .
git status --short | cut -c4- | xargs -I {} stat -f "%z {}" {} 2>/dev/null || stat -c "%s %n" {} 2>/dev/null
```

**Total should be < 100MB** (ideally < 50MB)

## 6. Test Git Add (Dry Run)

```bash
# See what would be added
git add --dry-run .

# Actually add files
git add .

# Verify nothing large was added
git diff --cached --stat
```

## 7. Verify Docker Build Still Works

If you excluded model files, Docker build might fail!

```bash
# Test the build
docker-compose build

# If it fails, update Dockerfile to handle missing files
```

**Fix if needed**:
```dockerfile
# Make model loading graceful
COPY models/*.pkl models/*.json ./models/ || true
```

## 8. Check for PII in Sample Data

```bash
# Check if test files contain real PII
grep -r "SSN\|social.*security\|credit.*card" test*.py
```

**Expected output**: None

## 9. Final Git Status Check

```bash
git status
```

### Should see:
- Modified: `.gitignore`
- New files: `README.md`, `Dockerfile`, `requirements.txt`, etc.
- **Not** see: Large `.pkl` files, `.db` files, `__pycache__`

### Should NOT see:
```
models/best_model_real_features.pkl (206MB)  ❌
data/processed_no_indicators/X_train.csv     ❌
predictions.db                                ❌
mlflow.db                                     ❌
```

## 10. Test Clean Clone

This simulates what others will get when they clone your repo:

```bash
# Create test directory
mkdir /tmp/test-clone
cd /tmp/test-clone

# Clone the repo (after pushing)
git clone https://github.com/YOUR_USERNAME/mlops-drift-detection.git
cd mlops-drift-detection

# Try to install
pip install -r requirements.txt

# Check what's missing
ls -lh models/
ls -lh data/processed_no_indicators/
```

## Common Issues & Fixes

### Issue 1: "File exceeds GitHub's 100MB limit"

**Cause**: Large file not in .gitignore

**Fix**:
```bash
# Remove from staging
git rm --cached models/best_model_real_features.pkl

# Add to .gitignore
echo "models/best_model_real_features.pkl" >> .gitignore

# Commit the change
git add .gitignore
git commit -m "Exclude large model file"
```

### Issue 2: "Database file is being committed"

**Fix**:
```bash
git rm --cached predictions.db
echo "*.db" >> .gitignore
git add .gitignore
git commit -m "Exclude database files"
```

### Issue 3: "Push rejected due to file size"

**Fix**:
```bash
# If already committed large file, remove from history
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch models/best_model_real_features.pkl" \
  --prune-empty --tag-name-filter cat -- --all

# Force push (WARNING: only if you haven't shared the repo yet)
git push origin --force --all
```

**Better**: Start fresh
```bash
# Remove .git directory
rm -rf .git

# Re-initialize
git init
git add .
git commit -m "Initial commit"
```

## Ready to Push Checklist

Before running `git push`:

- [ ] Ran `find . -type f -size +100M` - No results
- [ ] Ran `git status` - No .db or large .pkl files
- [ ] Ran `git ls-files | xargs du -h | sort -rh | head -10` - Largest file < 30MB
- [ ] Customized README.md with your info
- [ ] Updated SECURITY.md contact email
- [ ] Verified Dockerfile builds successfully
- [ ] No secrets or API keys in code
- [ ] All test files use dummy/synthetic data
- [ ] Created GitHub repository
- [ ] Ready to push!

## Push Commands

```bash
# Final commit
git add .
git commit -m "Initial commit: MLOps Loan Default Prediction System"

# Add remote (replace with your username)
git remote add origin https://github.com/YOUR_USERNAME/mlops-drift-detection.git

# Push
git branch -M main
git push -u origin main
```

## After Pushing

1. **Verify on GitHub**:
   - Check file sizes in GitHub UI
   - Verify large files are NOT there
   - Check README renders correctly

2. **Add Topics**:
   - machine-learning, mlops, fastapi, data-drift, python, docker

3. **Enable Discussions** (optional)
   - Settings → Features → Discussions

4. **Update Resume** with project link!

## If Something Goes Wrong

**Accidentally pushed sensitive data?**
1. Delete the repository immediately
2. Rotate any exposed credentials
3. Start fresh with cleaned history

**Need help?**
- GitHub Support: https://support.github.com
- Git Large File Storage (LFS): https://git-lfs.github.com
- BFG Repo-Cleaner: https://rtyley.github.io/bfg-repo-cleaner/
