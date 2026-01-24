# 🚀 Hugging Face Spaces Deployment Guide

Complete guide to deploy your Loan Default Prediction System to Hugging Face Spaces.

---

## 📋 Prerequisites

1. ✅ Hugging Face account (free) - [Sign up here](https://huggingface.co/join)
2. ✅ Git installed on your computer
3. ✅ Project files ready (you're all set!)

---

## 🎯 Deployment Steps

### Step 1: Create a Hugging Face Space

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click **"Create new Space"**
3. Fill in the details:
   - **Space name**: `loan-default-prediction` (or your choice)
   - **License**: MIT
   - **SDK**: Select **Docker**
   - **Visibility**: Public (recommended for portfolio)
4. Click **"Create Space"**

Your Space URL will be: `https://huggingface.co/spaces/YOUR-USERNAME/loan-default-prediction`

---

### Step 2: Prepare Your Repository

Open terminal/command prompt and run:

```bash
# Navigate to your project
cd c:\Users\ramse\OneDrive\Documents\mlops-drift-detection

# Rename the Hugging Face README
# Windows (Command Prompt):
copy README.md README_ORIGINAL.md
copy README_HF.md README.md

# OR Windows (PowerShell):
Copy-Item README.md -Destination README_ORIGINAL.md
Copy-Item README_HF.md -Destination README.md
```

---

### Step 3: Push to Hugging Face

```bash
# Add Hugging Face remote (replace YOUR-USERNAME)
git remote add huggingface https://huggingface.co/spaces/peytonramsey/loan-default-prediction

# Check current remotes
git remote -v

# Create a new branch for HF deployment (optional but recommended)
git checkout -b huggingface-deploy

# Add all files
git add .

# Commit
git commit -m "Deploy to Hugging Face Spaces"

# Push to Hugging Face
git push huggingface huggingface-deploy:main

# Or if you want to push from your current branch directly:
git push huggingface main:main
```

**Note**: You'll be prompted for credentials:
- **Username**: Your Hugging Face username
- **Password**: Use your **Hugging Face Access Token** (not your password!)
  - Get token at: https://huggingface.co/settings/tokens
  - Click "New token" → Name it "spaces-deploy" → Copy the token

---

### Step 4: Wait for Build

1. Go to your Space URL: `https://huggingface.co/spaces/YOUR-USERNAME/loan-default-prediction`
2. You'll see "Building..." in the top right
3. Build takes **5-10 minutes** (it's building your Docker image)
4. Click "Logs" to watch the build progress
5. When complete, you'll see "Running"

---

### Step 5: Test Your Deployment

1. Open your Space URL
2. You should see your beautiful UI! 🎉
3. Test a prediction with the form
4. Click "API Docs" to view `/docs`
5. Check "Drift Status" endpoint

---

## 🔧 Troubleshooting

### Build Fails?

**Check the logs** in your Space for errors. Common issues:

#### Issue 1: Missing Files
```
Error: models/best_model_real_features.pkl not found
```
**Fix**: Make sure your model files are committed to git:
```bash
git add models/*.pkl
git add models/*.json
git commit -m "Add model files"
git push huggingface main:main
```

#### Issue 2: Port Configuration
**Fix**: Hugging Face uses port 7860 by default. Update your Dockerfile:
```dockerfile
# Change from:
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# To:
EXPOSE 7860
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

#### Issue 3: Large File Size
```
Error: File size exceeds 10GB
```
**Fix**: Your model file (216 MB) is fine, but if you have data files:
```bash
# Don't commit large data files - they're not needed for inference
git rm --cached data/processed_no_indicators/*.csv
git commit -m "Remove large data files"
```

---

## 📝 After Deployment

### Update Your Resume

```
- Developed loan default prediction system on 148K samples achieving 88.93%
  accuracy, 0.86 F1-score, and 0.92 ROC-AUC using Random Forest and XGBoost;
  deployed at https://huggingface.co/spaces/YOUR-USERNAME/loan-default-prediction
  with <100ms latency.
```

### Update GitHub README

Add a badge to your GitHub README.md:
```markdown
[![Hugging Face Space](https://img.shields.io/badge/🤗-Hugging%20Face-yellow)](https://huggingface.co/spaces/YOUR-USERNAME/loan-default-prediction)

## 🚀 Live Demo
Try the live demo: [Hugging Face Space](https://huggingface.co/spaces/YOUR-USERNAME/loan-default-prediction)
```

### Add to LinkedIn

Post about it:
```
🚀 Excited to share my latest project: ML-powered Loan Default Prediction System!

✅ 88.93% accuracy on 148K applications
✅ Real-time drift detection
✅ Production-ready FastAPI deployment
✅ <100ms inference latency

Try it live: https://huggingface.co/spaces/YOUR-USERNAME/loan-default-prediction

Built with: scikit-learn, PyTorch, FastAPI, MLflow, Docker

#MachineLearning #MLOps #DataScience
```

---

## 🔄 Updating Your Deployment

Made changes? Push updates:

```bash
git add .
git commit -m "Update model/features"
git push huggingface main:main
```

Hugging Face will automatically rebuild and redeploy!

---

## 💡 Pro Tips

1. **Add Analytics**: Hugging Face provides built-in analytics for your Space
2. **Custom Domain**: You can add a custom domain in Space settings
3. **Duplicate**: Users can "duplicate" your Space to try modifications
4. **Embed**: You can embed your Space in other websites
5. **Monitor**: Check logs regularly for errors or issues

---

## 🎯 Alternative: Quick Deploy Method

If you want to skip Git and deploy quickly:

1. Create Space on Hugging Face
2. Click "Files and versions"
3. Click "Add file" → "Upload files"
4. Drag and drop these files:
   - `Dockerfile`
   - `requirements.txt`
   - `README_HF.md` (rename to `README.md`)
   - Entire `src/` folder
   - Entire `models/` folder
5. Click "Commit"

Build will start automatically!

---

## ✅ Verification Checklist

- [ ] Space created on Hugging Face
- [ ] README_HF.md renamed to README.md
- [ ] Code pushed to Hugging Face
- [ ] Build completed successfully
- [ ] UI loads at Space URL
- [ ] Predictions work correctly
- [ ] API docs accessible at `/docs`
- [ ] Drift status endpoint works
- [ ] URL added to resume
- [ ] GitHub README updated with badge
- [ ] LinkedIn post created

---

## 🆘 Need Help?

- Hugging Face Discord: https://huggingface.co/join/discord
- Documentation: https://huggingface.co/docs/hub/spaces-sdks-docker
- My GitHub Issues: https://github.com/peytonramsey/mlops-drift-detection/issues

---

## 🎉 Success!

Once deployed, your project will be accessible 24/7 at:
```
https://huggingface.co/spaces/YOUR-USERNAME/loan-default-prediction
```

**This is a HUGE boost for your job applications!** 🚀

Recruiters can now test your ML system with **zero setup** required.
