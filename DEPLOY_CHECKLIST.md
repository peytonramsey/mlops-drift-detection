# 🚀 Hugging Face Deployment Checklist

Quick checklist to deploy in 10 minutes!

---

## ✅ Pre-Deployment (Done!)

- [x] Dockerfile configured for Hugging Face (port 7860)
- [x] HTML uses relative URLs (no localhost)
- [x] README_HF.md created
- [x] Deployment guide written

---

## 📋 Your To-Do List

### 1. Create Hugging Face Account (2 min)
- [ ] Go to https://huggingface.co/join
- [ ] Sign up (free)
- [ ] Verify email

### 2. Create Access Token (1 min)
- [ ] Go to https://huggingface.co/settings/tokens
- [ ] Click "New token"
- [ ] Name: `spaces-deploy`
- [ ] Type: Write
- [ ] Copy token and save it somewhere safe

### 3. Create Space (1 min)
- [ ] Go to https://huggingface.co/spaces
- [ ] Click "Create new Space"
- [ ] Name: `loan-default-prediction` (or your choice)
- [ ] SDK: **Docker** ⚠️ Important!
- [ ] Visibility: Public
- [ ] License: MIT
- [ ] Click "Create Space"

### 4. Prepare Repository (2 min)

Open PowerShell/Terminal in your project folder and run:

```powershell
# Backup original README
Copy-Item README.md -Destination README_ORIGINAL.md

# Use Hugging Face README
Copy-Item README_HF.md -Destination README.md

# Commit changes
git add .
git commit -m "Prepare for Hugging Face deployment"
```

### 5. Push to Hugging Face (3 min)

**Replace `YOUR-USERNAME` with your Hugging Face username!**

```powershell
# Add Hugging Face remote
git remote add huggingface https://huggingface.co/spaces/YOUR-USERNAME/loan-default-prediction

# Push to Hugging Face
git push huggingface main:main
```

When prompted:
- **Username**: Your Hugging Face username
- **Password**: Paste the access token you created in step 2

### 6. Wait for Build (5-10 min)
- [ ] Go to your Space URL: `https://huggingface.co/spaces/YOUR-USERNAME/loan-default-prediction`
- [ ] Click "Logs" to watch build progress
- [ ] Wait for "Running" status (green)

### 7. Test Deployment (1 min)
- [ ] Open your Space URL
- [ ] UI loads correctly
- [ ] Make a test prediction
- [ ] Check `/docs` endpoint
- [ ] Check `/drift/status` endpoint

### 8. Share It! (5 min)
- [ ] Update resume with Space URL
- [ ] Add badge to GitHub README
- [ ] Post on LinkedIn
- [ ] Add to portfolio website

---

## 🎯 Your Space URL

After deployment, your project will be live at:

```
https://huggingface.co/spaces/YOUR-USERNAME/loan-default-prediction
```

**Replace `YOUR-USERNAME` with your actual Hugging Face username!**

---

## 📝 Quick Commands Reference

```powershell
# Navigate to project
cd c:\Users\ramse\OneDrive\Documents\mlops-drift-detection

# Prepare README
Copy-Item README_HF.md -Destination README.md

# Add Hugging Face remote (REPLACE YOUR-USERNAME!)
git remote add huggingface https://huggingface.co/spaces/YOUR-USERNAME/loan-default-prediction

# Commit and push
git add .
git commit -m "Deploy to Hugging Face Spaces"
git push huggingface main:main
```

---

## 🆘 Having Issues?

1. **Build failing?** Check the logs in your Space
2. **Model file missing?** Make sure it's committed: `git add models/*.pkl`
3. **Port issues?** We already configured port 7860 ✅
4. **Need help?** Read DEPLOYMENT_GUIDE.md for detailed troubleshooting

---

## 🎉 After Deployment

### Update Your Resume

```
Developed loan default prediction system on 148K samples achieving 88.93%
accuracy and 0.92 ROC-AUC; deployed at https://huggingface.co/spaces/YOUR-USERNAME/loan-default-prediction
```

### GitHub Badge

Add to your GitHub README.md:
```markdown
[![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow)](https://huggingface.co/spaces/YOUR-USERNAME/loan-default-prediction)
```

### LinkedIn Post

```
🚀 Excited to share my ML-powered Loan Default Prediction System!

✅ 88.93% accuracy on 148K applications
✅ Real-time drift detection
✅ Production-ready deployment
✅ <100ms latency

Try it live: https://huggingface.co/spaces/YOUR-USERNAME/loan-default-prediction

#MachineLearning #MLOps #DataScience
```

---

**Ready to deploy? Let's go! 🚀**

Total time: ~10-15 minutes
