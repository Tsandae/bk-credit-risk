# Module 5: finish in 4 steps

## 1. Add files to your GitHub repo (bk-credit-risk)
Unzip `module5_repo_additions.zip` into the root of your local repo folder (it adds `dashboard/`, `scripts/`, `tests/module5/`, `.streamlit/`).
Do NOT commit: `data/curated/loans_curated.parquet`, raw `loan.csv`, `.venv`, `mlflow.db`, `mlruns`, ANON_SALT.
The dashboard needs `dashboard/data/` (about 0.6 MB, already built) and `models/module4/credit_risk_model.joblib`.
Then, in the repo folder:
    git add dashboard scripts tests/module5 .streamlit
    git commit -m "Module 5: dashboard, API demo, tests"
    git push

## 2. Deploy on Streamlit Community Cloud (free)
1. Go to share.streamlit.io and sign in with GitHub.
2. New app -> repository `Tsandae/bk-credit-risk`, branch `main`, main file path `dashboard/app.py`.
3. Advanced settings -> Python 3.12 (match your local version if different).
4. Deploy. Dependencies come from `dashboard/requirements.txt` (scikit-learn 1.9.1 and lightgbm 4.7.0 are pinned because the saved model is version-sensitive).
5. If the model file is over GitHub's 100 MB limit or not in the repo, the app shows a clear error on page 2; fix by committing the .joblib (check its size first).
6. Open the URL in a private window to confirm it loads without login.

## 3. Put the URL in the deck
Slide 15 shows "[Live dashboard URL: add after deployment]". Replace it with your link (Insert > hyperlink). Or ask me and give me the URL and I will rebuild the deck.

## 4. Before submitting
- Personalise slide 24 (AI disclosure) and the "Own insight" row; check it matches your AI Disclosure Form.
- Check slide 2/13 stakeholder and commitment wording against your Module 1 Vision and Ethical AI Charter (I could not read them).
- Confirm references and add the LendingClub source and any Rwanda sources.
- Run `python scripts/api_demo.py --serve` once and keep a screenshot of the output.
