1.	Business Vertical and Problem Context

Vertical: Banking retail and digital lending. Core process: Loan application assessment and credit risk management. The main business question is: How can Bank of Kigali (BK) identify applicants who are more likely to default and support faster, more consistent lending decisions while keeping credit accessible and fair? This project will use three types of analytics. Descriptive analytics will identify applicant and portfolio patterns. Predictive analytics will estimate the probability of loan default. Prescriptive analytics will recommend different levels of human review based on risk, expected cost and available review capacity.

Organizational Context

Bank of Kigali needs to manage credit risk and at the same time continue to grow its digital lending services. BK Group reported a net profit of RWF 110.1 billion for FY2025 and an NPL ratio of 2.9% at December 2025 (BK Group Plc, 2026). The NPL ratio later increased to 4.8% in March 2026 (BK Group Plc, 2026) and was reported at 6.5% for the half-year (KT Press, 2026). At the same time, BK is expanding digital lending through BK QUICK+, a not secured personal loan that cangouptoRWF50M and it can be requested through the bank's digital channels, and can be received in just few hours after request (KT Press, 2025). Because these loans are not secured and decisions need to be fast, there is a need for a consistent way to identify higher risk applications. The project therefore supports three business objectives like protecting asset quality, grow digital lending safely, and improve responsible access to credit.

Dataset: The project will use the LendingClub loan dataset, I found on Kaggle. It contains 887,379 loans issued between 2007 and 2015 and 74 columns. After removing unresolved loan statuses, approximately 256,939 labelled loans remain, with about 18% classified as bad loans. This is more that the assignment asked for (5,000 +records). The dataset have unsecured US consumer loans, which makes it useful for developing the project idea. But, the dataset is not from Bank of Kigali. Which means the findings will not be presented as BK customer results. Any future use with BK customers would require current BK data, local validation and approval.

2.	Vision,Goals and Ethical Principles

Vision Statement: The vision is to develop an explainable and audited credit risk solution that can help BK's retail and digital lending teams make faster and more consistent decisions, which can reduce losses from loan defaults and support responsible lending. The system will support credit officers rather than replace them. It will not approve, reject or price a loan by itself.
Measurable Goals, The project will aim to:
1.	Themodelshouldreachatleast0.70ROC-AUC and0.25KSwhenitistestedonnewerdata.
2.	Find high risk customers:at 40 % of them who default should be found in the highest risk20%of applicants.
3.	The predicted default rate should be in 10% of the actual default rate in each risk group.
4.	Reduce loan losses: The new review process should reduce expected loan at least 10% compared with using one risk cut off, at the same time keep the same amount of review work.
5.	Explain the result:for every prediction,the system should show the three main reasons for the result in simple language (Lundberg & Lee, 2017).
6.	Keep the model fair and repeatable:The model should meet the fairness goals,and the same result should be possible to reproduce using the documented process.
 
Ethical AI Vision: The project will follow five main principles which are fairness, transparency, accountability, privacy and human review. A risks core represents statistical risk; and it is not proof that a customer is unreliable. Of course, credit officers will remain responsible for the final decision.

Fairness Objectives: TheLendingClub dataset does not have gender, age or race. Therefore, fairness will be tested using groups that are available like income bands, home ownership, employment length and region.

The project will check if the model different groups fairly. The difference in correct and incorrect predictions between groups should be no more than 5 percentage points when possible. The project will also check differences in approval rates using 0.80 as a warning level. The model should give identical risk results for different groups. Fields like as emp_title, desc and zip_code will not be used because they may give unfair results. Other variables will also be checked to see if they may indirectly create unfairness. Fairness results will be reported for every new version of the model.Before using the model in real Bank of Kigali system, fairness based on gender and age must be tested using suitable BK data.

Ethical AI Charter

1.	The model is going to support credit officers and does not make the final decision.
2.	No applicant will be refused only because of a model score.
3.	Predictions and their main reasons will be documented.
4.	Personal data will be protected and used according to privacy rules and bank policies.
5.	The model must be tested for fairness and checked by an independent person before it issued.
6.	The model will not be released if significant unexplained fairness or performance problems remain.
7.	The limitations of the LendingClub dataset will always be disclosed.

3.	Users,Stakeholders and Ethical Oversight/Users and Stakeholders
Group	Role and responsibility
Credit officers	Use the risks or and reasons to help make the final loan decision and record any changes they make.
Bank managers	Check loan growth, the team’s review capacity,and loan performance.
Risk team	Check default risk,model performance and changes in risk levels.
Data analysts/data scientists	Prepare the data,build the models,test fairness and maintain the system.
Senior Management	Set the bank‘s risk limit, check business value and approve major decisions.
Internal audit	Check the data,model,documents and controls independently.
Internal Audit,Legal and DPO	Review responsible lending, privacy and regulatory requirements.
IT teams	Support data access,security,system setup and technical controls.
National Bank of Rwanda	Provides rules and requirements that the bank must follow.
Applicants/customers	Are affected by loan decisions and should be treated fairly and clearly.
Communication: Each group will receive information based on their role. Credit officer will get clear reasons for each applicant’s risk. Managers will get loan performance reports. Risk teams will get information about model performance and change in risk. Senior management will get reports about risk, fairness and business results.

Ethics Oversight Body: The proposed BK AI and Credit Model Governance Committee would include representatives from credit risk,retail and digital banking,compliance,the data Protection Officer,legal
 
and model validation. It will review the project at important stages and it can stop the project if the model do not meet the needed standards for performance, fairness, privacy or governance.

4.	Analytics Product/Solution
The proposed product is the BK Credit Risk Intelligence Dashboard. It will use loan application data to give a quick, clear and consistent view of credit risk.
Main Features and Outputs: The dashboard will calculate a probability of default; put applicants into low, medium or higher risk groups; show the top reasons behind each prediction using SHAP; show overall loan risk information; recommend normal review, additional documents or extra human review; allow users to compare different risk levels and review capacity; check model performance and fairness; Keep an record of important decisions. The predictive part will compare a logistic regression with a LightGBM(Ke et al., 2017);. The prescriptive part will consider loan losses, unnecessary reviews and the number of applications staff can review. The credit officer will make the final decision.
Data Needs: The model will use information available when the loan application is made such as, loan amount and term; income;employment length;home ownership;verification status; loan purpose;debt-to-income ratio; previous delinquencies and credit inquiries; credit-line and revolving-balance information. Information collected after the loan is given, like total_pymnt, out_prncp, recoveries and last_pymnt_d will not be used because it could give the model information it would not have at the time of the loan decision.
TechnicalStackMatrix
Project phase	Tools
Data ngestion nd preparation	Python,pandas,KaggleAPI,Parquet
Data validation	Prefect,GreatExpectations
Exploration	Jupyter/VSCode,pandas,NumPy,Matplotlib
Modelling	scikit-learn,LightGBM,Optuna
Explainabilityandfairness	SHAP, Fairlearn
Versioningandtracking	Git,GitHub,DVC,MLflow
Dashboard/deployment	Streamlit,Docker,GoogleCloudRun
ProposedGitHubRepositoryStructure
bk-credit-risk/
├──README.mdLICENSEpyproject.toml
├──dvc.yamlDockerfile
├──.github/workflows/ci.yml
├──docs/
│	├──vision_document.pdf
│	├──ethics/(charter.md,risk_register.md)
│	├──datasheet.mdmodel_card.md
│	└──decisions/(ADR-001-*.md)
├──data/	#DVC-tracked,git-ignored
│	├──raw/interim/processed/	├──notebooks/
│	├──01_eda.ipynb02_features.ipynb
│	└──03_model.ipynb04_fairness.ipynb
├──src/
│	├──ingestion/features/models/
│	├──fairness/	optimisation/
│	└──app/(Streamlitdashboard)
├──pipelines/(Prefect flows)
├──models/(MLflowartefacts, DVC)
└──tests/(unit,data-validation, fairness)

5.	Requirements,Risks and Constraints
Functional Requirement: The system should first load and validate the dataset; handle missing data and record data problems; prepare the data in repeatable way ; train and compare at least two models; predict default risk; explain predictions using SHAP; Test fairness; create risk groups and review recommendations; show results through a dashboard; and keep record of data and model versions.
 
Non-Functional Requirements: The system should be easy to repeat, secure, easy to track and understand. It should give results quickly, control who can access the data, keep a record of where the data came from and allow others to check the model.
Main Constraints: The dataset is US lending data from 2007–2015, not BK data. It is also large, so the project may begin with a small sample and Parquet files to make the data easier to handle. The dataset does not include protected information like gender and age, which will limits fairness testing. The data also have has some problems, like loans that are still being paid and an unequal number of good and bad loans. The issue needs to be considered when testing the model.
Ethical Risk Register



Risk	Mitigation
Model performs differently across groups	Monitor group performance and fairness measures.
Good customers are incorrectly classified as high risk	Use threshold/cost analysis and human review.
Model becomes the final lending decision	Need credit officer approval.
Variables act as proxies for protected characteristics	Exclude high-risk fields and test remaining variables for proxy effects.
LendingClub data does not represent BK customers	Clearly state the limitation and require local validation before use.
Data leakage produces unrealistic results	Use application-time variables and out-of-time testing.
Privacy or inappropriate data use	Apply data minimization,access controls and privacy review.
Regulatory requirements are missed	Need Compliance,Legal and independent model validation.
SHAP explanations are treated as causes	Clearly state that SHAP shows model associations,not causation.


6.	Quality,Documentation and Governance Standards

Documentation and Quality: The project will keep a dataset datasheet, model card, architecture decision records and a record showing how business needs connect to the data, models, tests and dashboards. Data-validation and unit tests will be used to check quality. Code changes will be reviewed through GitHub, and important experiments will use the same settings so that the results can be repeated(Gebru etal.,2021).

The report and project will follow professional reporting standards, including a proper citations and references. The project will also use the NIST AI Risk Management Framework as a general guide for managing AI risk (NIST, 2023).

Versioning Strategy:Code will be managed using Git and GitHub, with branches and pull-request review. DVC will be used to track data versions, and model versions will be tracked using MLflow and semantic versioning. This will help with each result to be connected to the code, data and model version used.

Data Governance: Data access will follow the least privilege principle, which means people will only get access they need. Data will only be used for the mentioned credit risk project, and personal information will be kept to minimum and protected. Raw data will not be uploaded directly to GitHub. Data will be kept according to privacy rules, and for future BK data, Rwanda's personal data protection law and bank policy will be followed. The LendingClub results will always be kept separate from assumptions about BK customers. The solution will not be use in the real BK system until it has current BK data, local testing, fairness testing, regulatory review and approval.

References
BK Group Plc. (2026). Full year 2025 results. Rwanda Stock Exchange. file:///C:/Users/User/Downloads/1774982953_dc9a45d13e603aa2897c.pdf
BK Group Plc. (2026). Quarter 1 2026 results. Rwanda Stock Exchange. file:///C:/Users/User/Downloads/1780057884_d9bb84d731ad546ce821.pdf
Bird, S., Dudík, M., Edgar, R., Horn, B., Lutz, R., Milan, V., Sameki, M., Wallach, H., & Walker, K. (2020). Fairlearn: A toolkit for assessing and improving fairness in AI. Microsoft. https://www.microsoft.com/en-us/research/wp-content/uploads/2020/05/Fairlearn_WhitePaper-2020-09-22.pdf
Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé, H., & Crawford, K. (2021). Datasheets for datasets. Communications of the ACM, Vol64(12), 86–92. https://dl.acm.org/doi/epdf/10.1145/3458723
Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. https://arxiv.org/pdf/1610.02413
Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. https://proceedings.neurips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf
KT Press Staff Writer. (2025). Get a Rwf 50m Bank of Kigali loan with 15 hours. KT Press. https://www.ktpress.rw/2025/03/get-a-rwf-50m-bank-of-kigali-loan-with-15-hours/
KT Press Team. (2026, September). BK Group PLC shrinks its corporate footprint to focus on banking and investment. KT Press. https://www.ktpress.rw/2026/09/bk-group-plc-shrinks-its-corporate-footprint-to-focus-on-banking-and-investment/
R. G. (2021). Credit Risk Analysis .Kaggle. https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset
Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. file:///C:/Users/User/Downloads/SHAP_camera_ready2.pdf
Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, K. (2019). Model cards for model reporting. Proceedings of the Conference on Fairness, Accountability, and Transparency, 220–229. https://dl.acm.org/doi/epdf/10.1145/3287560.3287596
National Institute of Standards and Technology. (2023). Artificial intelligence risk management framework (AI RMF 1.0). https://nvlpubs.nist.gov/nistpubs/ai/nist.ai.100-1.pdf
Republic of Rwanda. (2021). Law No 058/2021 of 13/10/2021 relating to the protection of personal data and privacy. Official Gazette. https://cyber.gov.rw/fileadmin/user_upload/NCSA/Documents/Laws/OG_Special_of_15.10.2021_Amakuru_bwite.pdf


