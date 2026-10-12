1.	Business Verticaland Problem Context

Vertical: Banking retail and digital lending. Core process: Loan application assessment and credit risk management. The main business question is: How can Bank of Kigali (BK) identify applicants who are more likely to default and support faster, more consistent lending decisions while keeping credit accessible and fair? This project will use three types of analytics. Descriptive analytics will identify applicant and portfolio patterns. Predictive analytics will estimate the probability of loan default. Prescriptive analytics will recommend different levels of human review based on risk, expected cost and available review capacity.

Organizational Context

Bank of Kigali needs to manage credit risk and at the same time continue to grow its digital lending services. BK Group reported a net profit of RWF 110.1 billion for FY2025 and an NPL ratio of 2.9% at December 2025 (BK Group Plc, 2026).The NPL ratio later increased to 4.8% in March 2026 (BK Group Plc, 2026) and was reported at 6.5% for the half-year (KT Press, 2026). At the same time, BK is expanding digital lending through BK QUICK+, a not secured personal loan that cangouptoRWF50M and it can be requested through the bank's digital channels, and can be received in just few hours after request (KT Press, 2025). Because these loans are not secured and decisions need to be fast, there is a need for a consistent way to identify higher risk applications. The project therefore supports three business objectives like protecting asset quality, grow digital lending safely, and improve responsible access to credit.

Dataset: The project will use the LendingClub loan dataset, I found on Kaggle. It contains 887,379 loans issued between 2007 and 2015 and 74 columns. After removing unresolved loan statuses, approximately 256,939 labelled loans remain, with about 18% classified as bad loans. This is more that the assignment asked for (5,000 +records).The dataset have unsecured US consumer loans, which makes it useful for developing the project idea. But, the dataset is not from Bank of Kigali. Which means the findings will not be presented as BK customer results. Any future use with BK customers would require current BK data, local validation and approval.


2.	Vision,Goals and Ethical Principles

Vision Statement: The vision is to develop an explainable and audited credit risk solution that can help BK's retail and digital lending teams make faster and more consistent decisions, which can reduce losses from loan defaults and support responsible lending. The system will support credit officers rather than replace them. It will not approve, reject or price a loan by itself.
Measurable Goals, The project will aim to:
1.	Themodelshouldreachatleast0.70ROC-AUC and0.25KSwhenitistestedonnewerdata.
2.	Find high risk customers:at 40 % of them who default should be found inthe highest risk20%of applicants.
3.	Thepredicted defaultrateshouldbein10%oftheactualdefaultratein eachriskgroup.
4.	Reduce loan losses: The new review process should reduce expected loan at least 10% compared with using one risk cut off, at the same time keep the same amount ofreview work.
5.	Explain the result:forevery prediction,the system shouldshow the threemain reasonsforthe result in simple language (Lundberg & Lee, 2017).
6.	Keepthemodelfairandrepeatable:Themodelshouldmeetthefairnessgoals,andthesame result should be possible to reproduce using the documented process.
 
Ethical AI Vision: The project will follow five main principles which are fairness, transparency, accountability, privacyandhuman review. Ariskscorerepresents statisticalrisk; andit is not proof that a customer is unreliable. Of course, credit officers will remain responsible for the final decision.

Fairness Objectives: TheLendingClub dataset does not have gender, ageor race. Therefore, fairness will be tested using groups that are available like income bands, home ownership, employment length and region.

The project will check if the model different groups fairly. The difference in correct and incorrect predictions between groups should be no more than 5 percentage points when possible. The project will also check differences in approval rates using 0.80 as a warning level. The model should give identical risk results for different groups. Fields like as emp_title, desc and zip_code will not beused becausethey may give unfair results. Other variables will also be checked to see if they may indirectly create unfairness. Fairness results will be reported forevery new version of the model.Before using the modelin real Bank of Kigali system, fairness based on gender and age must be tested using suitable BK data.

EthicalAICharter

1.	Themodelisgoingtosupportcreditofficersanddoesnotmakethefinaldecision.
2.	Noapplicant willberefusedonlybecauseofamodelscore.
3.	Predictionsandtheir mainreasonswillbedocumented.
4.	Personaldatawillbeprotectedandusedaccordingtoprivacyrulesandbankpolicies.
5.	Themodelmust betestedforfairness andcheckedbyanindependent personbeforeit isused.
6.	Themodelwillnotbereleasedifsignificantunexplainedfairnessorperformanceproblems remain.
7.	ThelimitationsoftheLendingClubdatasetwillalwaysbedisclosed.

3.	Users,StakeholdersandEthicalOversight/UsersandStakeholders
Group	Roleandresponsibility
Creditofficers	Usetheriskscoreandreasonstohelpmakethefinalloandecisionandrecordanychanges
they make.
Bankmanagers	Checkloangrowth, theteam’sreviewcapacity,andloan performance.
Riskteam	Checkdefaultrisk,modelperformanceandchangesinrisklevels.
Dataanalysts/datascientists	Preparethedata,buildthemodels,testfairnessandmaintainthesystem.
SeniorManagement	Setthebank‘srisklimit, checkbusinessvalueandapprovemajordecisions.
Internalaudit	Checkthedata,model,documentsandcontrolsindependently.
InternalAudit,LegalandDPO	Reviewresponsiblelending, privacyandregulatoryrequirements.
ITteams	Supportdataaccess,security,systemsetupandtechnicalcontrols.
NationalBankofRwanda	Providesrulesandrequirementsthatthebankmust follow.
Applicants/customers	Areaffectedbyloan decisions andshouldbetreatedfairlyandclearly.
Communication: Each group will receive information based on their role. Credit officer will get clear reasons for each applicant’s risk. Managers will get loan performance reports. Risk teams will get information about model performance and change in risk. Senior management will get reports about risk, fairness and business results.

Ethics Oversight Body: The proposed BK AI and Credit Model Governance Committee would include representativesfromcreditrisk,retailanddigitalbanking,compliance,thedataProtectionOfficer,legal
 
and modelvalidation. It will reviewtheproject atimportant stages andit can stop theproject if the model do not meet the needed standards for performance, fairness, privacy or governance.

4.	AnalyticsProduct/Solution
The proposed product is the BK Credit Risk Intelligence Dashboard. It will use loan application data to give a quick, clear and consistent view of credit risk.
Main Features and Outputs: The dashboard will calculate aprobability of default; putapplicants into low, medium or higher risk groups; show the top reasons behind each prediction using SHAP; show overall loan risk information; recommend normal review, additional documents or extra human review; allow users to compare different risk levels and review capacity; check model performance and fairness; Keep an record of important decisions. The predictive part will compare a logistic regression with a LightGBM(Ke et al., 2017);. The prescriptive part will consider loan losses, unnecessaryreviews and the number of applications staff canreview. The credit officer will make the final decision.
Data Needs: The model will use information available when the loan application is made such as, loan amount and term; income;employment length;home ownership;verification status; loan purpose;debt-to-income ratio; previous delinquencies and credit inquiries; credit-line and revolving-balance information. Information collected after the loan is given, like total_pymnt, out_prncp, recoveries and last_pymnt_d will not be used because it could give the model information it would nothave at the timeof the loan decision.
TechnicalStackMatrix
Projectphase	Tools
Dataingestionandpreparation	Python,pandas,KaggleAPI,Parquet
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

5.	Requirements,RisksandConstraints
Functional Requirement: The system should first load and validate the dataset; handle missing data and record data problems; prepare the data in repeatable way ; train and compare at least two models; predict default risk; explain predictions using SHAP; Test fairness; create risk groups and review recommendations; show results through a dashboard; and keep record of data and model versions.
 
Non-Functional Requirements: Thesystemshould be easy to repeat, secure, easy to track and understand. It should giveresults quickly, control whocanaccess thedata, keep a record of wherethedata camefrom and allow others to check the model.
Main Constraints: The dataset is US lending data from 2007–2015, not BK data. It is also large, so the project may begin with a small sample and Parquet files to make the data easier to handle. The dataset does not include protected information like gender and age, which will limits fairness testing. The data also have has some problems, like loans that are still being paid and an unequal number of good and bad loans. The issue needs to be considered when testing the model.
EthicalRiskRegister



Risk	Mitigation
Modelperformsdifferentlyacross groups	Monitorgroupperformanceandfairnessmeasures.
Goodcustomersareincorrectlyclassifiedashighrisk	Usethreshold/costanalysisandhumanreview.
Modelbecomesthefinallendingdecision	Needcreditofficer approval.
Variablesactasproxiesfor protectedcharacteristics	Excludehigh-riskfieldsandtestremainingvariablesforproxyeffects.
LendingClubdatadoesnotrepresentBKcustomers	Clearlystatethelimitationandrequirelocalvalidationbeforeuse.
Dataleakageproducesunrealisticresults	Useapplication-timevariablesandout-of-timetesting.
Privacyor inappropriatedatause	Applydataminimization,accesscontrolsandprivacyreview.
Regulatoryrequirementsaremissed	NeedCompliance,Legalandindependentmodelvalidation.
SHAPexplanationsaretreatedas causes	ClearlystatethatSHAPshowsmodelassociations,not causation.


6.	Quality,DocumentationandGovernanceStandards

Documentation and Quality: The project will keep a dataset datasheet, model card, architecture decision recordsandarecordshowinghowbusiness needs connect tothedata, models, testsanddashboards. Data-validation and unit tests will be used to check quality. Code changes will be reviewed through GitHub, and important experiments will use the same settings so that the results can be repeated(Gebru etal.,2021).

The report and project will follow professional reporting standards, including a proper citations and references. The project will also use the NIST AI Risk Management Framework as a general guide for managing AI risk (NIST, 2023).

VersioningStrategy:Codewillbemanagedusing Git andGitHub, withbranches andpull-request review. DVC will be used to track data versions, and model versions will be tracked using MLflow and semantic versioning. This will help with each result to be connected to the code, data and model version used.

Data Governance: Data access will follow the least privilege principle, which means people will only get access they need. Data will only be used for the mentioned credit risk project, and personal information will be kept to minimum and protected. Raw data will not be uploaded directly to GitHub. Data will be kept according to privacy rules, and for future BK data, Rwanda's personal data protection law and bank policy will be followed. The LendingClub results will always be kept separate from assumptions about BK customers. The solution will not be use in the real BK system until it has current BK data, local testing, fairness testing, regulatory review and approval.


