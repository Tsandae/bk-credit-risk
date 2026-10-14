# Ethical Risk Register

Risk  ----------------------------------------------------------------- Mitigation

Model performs differently across groups ---------------------------- Monitor group performance and fairness measures.


Good customers are incorrectly classified as high risk---------------Use threshold/cost analysis and human review.


Model becomes the final lending decision------------------------------Need credit officer approval.


Variables act as proxies for protected characteristics---------------Exclude high-risk fields and test remaining variables for proxy effects.


LendingClub data does not represent BK customers----------------------Clearly state the limitation and require local validation before use.


Data leakage produces unrealistic results---------------------------- Use application-time variables and out-of-time testing.


Privacy or inappropriate data use------------------------------------ Apply data minimization, access controls and privacy review. 


Regulatory requirements are missed------------------------------------- Need Compliance, Legal and independent model validation. 


SHAP explanations are treated as causes-------------------------------- Clearly state that SHAP shows model associations, not causation.
