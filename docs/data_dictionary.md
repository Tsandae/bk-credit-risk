# Preliminary data dictionary (LendingClub loan.csv)

Generated from loan.csv and LCDataDictionary.xlsx. Roles follow the Module 1 Vision Document (FO4, Charter clauses 4–7).

| Field | Planned role | Missing share | Observed type | Rationale |
|---|---|---|---|---|
| id | Exclude: identifier / free text | 0.0% | int64 | Identifier or free text with re-identification or proxy risk (Vision Document FO4) |
| member_id | Exclude: identifier / free text | 0.0% | int64 | Identifier or free text with re-identification or proxy risk (Vision Document FO4) |
| loan_amnt | Feature | 0.0% | float64 | Application-time credit or loan attribute |
| funded_amnt | Exclude: post-decision | 0.0% | float64 | Known only after approval and funding; near-duplicate of loan_amnt |
| funded_amnt_inv | Exclude: post-decision | 0.0% | float64 | Known only after approval and funding; near-duplicate of loan_amnt |
| term | Feature | 0.0% | str | Application-time credit or loan attribute |
| int_rate | Benchmark only | 0.0% | float64 | LendingClub’s own risk assessment; excluded from the main model, used to benchmark it |
| installment | Exclude: derived | 0.0% | float64 | Derived from loan amount, rate and term, so it encodes int_rate |
| grade | Benchmark only | 0.0% | str | LendingClub’s own risk assessment; excluded from the main model, used to benchmark it |
| sub_grade | Benchmark only | 0.0% | str | LendingClub’s own risk assessment; excluded from the main model, used to benchmark it |
| emp_title | Exclude: identifier / free text | 5.8% | str | Identifier or free text with re-identification or proxy risk (Vision Document FO4) |
| emp_length | Feature + fairness group | 5.1% | str | Application-time predictor; also monitored as income band, ownership status and employment-length band (possible age proxy) |
| home_ownership | Feature + fairness group | 0.0% | str | Application-time predictor; also monitored as income band, ownership status and employment-length band (possible age proxy) |
| annual_inc | Feature + fairness group | 0.0% | float64 | Application-time predictor; also monitored as income band, ownership status and employment-length band (possible age proxy) |
| verification_status | Feature | 0.0% | str | Application-time credit or loan attribute |
| issue_d | Split key | 0.0% | str | Used for out-of-time train/validation/test split; not a model input |
| loan_status | Target | 0.0% | str | Label: Fully Paid = good (0); Charged Off / Default = bad (1); other statuses unresolved and excluded |
| pymnt_plan | Exclude: post-origination | 0.0% | str | Payment plan is set after origination (leakage) |
| url | Exclude: identifier / free text | 0.0% | str | Identifier or free text with re-identification or proxy risk (Vision Document FO4) |
| desc | Exclude: identifier / free text | 85.8% | str | Identifier or free text with re-identification or proxy risk (Vision Document FO4) |
| purpose | Feature | 0.0% | str | Application-time credit or loan attribute |
| title | Exclude: identifier / free text | 0.0% | str | Identifier or free text with re-identification or proxy risk (Vision Document FO4) |
| zip_code | Exclude: location proxy | 0.0% | str | 3-digit ZIP can proxy for race/income; excluded per FO4 |
| addr_state | Monitor only (test before use) | 0.0% | str | Fairness monitoring group (region); excluded as input unless proxy tests pass |
| dti | Feature | 0.0% | float64 | Application-time credit or loan attribute |
| delinq_2yrs | Feature | 0.0% | float64 | Application-time credit or loan attribute |
| earliest_cr_line | Feature | 0.0% | str | Application-time credit or loan attribute |
| inq_last_6mths | Feature | 0.0% | float64 | Application-time credit or loan attribute |
| mths_since_last_delinq | Feature | 51.2% | float64 | Application-time credit or loan attribute |
| mths_since_last_record | Feature | 84.6% | float64 | Application-time credit or loan attribute |
| open_acc | Feature | 0.0% | float64 | Application-time credit or loan attribute |
| pub_rec | Feature | 0.0% | float64 | Application-time credit or loan attribute |
| revol_bal | Feature | 0.0% | float64 | Application-time credit or loan attribute |
| revol_util | Feature | 0.1% | float64 | Application-time credit or loan attribute |
| total_acc | Feature | 0.0% | float64 | Application-time credit or loan attribute |
| initial_list_status | Feature | 0.0% | str | Application-time credit or loan attribute |
| out_prncp | Exclude: leakage | 0.0% | float64 | Recorded after origination; reveals the repayment outcome |
| out_prncp_inv | Exclude: leakage | 0.0% | float64 | Recorded after origination; reveals the repayment outcome |
| total_pymnt | Exclude: leakage | 0.0% | float64 | Recorded after origination; reveals the repayment outcome |
| total_pymnt_inv | Exclude: leakage | 0.0% | float64 | Recorded after origination; reveals the repayment outcome |
| total_rec_prncp | Exclude: leakage | 0.0% | float64 | Recorded after origination; reveals the repayment outcome |
| total_rec_int | Exclude: leakage | 0.0% | float64 | Recorded after origination; reveals the repayment outcome |
| total_rec_late_fee | Exclude: leakage | 0.0% | float64 | Recorded after origination; reveals the repayment outcome |
| recoveries | Exclude: leakage | 0.0% | float64 | Recorded after origination; reveals the repayment outcome |
| collection_recovery_fee | Exclude: leakage | 0.0% | float64 | Recorded after origination; reveals the repayment outcome |
| last_pymnt_d | Exclude: leakage | 2.0% | str | Recorded after origination; reveals the repayment outcome |
| last_pymnt_amnt | Exclude: leakage | 0.0% | float64 | Recorded after origination; reveals the repayment outcome |
| next_pymnt_d | Exclude: leakage | 28.5% | str | Recorded after origination; reveals the repayment outcome |
| last_credit_pull_d | Exclude: leakage | 0.0% | str | Recorded after origination; reveals the repayment outcome |
| collections_12_mths_ex_med | Feature | 0.0% | float64 | Application-time credit or loan attribute |
| mths_since_last_major_derog | Feature | 75.0% | float64 | Application-time credit or loan attribute |
| policy_code | Exclude: identifier / free text | 0.0% | float64 | Identifier or free text with re-identification or proxy risk (Vision Document FO4) |
| application_type | Feature | 0.0% | str | Application-time credit or loan attribute |
| annual_inc_joint | Exclude: >99% missing | 99.9% | float64 | Only for joint applications (one resolved loan) |
| dti_joint | Exclude: >99% missing | 99.9% | float64 | Only for joint applications (one resolved loan) |
| verification_status_joint | Exclude: >99% missing | 99.9% | str | Only for joint applications (one resolved loan) |
| acc_now_delinq | Feature | 0.0% | float64 | Application-time credit or loan attribute |
| tot_coll_amt | Feature | 7.9% | float64 | Application-time credit or loan attribute |
| tot_cur_bal | Feature | 7.9% | float64 | Application-time credit or loan attribute |
| open_acc_6m | Optional: very high missingness | 97.6% | float64 | Added by LendingClub in 2015; almost entirely missing in resolved loans |
| open_il_6m | Optional: very high missingness | 97.6% | float64 | Added by LendingClub in 2015; almost entirely missing in resolved loans |
| open_il_12m | Optional: very high missingness | 97.6% | float64 | Added by LendingClub in 2015; almost entirely missing in resolved loans |
| open_il_24m | Optional: very high missingness | 97.6% | float64 | Added by LendingClub in 2015; almost entirely missing in resolved loans |
| mths_since_rcnt_il | Optional: very high missingness | 97.7% | float64 | Added by LendingClub in 2015; almost entirely missing in resolved loans |
| total_bal_il | Optional: very high missingness | 97.6% | float64 | Added by LendingClub in 2015; almost entirely missing in resolved loans |
| il_util | Optional: very high missingness | 97.9% | float64 | Added by LendingClub in 2015; almost entirely missing in resolved loans |
| open_rv_12m | Optional: very high missingness | 97.6% | float64 | Added by LendingClub in 2015; almost entirely missing in resolved loans |
| open_rv_24m | Optional: very high missingness | 97.6% | float64 | Added by LendingClub in 2015; almost entirely missing in resolved loans |
| max_bal_bc | Optional: very high missingness | 97.6% | float64 | Added by LendingClub in 2015; almost entirely missing in resolved loans |
| all_util | Optional: very high missingness | 97.6% | float64 | Added by LendingClub in 2015; almost entirely missing in resolved loans |
| total_rev_hi_lim | Feature | 7.9% | float64 | Application-time credit or loan attribute |
| inq_fi | Optional: very high missingness | 97.6% | float64 | Added by LendingClub in 2015; almost entirely missing in resolved loans |
| total_cu_tl | Optional: very high missingness | 97.6% | float64 | Added by LendingClub in 2015; almost entirely missing in resolved loans |
| inq_last_12m | Optional: very high missingness | 97.6% | float64 | Added by LendingClub in 2015; almost entirely missing in resolved loans |
