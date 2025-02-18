# PPP Loan Fraud Detection System

## Overview

The PPP Loan Fraud Detection System is a comprehensive solution designed to analyze large-scale Paycheck Protection Program (PPP) loan data (8GB+ CSV files) to identify potentially fraudulent loans. Targeting loans between $5,000 and $21,000—where fraud has been observed at a higher frequency—the system processes data in manageable chunks and applies multiple, sophisticated fraud detection strategies. Each loan is assigned a weighted risk score based on a variety of indicators and patterns; loans exceeding a configurable risk threshold (default: 75) are flagged for further review.

## Installation and Usage

The instruction below have been tested on Ubuntu 23.04:

```bash
git clone https://github.com/Dicklesworthstone/ppp_loan_fraud_analysis
cd ppp_loan_fraud_analysis
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install wheel
python -m pip install --upgrade setuptools wheel
pip install -r requirements.txt
python simple_loan_fraud_score.py
python analyze_patterns_in_suspicious_loans.py
```

## Key Features

- **Efficient Large-Scale Processing:** Uses chunk-based processing and async operations to handle very large datasets while optimizing memory usage.
- **Multi-Faceted Fraud Detection:** Combines multiple strategies—including business name analysis, address validation, network mapping, loan amount analysis, demographics review, temporal patterns, and lender risk assessment.
- **Weighted Risk Scoring:** Implements a sophisticated scoring system that assigns and aggregates risk weights from diverse factors, with multipliers for specific combinations of risk flags.
- **Detailed Flagging and Reporting:** Generates comprehensive reports that include risk scores, specific flags, network connections, and geographic clusters to aid in further investigation.
- **False Positive Reduction:** Integrates validation checks, known-business verification, and contextual analysis to minimize false positives.
- **Secondary Pattern Analysis:** Offers an advanced statistical module to validate findings, discover deeper fraud patterns, and provide insights for ongoing system refinement.

## Core Fraud Detection Strategies

Fraud detection in PPP loans is a multifaceted challenge that requires a combination of data-driven analysis, historical insights, and a keen understanding of fraudsters’ tactics. Our system blends traditional heuristics with innovative pattern recognition to create a robust framework that not only detects fraud but also provides actionable intelligence. Below is a detailed look at our core strategies:

### 1. Business Name Analysis

Fraudsters often reveal subtle clues through the names they choose. Our subsystem digs deep into business names by:
- **Suspicious Keyword Detection:** Leveraging a meticulously curated dictionary—including luxury brands (e.g., "Gucci", "Rolex", "Bentley"), meme and internet references (e.g., "stonks", "diamond hands"), get-rich-quick phrases (e.g., "quick cash", "fast money"), and even PPP-specific terms—to flag potential red flags.
- **Pattern Recognition:** Detecting generic naming patterns (e.g., "consulting LLC", "holdings LLC"), overly simplistic or personal naming schemes, and unusual formatting that might be designed to obscure fraudulent intent.
- **Weighted Scoring:** Each red flag adds a calibrated risk weight based on historical fraud data, allowing the system to build a nuanced risk profile.

**Rationale:** Even minor anomalies in naming can unmask broader fraudulent schemes. By combining keyword detection with sophisticated pattern analysis, this approach transforms subtle hints into a clear signal of risk.

### 2. Address Validation and Analysis

A valid, verifiable address is the backbone of any legitimate business. Our analysis focuses on:
- **Residential vs. Commercial Detection:** Distinguishing residential addresses (e.g., those containing "apt", "unit", "suite") from commercial ones (e.g., "plaza", "building", "tower"). A residential address used for a business can be a significant red flag.
- **Pattern Analysis:** Spotting generic, fake, or PO box addresses that may be used to hide true business locations.
- **Cluster Analysis:** Identifying clusters where multiple businesses share a single address, which can indicate a coordinated attempt to manipulate the system.

**Rationale:** Fraudsters often use residential or non-standard addresses to mask their operations. This subsystem uses data patterns and historical trends to separate genuine business addresses from those meant to deceive.

### 3. Network Analysis

Fraud is rarely an isolated incident; it tends to occur in interconnected networks. Our system explores:
- **Business Networks:** Mapping relationships between businesses that share addresses or naming similarities to reveal potential fraud rings.
- **Lender Patterns:** Analyzing the timing and sequence of loan approvals from specific lenders to detect unusual approval patterns.
- **Geographic Clusters:** Tracking loan applications by ZIP code to expose localized clusters of suspicious activity.

**Rationale:** By linking spatial, temporal, and relational data, network analysis uncovers hidden connections that mimic real-world investigative techniques, revealing broader patterns of collusion.

### 4. Loan Amount Analysis

Fraudsters often manipulate loan amounts to maximize their gain. This subsystem scrutinizes numerical data through:
- **Per-Employee Analysis:** Calculating the loan amount per reported employee, with thresholds in place to flag unusually high or uniform amounts.
- **Maximum Amount Detection:** Identifying applications that request exact maximum amounts, a common tactic in fraudulent schemes.
- **Batch Analysis:** Looking for clusters of similar loan amounts and sequential patterns that hint at coordinated manipulation.

**Rationale:** Discrepancies in loan amounts can be as telling as any textual anomaly. By cross-referencing numeric patterns with historical fraud data, this strategy highlights irregularities that suggest systemic abuse.

### 5. Demographics and Documentation Analysis

The completeness and consistency of submitted information are key indicators of legitimacy:
- **Missing Information:** Systematically flagging applications where key demographic fields (Race, Gender, Ethnicity) or required documentation are absent.
- **Business Type Evaluation:** Weighing the declared business type against known risk profiles to gauge overall application integrity.
- **Documentation Quality:** Assessing the consistency and thoroughness of the submitted data to identify signs of deliberate omission.

**Rationale:** Incomplete or generic data is often a deliberate strategy to obscure fraudulent intent. By scrutinizing documentation quality alongside demographic consistency, this approach enhances our ability to differentiate between genuine and suspect applications.

### 6. Temporal Analysis

Timing can be a crucial clue in fraud detection. Our temporal analysis subsystem monitors:
- **Sequential Loan Detection:** Identifying rapid, sequential submissions that may indicate automated or orchestrated fraud.
- **Batch Timing Analysis:** Detecting clusters of loans approved within unusually short timeframes.
- **Temporal Correlation:** Cross-referencing submission times with other risk factors to contextualize anomalies.

**Rationale:** Coordinated fraud often unfolds in rapid bursts. By integrating timing data with other indicators, temporal analysis provides an essential dynamic layer to the overall risk assessment.

### 7. Lender Risk Analysis

The behavior of lenders is integral to the fraud ecosystem. This subsystem evaluates:
- **High-Risk Lender Identification:** Drawing on historical data to flag lenders with a proven track record of processing fraudulent loans.
- **Lender Pattern Analysis:** Monitoring approval patterns and loan sequences unique to each lender.
- **Reputation Assessment:** Incorporating each lender’s historical performance into the aggregated risk score.

**Rationale:** Not all lenders enforce the same level of scrutiny. By factoring in lender-specific risk, our system adds another dimension to the fraud detection model, ensuring that systemic vulnerabilities are not overlooked.

## Risk Scoring System

The system combines the scores from all fraud detection strategies into a cumulative risk score using a weighted approach:
- **Aggregation and Multipliers:** Each risk factor contributes a score that is summed, with multipliers applied for combinations of flags that strongly indicate fraud.
- **Risk Levels:** Final risk scores are categorized as follows:
  - **Very High Risk:** ≥75
  - **High Risk:** 50–74
  - **Medium Risk:** 25–49
  - **Low Risk:** <25

## False Positive Reduction

Reducing false positives is key to the system’s reliability:
- **Validation Checks:** Requires multiple independent risk factors before a loan is classified as high risk.
- **Negative Scoring:** Incorporates counter-indicators (such as known legitimate businesses, operating history, or positive repayment behavior) to lower risk scores where appropriate.
- **Known Business Verification:** Maintains an external list of validated businesses to help exclude legitimate loans from high-risk flagging.

## Output and Reporting

The system generates detailed output that aids in both immediate review and longer-term analysis:
- **Detailed Reports:** Output includes comprehensive risk scores, specific risk flags, network relationships, and geographic clustering information.
- **Data Exports:** Produces CSV files of suspicious loans along with detailed log files and statistics for monitoring progress and performance.
- **Visualization Aids:** Progress bars, logs, and optional graphical outputs help operators understand the decision-making process and refine detection strategies.

## Technical Implementation

Key technical features ensure the system is robust, efficient, and scalable:
- **Chunk Processing:** Splits large datasets into manageable segments to optimize memory usage and processing speed.
- **Async Operations:** Utilizes asynchronous file operations to further boost efficiency when handling large data volumes.
- **Progress Tracking and Logging:** Implements comprehensive logging and real-time progress tracking to enable transparent operation and troubleshooting.
- **Error Handling:** Robust error-handling mechanisms ensure that any issues in data processing are captured and reported.
- **Memory Management:** Designed to work with very large datasets without compromising performance.

## Usage

The system is configured to be both powerful and flexible:
- **Inputs:**
  - PPP loan dataset CSV file (auto-downloaded if not present)
  - Optional configuration parameters (risk threshold, chunk size)
  - Optional known business list for validation purposes
- **Outputs:**
  - CSV file of flagged suspicious loans
  - Detailed log files and progress reports
  - Comprehensive output reports with risk scores and detailed fraud indicators

## Secondary Pattern Analysis System

Beyond the primary fraud detection engine, a secondary analysis module (`analyze_patterns_in_suspicious_loans.py`) is provided for deeper statistical insight and validation:

### Purpose and Capabilities

- **Validation:** Confirms the primary system’s findings using statistical tests and multivariate analysis.
- **New Pattern Discovery:** Identifies previously unnoticed fraud indicators and aggregates subtle patterns that might not be obvious in individual loan analysis.
- **Systemic Issue Identification:** Uncovers patterns that suggest organized fraud rings or compromised lending processes.
- **Risk Refinement:** Generates insights that inform adjustments to risk weights and detection strategies.

**Rationale:** The secondary analysis module acts as an essential backstop to the primary detection engine. By applying rigorous statistical methods, it not only validates initial findings but also uncovers hidden correlations and emergent fraud patterns. This extra layer of scrutiny is vital for refining risk models, ensuring that the system evolves in step with emerging fraud tactics.

### Statistical Analysis Components

- **Geographic Pattern Analysis:** Uses chi-square and Fisher’s exact tests to identify clusters and over-representation of suspicious loans by region.
- **Lender Pattern Analysis:** Compares lender distributions in suspicious versus non-suspicious loans to pinpoint problematic lenders.
- **Business Pattern Analysis:** Analyzes NAICS codes and business types, applying statistical tests (t-tests, Mann-Whitney U tests) to evaluate significance.
- **Name Pattern Analysis:** Deconstructs business names using word count, pattern matching, and tests (Kolmogorov-Smirnov tests) to detect abnormal naming conventions.
- **Demographic Analysis:** Evaluates missing information and demographic distributions, ensuring that fraud detection does not inadvertently introduce bias.
- **Risk Flag Meta-Analysis:** Reviews the co-occurrence and distribution of individual risk flags to refine the overall scoring strategy.

**Rationale:** Each statistical component is designed to explore a distinct dimension of the fraud landscape. Geographic analysis highlights regional anomalies, while lender and business analyses expose operational irregularities. Name and demographic analyses further ensure that subtle inconsistencies are captured. Together, these components provide a multifaceted view of the data, ensuring that individual risk signals are contextualized within the broader fraud ecosystem.

### Advanced Statistical Techniques

- **Multivariate Analysis:** Uses logistic regression and correlation analysis to understand the interplay between different risk factors.
- **Feature Importance:** Assesses which features most strongly predict fraud, enabling continuous refinement of the detection system.
- **Cross-Validation:** Employs rigorous cross-validated model evaluations to ensure statistical robustness.

**Rationale:** Advanced techniques enable the system to capture complex interdependencies and avoid oversimplification. Multivariate analysis reveals how multiple factors interact to signal fraud, while feature importance ranking identifies the most critical predictors. Cross-validation safeguards against overfitting, ensuring that the model maintains its predictive power across diverse datasets. Collectively, these techniques ensure that the secondary analysis remains both robust and adaptive to new fraud patterns.

## Final Remarks

The PPP Loan Fraud Detection System is designed to provide a robust, multi-layered defense against fraudulent activity in PPP loan datasets. By combining detailed rule-based checks with advanced statistical and network analyses, the system not only flags suspicious loans in real time but also continually refines its detection capabilities based on emerging fraud patterns. Its comprehensive reporting and validation tools ensure that operators have the insights necessary to both investigate flagged loans and improve the system over time.
