import pandas as pd 
import numpy as np
import re
from typing import Tuple
from scipy import stats
from nameparser import HumanName
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

class SuspiciousLoanAnalyzer:
    def __init__(self, suspicious_file: str, full_data_file: str):
        self.suspicious_file = suspicious_file
        self.full_data_file = full_data_file
        self.sus_data = None
        self.full_data = None

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load both suspicious and full datasets with proper error handling."""
        print("Loading suspicious loans data...")
        try:
            sus = pd.read_csv(self.suspicious_file, low_memory=False)
            print("Loading full loan dataset...")
            cols = [
                "LoanNumber", "BorrowerName", "BorrowerCity", "BorrowerState",
                "OriginatingLender", "InitialApprovalAmount", "BusinessType",
                "Race", "Gender", "Ethnicity", "NAICSCode", "JobsReported"
            ]
            full = pd.read_csv(
                self.full_data_file,
                usecols=cols,
                dtype={"LoanNumber": str, "NAICSCode": str},
                low_memory=False
            )
            
            # Filter amount range
            full = full[
                (full["InitialApprovalAmount"] >= 5000) &
                (full["InitialApprovalAmount"] < 22000)
            ]
            
            # Store data for later use
            self.sus_data = sus.copy()
            self.full_data = full.copy()
            
            print(f"Loaded {len(sus):,} suspicious loans and {len(full):,} total loans in range.")
            return sus, full
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def safe_divide(self, a: float, b: float, default: float = 0.0) -> float:
        """Safely divide two numbers, handling division by zero."""
        try:
            if b == 0:
                return default
            result = a / b
            if np.isinf(result) or np.isnan(result):
                return default
            return result
        except Exception:
            return default

    def calculate_representation_ratio(
        self,
        suspicious_counts: pd.Series,
        full_counts: pd.Series,
        min_occurrences: int = 5,
    ) -> pd.DataFrame:
        """Calculate representation ratios with corrected handling of percentages."""
        try:
            # Ensure counts are positive
            suspicious_counts = suspicious_counts.clip(lower=0)
            full_counts = full_counts.clip(lower=0)
            
            # Calculate suspicious percentage using total suspicious loans
            sus_total = suspicious_counts.sum()
            sus_pct = suspicious_counts / sus_total if sus_total > 0 else 0
            
            # Calculate full percentage using total loans
            full_total = full_counts.sum()
            full_pct = full_counts / full_total if full_total > 0 else 0
            
            # Calculate ratios with safe division
            ratios = pd.Series(
                {idx: self.safe_divide(
                    sus_pct.get(idx, 0),
                    full_pct.get(idx, 0),
                    default=0.0
                ) for idx in set(suspicious_counts.index) | set(full_counts.index)}
            )
            
            df = pd.DataFrame({
                "Suspicious_Count": suspicious_counts,
                "Total_Count": full_counts,
                "Suspicious_Pct": sus_pct,
                "Overall_Pct": full_pct,
                "Representation_Ratio": ratios
            })
            
            # Filter and sort
            result = df[df["Suspicious_Count"] >= min_occurrences].sort_values(
                "Representation_Ratio", ascending=False
            )
            
            # Replace inf and nan with 0
            result = result.replace([np.inf, -np.inf, np.nan], 0)
            
            return result
            
        except Exception as e:
            print(f"Error calculating representation ratio: {str(e)}")
            return pd.DataFrame()

    def analyze_categorical_patterns(
            self,
            sus: pd.DataFrame,
            full: pd.DataFrame,
            column: str,
            title: str,
            min_occurrences: int = 5,
        ) -> None:
            """Analyze patterns in categorical variables with proper chi-squared testing."""
            try:
                print(f"\nAnalyzing {title}...")
                
                # Handle missing values
                sus[column] = sus[column].fillna('Unknown')
                full[column] = full[column].fillna('Unknown')
                
                # Calculate value counts
                s_counts = sus[column].value_counts()
                f_counts = full[column].value_counts()
                
                # Create contingency table
                categories = sorted(set(s_counts.index) | set(f_counts.index))
                cont_table = np.zeros((2, len(categories)))
                
                # Fill in suspicious counts
                for i, cat in enumerate(categories):
                    cont_table[0, i] = s_counts.get(cat, 0)  # Suspicious
                    cont_table[1, i] = f_counts.get(cat, 0) - s_counts.get(cat, 0)  # Non-suspicious
                
                # Ensure no negative values
                cont_table[cont_table < 0] = 0
                
                # Remove columns with all zeros
                non_zero_cols = cont_table.sum(axis=0) > 0
                cont_table = cont_table[:, non_zero_cols]
                categories = [cat for i, cat in enumerate(categories) if non_zero_cols[i]]
                
                # Calculate chi-square test if possible
                p_chi2 = None
                if cont_table.shape[1] >= 2 and (cont_table > 0).all():
                    try:
                        _, p_chi2, _, _ = stats.chi2_contingency(cont_table)
                    except Exception as e:
                        print(f"Chi-square test calculation failed: {str(e)}")
                
                # Calculate representation ratios
                analysis = self.calculate_representation_ratio(
                    s_counts, f_counts, min_occurrences
                )
                
                # Print results
                print("=" * 80)
                print(f"{title} Analysis")
                print(f"Chi-square test p-value: {p_chi2 if p_chi2 is not None else 'N/A'}")
                
                if not analysis.empty:
                    print("Top over-represented categories (top 100):")
                    for idx, row in analysis.head(100).iterrows():
                        if row["Representation_Ratio"] > 0:
                            print(
                                f"{idx}: {row['Representation_Ratio']:.2f}x more common in suspicious loans "
                                f"({int(row['Suspicious_Count'])} occurrences, {row['Suspicious_Pct']:.3%} vs {row['Overall_Pct']:.3%})"
                            )
                else:
                    print("No categories met the minimum occurrence threshold")
                    
            except Exception as e:
                print(f"Error analyzing {title}: {str(e)}")
            
    def analyze_geographic_clusters(self, sus: pd.DataFrame, full: pd.DataFrame) -> None:
        """Analyze geographic clusters with proper error handling."""
        try:
            # Ensure consistent location formatting
            sus["Location"] = sus.apply(
                lambda x: f"{str(x['BorrowerCity']).strip()}, {str(x['BorrowerState']).strip()}",
                axis=1
            )
            full["Location"] = full.apply(
                lambda x: f"{str(x['BorrowerCity']).strip()}, {str(x['BorrowerState']).strip()}",
                axis=1
            )
            
            self.analyze_categorical_patterns(
                sus, full, "Location", "Geographic Location Patterns", 3
            )
            
        except Exception as e:
            print(f"Error in geographic clusters analysis: {str(e)}")

    def analyze_lender_patterns(self, sus: pd.DataFrame, full: pd.DataFrame) -> None:
        """Analyze lender patterns with proper error handling."""
        try:
            self.analyze_categorical_patterns(
                sus, full, "OriginatingLender", "Lender Patterns"
            )
        except Exception as e:
            print(f"Error in lender patterns analysis: {str(e)}")

    def analyze_business_patterns(self, sus: pd.DataFrame, full: pd.DataFrame) -> None:
        """Analyze business patterns with proper error handling."""
        try:
            self.analyze_categorical_patterns(
                sus, full, "BusinessType", "Business Type Patterns"
            )
            
            # Analyze NAICS sectors
            sus["NAICSSector"] = sus["NAICSCode"].astype(str).str[:2]
            full["NAICSSector"] = full["NAICSCode"].astype(str).str[:2]
            
            self.analyze_categorical_patterns(
                sus, full, "NAICSSector", "Industry Sector Patterns"
            )
        except Exception as e:
            print(f"Error in business patterns analysis: {str(e)}")

    def parse_name(self, name_str: str) -> pd.Series:
        """Parse a name string into first and last name components."""
        try:
            if not name_str or pd.isna(name_str):
                return pd.Series({"FirstName": "Unknown", "LastName": "Unknown"})
                
            # Clean business suffixes
            clean = re.sub(
                r"\b(LLC|INC|CORP|CORPORATION|LTD|LIMITED|CO|COMPANY)\b\.?",
                "",
                str(name_str),
                flags=re.IGNORECASE
            ).strip()
            
            # Parse name
            n = HumanName(clean)
            
            return pd.Series({
                "FirstName": n.first if n.first else "Unknown",
                "LastName": n.last if n.last else "Unknown"
            })
            
        except Exception as e:
            print(f"Error parsing name '{name_str}': {str(e)}")
            return pd.Series({"FirstName": "Unknown", "LastName": "Unknown"})

    def extract_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract first and last names from business names."""
        try:
            df = df.copy()
            name_parts = df["BorrowerName"].fillna("").astype(str).apply(self.parse_name)
            df["FirstName"] = name_parts["FirstName"]
            df["LastName"] = name_parts["LastName"]
            return df
        except Exception as e:
            print(f"Error extracting names: {str(e)}")
            df["FirstName"] = "Unknown"
            df["LastName"] = "Unknown"
            return df

    def analyze_name_patterns(self, sus: pd.DataFrame, full: pd.DataFrame) -> None:
        """Analyze name patterns with proper error handling."""
        try:
            sus = self.extract_names(sus.copy())
            full = self.extract_names(full.copy())
            
            self.analyze_categorical_patterns(
                sus, full, "FirstName", "First Name Patterns", 3
            )
            self.analyze_categorical_patterns(
                sus, full, "LastName", "Last Name Patterns", 3
            )
            
            # Name length analysis
            sus["NameLength"] = sus["BorrowerName"].astype(str).str.len()
            full["NameLength"] = full["BorrowerName"].astype(str).str.len()
            
            avg_sus = sus["NameLength"].mean()
            avg_full = full["NameLength"].mean()
            
            print("=" * 80)
            print("Name Length Analysis")
            print(f"Suspicious avg: {avg_sus:.1f} chars; Overall avg: {avg_full:.1f} chars")
            
            # Statistical tests
            name_length_tests = [
                ("t-test", stats.ttest_ind),
                ("KS test", stats.ks_2samp),
                ("Mann-Whitney U", lambda x, y: stats.mannwhitneyu(x, y, alternative="two-sided"))
            ]
            
            for test_name, test_func in name_length_tests:
                try:
                    _, p_val = test_func(
                        sus["NameLength"].dropna(),
                        full["NameLength"].dropna()
                    )
                    print(f"{test_name} p-value: {p_val}")
                except Exception as e:
                    print(f"{test_name} failed: {str(e)}")
            
            # Word count analysis
            sus["WordCount"] = sus["BorrowerName"].astype(str).apply(lambda x: len(x.split()))
            full["WordCount"] = full["BorrowerName"].astype(str).apply(lambda x: len(x.split()))
            
            avg_words_sus = sus["WordCount"].mean()
            avg_words_full = full["WordCount"].mean()
            
            print("\nWord Count Analysis")
            print(f"Suspicious avg: {avg_words_sus:.1f} words; Overall avg: {avg_words_full:.1f} words")
            
            for test_name, test_func in name_length_tests:
                try:
                    _, p_val = test_func(
                        sus["WordCount"].dropna(),
                        full["WordCount"].dropna()
                    )
                    print(f"{test_name} p-value: {p_val}")
                except Exception as e:
                    print(f"{test_name} failed: {str(e)}")
                    
        except Exception as e:
            print(f"Error in name patterns analysis: {str(e)}")

    def analyze_business_name_patterns(self, sus: pd.DataFrame, full: pd.DataFrame) -> None:
        """Analyze business name patterns with proper error handling."""
        patterns = {
            "Contains Numbers": r"\d+",
            "ALL CAPS": lambda x: str(x).isupper(),
            "Multiple Spaces": r"\s{2,}",
            "Special Characters": r"[^a-zA-Z0-9\s]",
            "Starts with The": r"^THE\s",
            "Ends with LLC": r"LLC$",
            "Contains DBA": r"\bDBA\b",
            "Contains Trading": r"\bTRADING\b",
            "Contains Consulting": r"\bCONSULTING\b",
            "Contains Services": r"\bSERVICES\b",
            "Contains Solutions": r"\bSOLUTIONS\b",
        }
        
        try:
            print("=" * 80)
            print("Business Name Pattern Analysis")
            
            for pname, pat in patterns.items():
                try:
                    if callable(pat):
                        sus_match = sus["BorrowerName"].astype(str).apply(pat).mean()
                        full_match = full["BorrowerName"].astype(str).apply(pat).mean()
                    else:
                        sus_match = (
                            sus["BorrowerName"]
                            .astype(str)
                            .str.contains(pat, case=False, regex=True, na=False)
                            .mean()
                        )
                        full_match = (
                            full["BorrowerName"]
                            .astype(str)
                            .str.contains(pat, case=False, regex=True, na=False)
                            .mean()
                        )
                    ratio = self.safe_divide(sus_match, full_match)
                    print(
f"{pname}: {sus_match:.3%} in suspicious vs {full_match:.3%} overall ({ratio:.2f}x)"
                    )
                except Exception as e:
                    print(f"Error analyzing pattern {pname}: {str(e)}")
                    
        except Exception as e:
            print(f"Error in business name patterns analysis: {str(e)}")

    def analyze_demographic_patterns(self, sus: pd.DataFrame, full: pd.DataFrame) -> None:
        """Analyze demographic patterns with proper error handling."""
        try:
            for col in ["Race", "Gender", "Ethnicity"]:
                # Handle missing values consistently
                sus[col] = sus[col].fillna("Unknown/NotStated")
                full[col] = full[col].fillna("Unknown/NotStated")
                
                self.analyze_categorical_patterns(
                    sus, full, col, f"{col} Distribution Analysis"
                )
                
        except Exception as e:
            print(f"Error in demographic patterns analysis: {str(e)}")

    def analyze_risk_flags(self, sus: pd.DataFrame) -> None:
        """Analyze risk flags with proper error handling."""
        if "RiskFlags" not in sus.columns:
            print("No RiskFlags column found in suspicious data.")
            return
            
        try:
            # Extract and count flags
            flag_series = (
                sus["RiskFlags"]
                .fillna("")
                .apply(lambda x: [f.strip() for f in str(x).split(";") if f.strip()])
            )
            
            # Count all flags
            all_flags = [f for sub in flag_series for f in sub]
            flag_counts = pd.Series(all_flags).value_counts()
            
            # Calculate percentages
            total_loans = len(sus)
            
            print("\nTop risk flags in suspicious loans:")
            for flag, count in flag_counts.head(100).items():
                percentage = (count / total_loans) * 100
                print(f"{flag}: {count:,} occurrences ({percentage:.1f}% of loans)")
                
        except Exception as e:
            print(f"Error analyzing risk flags: {str(e)}")

    def analyze_loan_amount_distribution(self, sus: pd.DataFrame, full: pd.DataFrame) -> None:
        """Analyze loan amount distribution with proper error handling."""
        try:
            # Ensure numeric values
            s_amt = pd.to_numeric(sus["InitialApprovalAmount"], errors="coerce")
            f_amt = pd.to_numeric(full["InitialApprovalAmount"], errors="coerce")
            
            # Calculate statistics
            stats_dict = {
                "Suspicious": {
                    "Mean": s_amt.mean(),
                    "Median": s_amt.median(),
                    "Std": s_amt.std(),
                    "Min": s_amt.min(),
                    "Max": s_amt.max(),
                    "IQR": s_amt.quantile(0.75) - s_amt.quantile(0.25),
                    "Skewness": s_amt.skew(),
                    "Kurtosis": s_amt.kurtosis()
                },
                "Overall": {
                    "Mean": f_amt.mean(),
                    "Median": f_amt.median(),
                    "Std": f_amt.std(),
                    "Min": f_amt.min(),
                    "Max": f_amt.max(),
                    "IQR": f_amt.quantile(0.75) - f_amt.quantile(0.25),
                    "Skewness": f_amt.skew(),
                    "Kurtosis": f_amt.kurtosis()
                }
            }
            
            print("\nLoan Amount Distribution")
            for group, stats in stats_dict.items():
                print(f"\n{group}:")
                for stat_name, value in stats.items():
                    if stat_name in ["Skewness", "Kurtosis"]:
                        print(f"  {stat_name}: {value:.3f}")
                    else:
                        print(f"  {stat_name}: ${value:,.2f}")
            
            # Statistical tests
            tests = [
                ("T-test", lambda x, y: stats.ttest_ind(x.dropna(), y.dropna(), equal_var=False)),
                ("KS test", lambda x, y: stats.ks_2samp(x.dropna(), y.dropna())),
                ("Mann-Whitney U", lambda x, y: stats.mannwhitneyu(x.dropna(), y.dropna(), alternative="two-sided"))
            ]
            
            print("\nStatistical Tests:")
            for test_name, test_func in tests:
                try:
                    _, p_val = test_func(s_amt, f_amt)
                    print(f"{test_name} p-value: {p_val}")
                except Exception as e:
                    print(f"{test_name} failed: {str(e)}")
                    
        except Exception as e:
            print(f"Error analyzing loan amount distribution: {str(e)}")

    def analyze_jobs_reported_patterns(self, sus: pd.DataFrame, full: pd.DataFrame) -> None:
        """Analyze jobs reported patterns with proper error handling."""
        try:
            # Convert to numeric and handle missing values
            s_jobs = pd.to_numeric(sus["JobsReported"], errors="coerce").fillna(0)
            f_jobs = pd.to_numeric(full["JobsReported"], errors="coerce").fillna(0)
            
            # Calculate statistics
            stats_dict = {
                "Suspicious": {
                    "Mean": s_jobs.mean(),
                    "Median": s_jobs.median(),
                    "Std": s_jobs.std(),
                    "Min": s_jobs.min(),
                    "Max": s_jobs.max(),
                    "Zero jobs": (s_jobs == 0).mean() * 100,
                    "Single job": (s_jobs == 1).mean() * 100,
                    "Multiple jobs": (s_jobs > 1).mean() * 100
                },
                "Overall": {
                    "Mean": f_jobs.mean(),
                    "Median": f_jobs.median(),
                    "Std": f_jobs.std(),
                    "Min": f_jobs.min(),
                    "Max": f_jobs.max(),
                    "Zero jobs": (f_jobs == 0).mean() * 100,
                    "Single job": (f_jobs == 1).mean() * 100,
                    "Multiple jobs": (f_jobs > 1).mean() * 100
                }
            }
            
            print("\nJobs Reported Analysis")
            for group, stats in stats_dict.items():
                print(f"\n{group}:")
                for stat_name, value in stats.items():
                    if "jobs" in stat_name.lower():
                        print(f"  {stat_name}: {value:.1f}%")
                    else:
                        print(f"  {stat_name}: {value:.2f}")
            
            # Statistical tests
            tests = [
                ("T-test", lambda x, y: stats.ttest_ind(x.dropna(), y.dropna(), equal_var=False)),
                ("Mann-Whitney U", lambda x, y: stats.mannwhitneyu(x.dropna(), y.dropna(), alternative="two-sided")),
                ("KS test", lambda x, y: stats.ks_2samp(x.dropna(), y.dropna()))
            ]
            
            print("\nStatistical Tests:")
            for test_name, test_func in tests:
                try:
                    _, p_val = test_func(s_jobs, f_jobs)
                    print(f"{test_name} p-value: {p_val}")
                except Exception as e:
                    print(f"{test_name} failed: {str(e)}")
                    
        except Exception as e:
            print(f"Error analyzing jobs reported patterns: {str(e)}")

    def analyze_risk_score_distribution(self, sus: pd.DataFrame) -> None:
        """Analyze risk score distribution with proper error handling."""
        if "RiskScore" not in sus.columns:
            print("\nNo RiskScore column found in suspicious data.")
            return
            
        try:
            scores = pd.to_numeric(sus["RiskScore"], errors="coerce").dropna()
            
            # Calculate detailed statistics
            stats = {
                "Mean": scores.mean(),
                "Median": scores.median(),
                "Std": scores.std(),
                "Min": scores.min(),
                "Max": scores.max(),
                "25th percentile": scores.quantile(0.25),
                "75th percentile": scores.quantile(0.75),
                "Skewness": scores.skew(),
                "Kurtosis": scores.kurtosis()
            }
            
            print("\nRisk Score Distribution in Suspicious Loans")
            for stat_name, value in stats.items():
                print(f"{stat_name}: {value:.2f}")
            
            # Calculate risk level distributions
            bins = [0, 25, 50, 75, 100]
            labels = ["Low Risk", "Medium Risk", "High Risk", "Very High Risk"]
            risk_levels = pd.cut(scores, bins=bins, labels=labels)
            risk_counts = risk_levels.value_counts().sort_index()
            
            print("\nRisk Level Distribution:")
            total_loans = len(scores)
            for level, count in risk_counts.items():
                percentage = (count / total_loans) * 100
                print(f"{level}: {count:,} loans ({percentage:.1f}%)")
                
        except Exception as e:
            print(f"Error analyzing risk score distribution: {str(e)}")

    def analyze_flag_count_distribution(self, sus: pd.DataFrame) -> None:
        """Analyze flag count distribution with proper error handling."""
        if "RiskFlags" not in sus.columns:
            print("\nNo RiskFlags column found in suspicious data.")
            return
            
        try:
            # Calculate flag counts
            flag_counts = (
                sus["RiskFlags"]
                .fillna("")
                .apply(lambda x: len([f for f in str(x).split(";") if f.strip()]))
            )
            
            # Calculate statistics
            stats = {
                "Mean": flag_counts.mean(),
                "Median": flag_counts.median(),
                "Std": flag_counts.std(),
                "Min": flag_counts.min(),
                "Max": flag_counts.max(),
                "25th percentile": flag_counts.quantile(0.25),
                "75th percentile": flag_counts.quantile(0.75)
            }
            
            print("\nRisk Flags Count Distribution in Suspicious Loans")
            for stat_name, value in stats.items():
                print(f"{stat_name}: {value:.2f}")
            
            # Distribution of flag counts
            count_distribution = flag_counts.value_counts().sort_index()
            total_loans = len(flag_counts)
            
            print("\nFlag Count Distribution:")
            for count, frequency in count_distribution.items():
                percentage = (frequency / total_loans) * 100
                print(f"{count} flags: {frequency:,} loans ({percentage:.1f}%)")
            
            # One-sample t-test
            try:
                t_stat, p_val = stats.ttest_1samp(flag_counts, 0)
                print(f"\nOne-sample t-test p-value (against 0): {p_val}")
            except Exception as e:
                print(f"T-test failed: {str(e)}")
                
        except Exception as e:
            print(f"Error analyzing flag count distribution: {str(e)}")

    def analyze_correlations(self, full: pd.DataFrame) -> None:
        """Analyze correlations between numerical features with proper error handling."""
        try:
            print("\nCorrelation Analysis among Numerical Features")
            
            # Prepare features
            prepared_df = self.prepare_features(full.copy(), add_flags=True)
            
            # Select features for correlation analysis
            features = [
                "InitialApprovalAmount",
                "JobsReported", 
                "NameLength",
                "WordCount",
                "FlagCount"
            ]
            
            # Calculate correlations
            correlation_matrix = prepared_df[features].corr()
            
            # Print formatted correlation matrix
            pd.set_option('display.float_format', lambda x: '{:.6f}'.format(x))
            print(correlation_matrix.round(6))
            pd.reset_option('display.float_format')
            
        except Exception as e:
            print(f"Error analyzing correlations: {str(e)}")

    def prepare_features(self, df: pd.DataFrame, add_flags: bool = False) -> pd.DataFrame:
        """Prepare features for analysis with proper error handling."""
        try:
            df = df.copy()
            
            # Handle numerical features
            df['JobsReported'] = pd.to_numeric(df['JobsReported'], errors='coerce').fillna(0)
            df['InitialApprovalAmount'] = pd.to_numeric(df['InitialApprovalAmount'], errors='coerce').fillna(0)
            
            # Add name-based features
            df['NameLength'] = df['BorrowerName'].astype(str).str.len()
            df['WordCount'] = df['BorrowerName'].astype(str).apply(lambda x: len(x.split()))
            
            # Add flag count if requested
            if add_flags and 'RiskFlags' in df.columns:
                df['FlagCount'] = df['RiskFlags'].fillna('').apply(
                    lambda x: len([f for f in str(x).split(";") if f.strip()])
                )
            elif add_flags:
                df['FlagCount'] = 0
                
            return df
            
        except Exception as e:
            print(f"Error preparing features: {str(e)}")
            return df

    def analyze_multivariate(self, sus: pd.DataFrame, full: pd.DataFrame) -> None:
        """Perform multivariate analysis with proper class balancing."""
        try:
            print("\nMultivariate Analysis via Logistic Regression")
            
            # Prepare features
            features = [
                "InitialApprovalAmount", "JobsReported", "NameLength",
                "WordCount", "FlagCount"
            ]
            
            # Add target variable to full dataset
            full_prepared = self.prepare_features(full, add_flags=True)
            full_prepared["Flagged"] = full_prepared["LoanNumber"].isin(
                sus["LoanNumber"]
            ).astype(int)
            
            # Prepare X and y
            X = full_prepared[features].fillna(0)
            y = full_prepared["Flagged"]
            
            # Handle class imbalance through resampling
            if len(y.unique()) >= 2:
                df_majority = full_prepared[y == 0]
                df_minority = full_prepared[y == 1]
                
                # Upsample minority class
                df_minority_upsampled = resample(
                    df_minority,
                    replace=True,
                    n_samples=len(df_majority),
                    random_state=42
                )
                
                # Combine majority and upsampled minority
                df_balanced = pd.concat([df_majority, df_minority_upsampled])
                
                # Prepare balanced dataset
                X = df_balanced[features].fillna(0)
                y = df_balanced["Flagged"]
                
                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.3, random_state=42, stratify=y
                )
                
                # Train model
                model = LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42
                )
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Print results
                print("\nClassification Report:")
                print(classification_report(y_test, y_pred, digits=3))
                
                print("\nConfusion Matrix:")
                print(confusion_matrix(y_test, y_pred))
                
                # Print feature importance
                coef_dict = dict(zip(features, model.coef_[0]))
                print("\nLogistic Regression Coefficients:")
                for feat, coef in sorted(
                    coef_dict.items(), key=lambda x: abs(x[1]), reverse=True
                ):
                    print(f"{feat}: {coef:.4f}")
            else:
                print("Not enough classes for logistic regression, even after resampling.")
                
        except Exception as e:
            print(f"Error in multivariate analysis: {str(e)}")

    def run_analysis(self) -> None:
        """Run the complete analysis pipeline with proper error handling."""
        try:
            sus, full = self.load_data()
            print(
                f"\nStarting analysis of {len(sus):,} suspicious loans "
                f"out of {len(full):,} loans in the $5k-$22k range"
            )
            
            # Define analysis steps
            analysis_steps = [
                ("Geographic Clusters", self.analyze_geographic_clusters),
                ("Lender Patterns", self.analyze_lender_patterns),
                ("Business Patterns", self.analyze_business_patterns),
                ("Name Patterns", self.analyze_name_patterns),
                ("Business Name Patterns", self.analyze_business_name_patterns),
                ("Demographic Patterns", self.analyze_demographic_patterns),
                ("Risk Flags Analysis", lambda s, f: self.analyze_risk_flags(s)),
                ("Loan Amount Distribution", self.analyze_loan_amount_distribution),
                ("Jobs Reported Patterns", self.analyze_jobs_reported_patterns),
                ("Risk Score Distribution", lambda s, f: self.analyze_risk_score_distribution(s)),
                ("Risk Flags Count Distribution", lambda s, f: self.analyze_flag_count_distribution(s)),
                ("Correlations", lambda s, f: self.analyze_correlations(f)),
                ("Multivariate Analysis", self.analyze_multivariate),
            ]
            
            # Run each analysis step
            for title, func in analysis_steps:
                print(f"\nStarting {title} analysis...")
                try:
                    func(sus, full)
                except Exception as e:
                    print(f"Error in {title} analysis: {str(e)}")
            
            # Calculate final statistics
            flag_rate = (len(sus) / len(full)) * 100
            print(
                f"\nOverall flagging rate: {flag_rate:.1f}% of loans in the "
                f"$5k-$22k range"
            )
            
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            raise

def main():
    """Main function to run the analysis."""
    analyzer = SuspiciousLoanAnalyzer(
        suspicious_file="suspicious_loans.csv",
        full_data_file="ppp-full.csv"
    )
    try:
        analyzer.run_analysis()
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()