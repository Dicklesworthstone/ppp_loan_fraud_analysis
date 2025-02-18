import os
import psutil
import re
from typing import Tuple
import pandas as pd 
import numpy as np
from scipy import stats
from nameparser import HumanName
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.utils import resample
import xgboost as xgb
import shap
from joblib import Parallel, delayed
from functools import lru_cache
from multiprocessing import Pool
import statsmodels.api as sm

@lru_cache(maxsize=1000000)
def parse_name_cached(name_str: str):
    import re
    from nameparser import HumanName
    if not name_str or pd.isna(name_str):
        return {"FirstName": "Unknown", "LastName": "Unknown"}
    clean = re.sub(r"\b(LLC|INC|CORP|CORPORATION|LTD|LIMITED|CO|COMPANY)\b\.?", "", name_str, flags=re.IGNORECASE).strip()
    n = HumanName(clean)
    return {"FirstName": n.first if n.first else "Unknown", "LastName": n.last if n.last else "Unknown"}

def run_stat_test(test_name, test_func, x, y):
    try:
        _, p_val = test_func(x, y)
        return f"{test_name} p-value: {p_val}"
    except Exception as e:
        return f"{test_name} failed: {str(e)}"

def process_name_chunk(chunk):
    """Process a chunk of names using parse_name_cached."""
    return [parse_name_cached(name) for name in chunk]

class XGBoostAnalyzer:
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.categorical_features = None
        
    def prepare_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare enhanced features for XGBoost analysis."""
        try:
            df = df.copy()
            
            # Basic numerical features
            df['JobsReported'] = pd.to_numeric(df['JobsReported'], errors='coerce').fillna(0)
            df['InitialApprovalAmount'] = pd.to_numeric(df['InitialApprovalAmount'], errors='coerce').fillna(0)
            df['NameLength'] = df['BorrowerName'].astype(str).str.len()
            df['WordCount'] = df['BorrowerName'].astype(str).apply(lambda x: len(x.split()))
            
            # Advanced amount features
            df['AmountPerEmployee'] = df.apply(
                lambda x: x['InitialApprovalAmount'] / x['JobsReported']
                if x['JobsReported'] > 0 else x['InitialApprovalAmount'],
                axis=1
            )
            df['IsRoundAmount'] = df['InitialApprovalAmount'].apply(
                lambda x: x % 100 == 0
            ).astype(int)
            
            # Address-based features
            residential_indicators = {
                'apt', 'unit', 'suite', '#', 'po box', 'residence', 'residential',
                'apartment', 'room', 'floor'
            }
            commercial_indicators = {
                'plaza', 'building', 'tower', 'office', 'complex', 'center',
                'mall', 'commercial', 'industrial', 'park'
            }
            
            address_str = df['BorrowerAddress'].astype(str).str.lower()
            df['HasResidentialIndicator'] = address_str.apply(
                lambda x: any(ind in x for ind in residential_indicators)
            ).astype(int)
            df['HasCommercialIndicator'] = address_str.apply(
                lambda x: any(ind in x for ind in commercial_indicators)
            ).astype(int)
            
            # Business concentration features
            address_key = df.apply(
                lambda x: f"{x['BorrowerAddress']}_{x['BorrowerCity']}_{x['BorrowerState']}",
                axis=1
            )
            address_counts = address_key.value_counts()
            df['BusinessesAtAddress'] = address_key.map(address_counts)
            
            # Loan amount pattern features
            max_amounts = [20832, 20833, 20834]
            df['IsExactMaxAmount'] = df['InitialApprovalAmount'].apply(
                lambda x: int(x) in max_amounts
            ).astype(int)
            
            # Business name pattern features
            name_str = df['BorrowerName'].astype(str).str.lower()
            suspicious_keywords = {
                'consulting', 'holdings', 'enterprise', 'solutions', 'services',
                'investment', 'trading', 'group', 'international', 'global'
            }
            df['HasSuspiciousKeyword'] = name_str.apply(
                lambda x: any(kw in x for kw in suspicious_keywords)
            ).astype(int)
            
            # Demographic completeness features
            demographic_fields = ['Race', 'Gender', 'Ethnicity']
            df['MissingDemographics'] = df[demographic_fields].isna().sum(axis=1)
            
            # Additional categorical features
            df['Location'] = df.apply(
                lambda x: f"{str(x['BorrowerCity']).strip()}, {str(x['BorrowerState']).strip()}",
                axis=1
            )
            df['NAICSSector'] = df['NAICSCode'].astype(str).str[:2]
            df['OriginatingLender'] = df['OriginatingLender'].fillna('Unknown')
            
            # Handle categorical variables with encoding
            categorical_features = [
                'BusinessType', 'Race', 'Gender', 'Ethnicity', 'NAICSCode',
                'BorrowerState', 'OriginatingLender', 'Location', 'NAICSSector'
            ]
            
            # Store categorical features for later use
            self.categorical_features = categorical_features
            
            # Encode categorical variables and ensure they are numeric
            for feature in categorical_features:
                df[feature] = df[feature].fillna('Unknown')
                df[feature] = pd.Categorical(df[feature]).codes.astype(int)  # Explicitly convert to int
            
            # Store feature names
            self.feature_names = df.columns.tolist()
            
            return df
            
        except Exception as e:
            print(f"Error preparing features: {str(e)}")
            return df

    def analyze_with_xgboost(self, sus: pd.DataFrame, full: pd.DataFrame, n_iter: int = 10) -> None:
        try:
            print("\nEnhanced XGBoost Analysis")
            
            full_prepared = self.prepare_enhanced_features(full.copy())
            full_prepared["Flagged"] = full_prepared["LoanNumber"].isin(sus["LoanNumber"]).astype(int)
            
            # Exclude non-numeric columns explicitly
            exclude_cols = ['LoanNumber', 'BorrowerName', 'BorrowerAddress', 'BorrowerCity', 'Flagged']
            feature_cols = [col for col in full_prepared.columns if col not in exclude_cols]
            
            # Verify all columns are numeric
            X = full_prepared[feature_cols]
            non_numeric_cols = X.select_dtypes(exclude=['int', 'float', 'bool']).columns
            if len(non_numeric_cols) > 0:
                print(f"Warning: Dropping non-numeric columns: {list(non_numeric_cols)}")
                X = X.drop(columns=non_numeric_cols)
                feature_cols = [col for col in feature_cols if col not in non_numeric_cols]
            
            y = full_prepared["Flagged"]
            
            # Downsample if dataset is too large
            if len(X) > 50000:
                X, _, y, _ = train_test_split(X, y, train_size=50000, stratify=y, random_state=42)
                print("Reduced dataset to 50,000 samples for tuning to maintain responsiveness")
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            param_grid = {
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1],
                'n_estimators': [100, 200],
                'min_child_weight': [1, 3],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'scale_pos_weight': [sum(y == 0) / sum(y == 1) if sum(y == 1) > 0 else 1]
            }
            
            xgb_clf = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='auc',
                random_state=42,
                tree_method='hist',
                nthread=8,
                max_bin=256
            )
            
            random_search = RandomizedSearchCV(
                xgb_clf,
                param_distributions=param_grid,
                n_iter=n_iter,
                scoring='roc_auc',
                cv=2,
                random_state=42,
                n_jobs=4,
                verbose=2
            )
            
            print("Performing hyperparameter tuning... (Press Ctrl+C to interrupt if needed)")
            random_search.fit(X_train, y_train)  # No early stopping here
            
            self.model = random_search.best_estimator_
            
            # Configure early stopping for final fit with eval_set
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
                early_stopping_rounds=10,
                verbose=1
            )
            
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            avg_precision = average_precision_score(y_test, y_pred_proba)
            
            print("\nModel Performance:")
            print(f"ROC-AUC Score: {roc_auc:.3f}")
            print(f"Average Precision Score: {avg_precision:.3f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, digits=3))
            
            self._analyze_feature_importance(X, feature_cols)
            self._analyze_shap_values(X_test, feature_cols)
            
            full_prepared['PredictedProbability'] = self.model.predict_proba(full_prepared[feature_cols])[:, 1]
            self._analyze_high_probability_patterns(full_prepared)
            
        except KeyboardInterrupt:
            print("\nXGBoost analysis interrupted by user")
            return
        except Exception as e:
            print(f"Error in XGBoost analysis: {str(e)}")
            raise

    def _analyze_feature_importance(
        self,
        X: pd.DataFrame,
        feature_cols: list
    ) -> None:
        """Analyze and display feature importance."""
        try:
            importance_df = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': self.model.feature_importances_
            })
            importance_df = importance_df.sort_values(
                'Importance', ascending=False
            ).head(20)
            
            print("\nTop 20 Most Important Features:")
            for _, row in importance_df.iterrows():
                print(f"{row['Feature']}: {row['Importance']:.4f}")
                
        except Exception as e:
            print(f"Error analyzing feature importance: {str(e)}")

    def _analyze_shap_values(
        self,
        X_test: pd.DataFrame,
        feature_cols: list
    ) -> None:
        """Analyze SHAP values for feature interactions."""
        try:
            print("\nAnalyzing feature interactions with SHAP values...")
            
            # Get top feature interactions
            interaction_values = shap.TreeExplainer(
                self.model
            ).shap_interaction_values(X_test)
            
            np.fill_diagonal(interaction_values, 0)
            interaction_sum = np.sum(np.abs(interaction_values), axis=(0, 1))
            
            # Get top 10 interacting features
            top_interactions = pd.DataFrame({
                'Feature': feature_cols,
                'Interaction_Strength': interaction_sum
            }).sort_values('Interaction_Strength', ascending=False).head(10)
            
            print("\nTop 10 Feature Interactions:")
            for _, row in top_interactions.iterrows():
                print(f"{row['Feature']}: {row['Interaction_Strength']:.4f}")
                
        except Exception as e:
            print(f"Error analyzing SHAP values: {str(e)}")
                
    def _analyze_high_probability_patterns(self, df: pd.DataFrame) -> None:
        """Analyze patterns in high-probability suspicious loans."""
        try:
            # Define high probability threshold (e.g., top 1%)
            high_prob_threshold = df['PredictedProbability'].quantile(0.99)
            high_prob_loans = df[df['PredictedProbability'] >= high_prob_threshold]
            
            print("\nAnalysis of High-Probability Suspicious Loans:")
            print(f"Number of high-probability loans: {len(high_prob_loans)}")
            
            # Analyze patterns in numerical features
            numerical_features = ['InitialApprovalAmount', 'JobsReported', 
                                'AmountPerEmployee', 'BusinessesAtAddress']
            
            print("\nNumerical Feature Patterns:")
            for feature in numerical_features:
                mean_all = df[feature].mean()
                mean_high_prob = high_prob_loans[feature].mean()
                print(f"{feature}:")
                print(f"  All loans mean: {mean_all:.3f}")
                print(f"  High-prob loans mean: {mean_high_prob:.3f}")
                print(f"  Ratio: {mean_high_prob/mean_all:.2f}x")
            
            # Analyze categorical feature patterns
            if self.categorical_features:
                print("\nCategorical Feature Patterns in High-Probability Loans:")
                print("Note: Patterns reflect data distribution, not targeted focus on any group.")
                for feature in self.categorical_features:
                    print(f"\n{feature} Distribution:")
                    if feature == "Race":
                        print("  Note: Racial patterns reflect data distribution, not targeted focus.")
                    value_counts = high_prob_loans[feature].value_counts()
                    total = len(high_prob_loans)
                    for value, count in value_counts.items():  # Show all categories
                        percentage = (count / total) * 100
                        print(f"  {value}: {count:,} ({percentage:.3f}%)")
                        
        except Exception as e:
            print(f"Error analyzing high-probability patterns: {str(e)}")

class SuspiciousLoanAnalyzer:
    def __init__(self, suspicious_file: str, full_data_file: str):
        self.suspicious_file = suspicious_file
        self.full_data_file = full_data_file
        self.sus_data = None
        self.full_data = None
        self.naics_lookup = self._load_naics_codes()

    def _load_naics_codes(self) -> dict:
        """Define NAICS 2-digit codes inline."""
        # Inline NAICS 2022 2-digit codes with ranges expanded
        naics_dict = {
            "11": "Agriculture, Forestry, Fishing and Hunting",
            "21": "Mining, Quarrying, and Oil and Gas Extraction",
            "22": "Utilities",
            "23": "Construction",
            "31": "Manufacturing - Food, Textile, and Leather",
            "32": "Manufacturing - Wood, Paper, Chemical, and Plastics",
            "33": "Manufacturing - Metal, Machinery, Electronics, and Transportation Equipment",
            "42": "Wholesale Trade",
            "44": "Retail Trade - Motor Vehicle, Furniture, Electronics, Building Materials, Food, Health, and Gasoline",
            "45": "Retail Trade - Sporting Goods, Books, Music, General Merchandise, and Non-store",
            "48": "Transportation and Warehousing - Air, Rail, Water, Truck, Transit, Pipeline, and Scenic/Sightseeing",
            "49": "Transportation and Warehousing - Postal Service, Couriers, Messengers, and Storage",
            "51": "Information",
            "52": "Finance and Insurance",
            "53": "Real Estate and Rental and Leasing",
            "54": "Professional, Scientific, and Technical Services",
            "55": "Management of Companies and Enterprises",
            "56": "Administrative and Support and Waste Management and Remediation Services",
            "61": "Educational Services",
            "62": "Health Care and Social Assistance",
            "71": "Arts, Entertainment, and Recreation",
            "72": "Accommodation and Food Services",
            "81": "Other Services (except Public Administration)",
            "92": "Public Administration",
            "99": "Unclassified Establishments"
        }
        return naics_dict
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load both suspicious and full datasets with proper error handling."""
        print("Loading suspicious loans data...")
        try:
            sus = pd.read_csv(self.suspicious_file, low_memory=False)
            print("Loading full loan dataset...")
            cols = [
                "LoanNumber", "BorrowerName", "BorrowerAddress", "BorrowerCity", "BorrowerState",
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
            
            # Debugging checks
            print(f"Suspicious dataset shape: {sus.shape}")
            print(f"Full dataset shape before filtering: {pd.read_csv(self.full_data_file, nrows=0).shape}")
            print(f"Full dataset shape after filtering: {full.shape}")
            print(f"Suspicious LoanNumbers unique: {sus['LoanNumber'].nunique()}")
            print(f"Full LoanNumbers unique: {full['LoanNumber'].nunique()}")
            
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
        high_threshold_occurrences: int = 25,
    ) -> None:
        """Analyze patterns in categorical variables with proper chi-squared testing and dual thresholds."""
        try:
            print(f"\nAnalyzing {title}...")
            
            # Handle missing values consistently
            sus[column] = sus[column].fillna('Unknown')
            full[column] = full[column].fillna('Unknown')
            
            # Calculate value counts
            s_counts = sus[column].value_counts()
            f_counts = full[column].value_counts()
            
            # Create contingency table correctly for chi-square test
            all_categories = sorted(set(s_counts.index) | set(f_counts.index))
            
            contingency = []
            for category in all_categories:
                suspicious = s_counts.get(category, 0)
                total = f_counts.get(category, 0)
                non_suspicious = max(0, total - suspicious)
                if suspicious > 0 or non_suspicious > 0:
                    contingency.append([suspicious, non_suspicious])
            
            cont_table = np.array(contingency)
            
            p_chi2 = None
            if cont_table.size > 0 and cont_table.shape[0] > 1:
                if np.all(cont_table.sum(axis=1) > 0) and np.all(cont_table.sum(axis=0) > 0):
                    try:
                        _, p_chi2, _, _ = stats.chi2_contingency(cont_table)
                    except Exception as e:
                        print(f"Chi-square test calculation failed: {str(e)}")
            
            # Calculate representation ratios with original threshold (min 5)
            analysis_original = self.calculate_representation_ratio(
                s_counts, f_counts, min_occurrences
            )
            
            # Calculate representation ratios with higher threshold (min 25)
            analysis_high_threshold = self.calculate_representation_ratio(
                s_counts, f_counts, high_threshold_occurrences
            )
            
            # Print results
            print("=" * 80)
            print(f"{title} Analysis")
            print(f"Chi-square test p-value: {p_chi2 if p_chi2 is not None else 'N/A'}")
            
            # Original threshold results (min 5)
            if not analysis_original.empty:
                print(f"\nTop over-represented categories (min {min_occurrences} occurrences, top 100):")
                for idx, row in analysis_original.head(100).iterrows():
                    if row["Representation_Ratio"] > 0:
                        # Special handling for NAICSSector to include description
                        if column == "NAICSSector" and self.naics_lookup:
                            idx_str = str(idx).zfill(2)
                            desc = self.naics_lookup.get(idx_str, "Unknown Sector")
                            display_name = f"{idx_str} - {desc}"
                        else:
                            display_name = str(idx)
                        
                        print(
                            f"{display_name}: {row['Representation_Ratio']:.2f}x more common in suspicious loans "
                            f"({int(row['Suspicious_Count']):,} occurrences, {row['Suspicious_Pct']:.3%} vs {row['Overall_Pct']:.3%})"
                        )
            else:
                print(f"No categories met the minimum occurrence threshold of {min_occurrences}")
            
            # Check if high threshold analysis differs from original
            if not analysis_high_threshold.empty:
                # Extract top 100 from both analyses for comparison
                orig_top = analysis_original.head(100)
                high_top = analysis_high_threshold.head(100)
                
                # Compare indices (categories) and check counts
                orig_indices = set(orig_top.index)
                high_indices = set(high_top.index)
                all_above_25 = all(s_counts.get(idx, 0) >= high_threshold_occurrences for idx in orig_indices)
                
                # Only print high threshold if it differs or not all min-5 categories exceed 25
                if orig_indices != high_indices or not all_above_25:
                    print(f"\nTop over-represented categories (min {high_threshold_occurrences} occurrences, top 100):")
                    for idx, row in high_top.iterrows():
                        if row["Representation_Ratio"] > 0:
                            if column == "NAICSSector" and self.naics_lookup:
                                idx_str = str(idx).zfill(2)
                                desc = self.naics_lookup.get(idx_str, "Unknown Sector")
                                display_name = f"{idx_str} - {desc}"
                            else:
                                display_name = str(idx)
                            
                            print(
                                f"{display_name}: {row['Representation_Ratio']:.2f}x more common in suspicious loans "
                                f"({int(row['Suspicious_Count']):,} occurrences, {row['Suspicious_Pct']:.3%} vs {row['Overall_Pct']:.3%})"
                            )
                # If identical, skip printing to avoid redundancy
            else:
                print(f"No categories met the minimum occurrence threshold of {high_threshold_occurrences}")
                        
        except Exception as e:
            print(f"Error analyzing {title}: {str(e)}")

    def analyze_feature_discrimination(self, sus: pd.DataFrame, full: pd.DataFrame) -> None:
        """Analyze feature discrimination using AUPRC with detailed category breakdown and significance."""
        try:
            print("\nFeature Discrimination Analysis using AUPRC")
            
            # Prepare enhanced features with all available flags
            full_prepared = self.prepare_enhanced_features(full.copy())
            full_prepared["Flagged"] = full_prepared["LoanNumber"].isin(sus["LoanNumber"]).astype(int)
            
            # Include all features and flags from prepare_enhanced_features
            features_to_analyze = [
                "InitialApprovalAmount", "JobsReported", "NameLength", "WordCount",
                "AmountPerEmployee", "HasResidentialAddress", "HasMultipleBusinesses",
                "IsExactMaxAmount", "IsRoundAmount", "HasCommercialIndicator",
                "HasSuspiciousKeyword", "MissingDemographics", "BusinessesAtAddress",
                "BusinessType", "Race", "Gender", "Ethnicity"
            ]
            
            y_true = full_prepared["Flagged"]
            feature_auprc = {}
            category_auprc_dict = {}
            
            # Calculate AUPRC and significance for each feature
            for feature in features_to_analyze:
                if feature not in full_prepared.columns:
                    continue
                    
                if full_prepared[feature].dtype in ['object', 'category']:
                    # One-hot encode categorical features
                    X = pd.get_dummies(full_prepared[feature], prefix=feature)
                    if X.shape[1] > 1:
                        model = LogisticRegression(max_iter=1000, random_state=42)
                        model.fit(X, y_true)
                        y_scores = model.predict_proba(X)[:, 1]
                        auprc = average_precision_score(y_true, y_scores)
                        feature_auprc[feature] = auprc
                        
                        # Chi-square test for significance
                        contingency = pd.crosstab(full_prepared[feature], y_true)
                        _, p_val, _, _ = stats.chi2_contingency(contingency)
                        
                        # Category-level AUPRC
                        category_auprc = {}
                        for col in X.columns:
                            single_feature = X[[col]]
                            model.fit(single_feature, y_true)
                            y_scores = model.predict_proba(single_feature)[:, 1]
                            category_auprc[col] = average_precision_score(y_true, y_scores)
                        category_auprc_dict[feature] = category_auprc
                    else:
                        y_scores = X.iloc[:, 0]
                        feature_auprc[feature] = average_precision_score(y_true, y_scores)
                        p_val = stats.ttest_ind(X.iloc[:, 0][y_true == 1], 
                                            X.iloc[:, 0][y_true == 0], 
                                            equal_var=False).pvalue
                else:
                    # Numerical features
                    X = full_prepared[[feature]].fillna(0)
                    model = LogisticRegression(max_iter=1000, random_state=42)
                    model.fit(X, y_true)
                    y_scores = model.predict_proba(X)[:, 1]
                    feature_auprc[feature] = average_precision_score(y_true, y_scores)
                    # T-test for significance
                    p_val = stats.ttest_ind(X[feature][y_true == 1], 
                                        X[feature][y_true == 0], 
                                        equal_var=False).pvalue
                
                # Store p-value for reporting
                feature_auprc[feature] = (feature_auprc[feature], p_val)
            
            # Display results
            print("\nDetailed Feature Analysis:")
            print("Note: Patterns reflect data distribution and statistical significance, not targeted focus on any group.")
            sorted_features = sorted(feature_auprc.items(), key=lambda x: x[1][0], reverse=True)
            for feature, (auprc, p_val) in sorted_features:
                sig_note = " (p < 0.05, significant)" if p_val < 0.05 else " (p >= 0.05, not significant)"
                print(f"{feature}: AUPRC = {auprc:.4f}{sig_note}")
                
                # Category breakdown for all categorical features
                if feature in category_auprc_dict:
                    print(f"  All categories for {feature}:")
                    if feature == "Race":
                        print("  Note: Racial patterns reflect data distribution, not targeted focus.")
                    sorted_categories = sorted(category_auprc_dict[feature].items(), 
                                            key=lambda x: x[1], reverse=True)
                    for i, (cat, cat_auprc) in enumerate(sorted_categories):  # Show all categories
                        prefix = "**" if i == 0 else "  "  # Bold top category
                        suffix = "**" if i == 0 else ""
                        print(f"    {prefix}{cat}: AUPRC = {cat_auprc:.4f}{suffix}")
            
            # Overall feature ranking
            print("\nFeatures ranked by AUPRC (discriminative power):")
            for feature, (auprc, _) in sorted_features:
                print(f"{feature}: AUPRC = {auprc:.4f}")
            
            # Baseline
            baseline_auprc = y_true.mean()
            print(f"\nBaseline AUPRC (random guessing): {baseline_auprc:.4f}")
            
        except Exception as e:
            print(f"Error in feature discrimination analysis: {str(e)}")


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
            names = df["BorrowerName"].fillna("").astype(str).tolist()
            parsed = Parallel(n_jobs=32)(delayed(parse_name_cached)(name) for name in names)
            df["FirstName"] = [p["FirstName"] for p in parsed]
            df["LastName"] = [p["LastName"] for p in parsed]            
            return df
        except Exception as e:
            print(f"Error extracting names: {str(e)}")
            df["FirstName"] = "Unknown"
            df["LastName"] = "Unknown"
            return df
        
    def extract_names_optimized(self, df: pd.DataFrame, n_jobs: int = 32, chunk_size: int = 1000) -> pd.DataFrame:
        """Extract first and last names from business names using optimized parallel processing."""
        try:
            df = df.copy()
            names = df["BorrowerName"].fillna("").astype(str).values  # Use .values for faster access
            
            # Split into chunks for better memory and CPU utilization
            chunks = [names[i:i + chunk_size] for i in range(0, len(names), chunk_size)]
            
            # Use multiprocessing.Pool for parallel execution
            with Pool(processes=n_jobs) as pool:
                results = pool.map(process_name_chunk, chunks)
            
            # Flatten results
            parsed = [item for sublist in results for item in sublist]
            df["FirstName"] = [p["FirstName"] for p in parsed]
            df["LastName"] = [p["LastName"] for p in parsed]
            return df
        
        except Exception as e:
            print(f"Error extracting names: {str(e)}")
            df["FirstName"] = "Unknown"
            df["LastName"] = "Unknown"
            return df
    
    def analyze_name_patterns(self, sus: pd.DataFrame, full: pd.DataFrame) -> None:
        """Analyze name patterns with proper error handling."""
        try:
            sus = self.extract_names_optimized(sus.copy())
            full = self.extract_names_optimized(full.copy())
            
            self.analyze_categorical_patterns(
                sus, full, "FirstName", "First Name Patterns", 3
            )
            self.analyze_categorical_patterns(
                sus, full, "LastName", "Last Name Patterns", 3
            )
            
            # Name length analysis
            sus["NameLength"] = sus["BorrowerName"].astype(str).str.len().fillna(0).astype(int)
            full["NameLength"] = full["BorrowerName"].astype(str).str.len().fillna(0).astype(int)
            
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
            
            # For NameLength
            results = Parallel(n_jobs=3)(
                delayed(run_stat_test)(test_name, test_func, sus["NameLength"].dropna(), full["NameLength"].dropna())
                for test_name, test_func in name_length_tests
            )
            for result in results:
                print(result)
            
            # Word count analysis
            sus["WordCount"] = sus["BorrowerName"].astype(str).str.split().str.len().fillna(0).astype(int)
            full["WordCount"] = full["BorrowerName"].astype(str).str.split().str.len().fillna(0).astype(int)

            avg_words_sus = sus["WordCount"].mean()
            avg_words_full = full["WordCount"].mean()
            
            print("\nWord Count Analysis")
            print(f"Suspicious avg: {avg_words_sus:.1f} words; Overall avg: {avg_words_full:.1f} words")
            
            # For WordCount
            results = Parallel(n_jobs=3)(
                delayed(run_stat_test)(test_name, test_func, sus["WordCount"].dropna(), full["WordCount"].dropna())
                for test_name, test_func in name_length_tests
            )
            for result in results:
                print(result)
                                
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
                print(f"{flag}: {count:,} occurrences ({percentage:.3f}% of loans)")
                
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
            for group, current_stats in stats_dict.items():
                print(f"\n{group}:")
                for stat_name, value in current_stats.items():
                    if stat_name in ["Skewness", "Kurtosis"]:
                        print(f"  {stat_name}: {value:.3f}")
                    else:
                        print(f"  {stat_name}: ${value:,.3f}")
            
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
                    print(f"{test_name} p-value: {p_val:.6f}")
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
            for group, current_stats in stats_dict.items():
                print(f"\n{group}:")
                for stat_name, value in current_stats.items():
                    if "jobs" in stat_name.lower():
                        print(f"  {stat_name}: {value:.3f}%")
                    else:
                        print(f"  {stat_name}: {value:.3f}")
            
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
                    print(f"{test_name} p-value: {p_val:.6f}")
                except Exception as e:
                    print(f"{test_name} failed: {str(e)}")
                    
        except Exception as e:
            print(f"Error analyzing jobs reported patterns: {str(e)}")

    def analyze_risk_score_distribution(self, sus: pd.DataFrame) -> None:
        if "RiskScore" not in sus.columns:
            print("\nNo RiskScore column found in suspicious data.")
            return
        
        try:
            scores = pd.to_numeric(sus["RiskScore"], errors="coerce").dropna()
            
            # Use a different name for the stats dictionary
            risk_stats = {
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
            for stat_name, value in risk_stats.items():
                print(f"{stat_name}: {value:.3f}")
            
            bins = [0, 25, 50, 75, 100]
            labels = ["Low Risk", "Medium Risk", "High Risk", "Very High Risk"]
            risk_levels = pd.cut(scores, bins=bins, labels=labels)
            risk_counts = risk_levels.value_counts().sort_index()
            
            print("\nRisk Level Distribution:")
            total_loans = len(scores)
            for level, count in risk_counts.items():
                percentage = (count / total_loans) * 100
                print(f"{level}: {count:,} loans ({percentage:.3f}%)")
        
        except Exception as e:
            print(f"Error analyzing risk score distribution: {str(e)}")

    def analyze_flag_count_distribution(self, sus: pd.DataFrame) -> None:
        if "RiskFlags" not in sus.columns:
            print("\nNo RiskFlags column found in suspicious data.")
            return
        
        try:
            flag_counts = (
                sus["RiskFlags"]
                .fillna("")
                .apply(lambda x: len([f for f in str(x).split(";") if f.strip()]))
            )
            
            flag_stats = {
                "Mean": flag_counts.mean(),
                "Median": flag_counts.median(),
                "Std": flag_counts.std(),
                "Min": flag_counts.min(),
                "Max": flag_counts.max(),
                "25th percentile": flag_counts.quantile(0.25),
                "75th percentile": flag_counts.quantile(0.75)
            }
            
            print("\nRisk Flags Count Distribution in Suspicious Loans")
            for stat_name, value in flag_stats.items():
                print(f"{stat_name}: {value:.3f}")
            
            count_distribution = flag_counts.value_counts().sort_index()
            total_loans = len(flag_counts)
            
            print("\nFlag Count Distribution:")
            for count, frequency in count_distribution.items():
                percentage = (frequency / total_loans) * 100
                print(f"{count} flags: {frequency:,} loans ({percentage:.3f}%)")
            
            try:
                t_stat, p_val = stats.ttest_1samp(flag_counts, 0)
                print(f"\nOne-sample t-test p-value (against 0): {p_val:.6f}")
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
                "WordCount"
            ]
            if "FlagCount" in prepared_df.columns and prepared_df["FlagCount"].nunique() > 1:
                features.append("FlagCount")

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

    def prepare_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare enhanced features for XGBoost analysis."""
        try:
            df = df.copy()
            
            # Basic numerical features
            df['JobsReported'] = pd.to_numeric(df['JobsReported'], errors='coerce').fillna(0)
            df['InitialApprovalAmount'] = pd.to_numeric(df['InitialApprovalAmount'], errors='coerce').fillna(0)
            df['NameLength'] = df['BorrowerName'].astype(str).str.len()
            df['WordCount'] = df['BorrowerName'].astype(str).apply(lambda x: len(x.split()))
            
            # Advanced amount features
            df['AmountPerEmployee'] = df.apply(
                lambda x: x['InitialApprovalAmount'] / x['JobsReported']
                if x['JobsReported'] > 0 else x['InitialApprovalAmount'],
                axis=1
            )
            df['IsRoundAmount'] = df['InitialApprovalAmount'].apply(
                lambda x: x % 100 == 0
            ).astype(int)
            
            # Address-based features
            residential_indicators = {
                'apt', 'unit', 'suite', '#', 'po box', 'residence', 'residential',
                'apartment', 'room', 'floor'
            }
            commercial_indicators = {
                'plaza', 'building', 'tower', 'office', 'complex', 'center',
                'mall', 'commercial', 'industrial', 'park'
            }
            
            address_str = df['BorrowerAddress'].astype(str).str.lower()
            df['HasResidentialIndicator'] = address_str.apply(
                lambda x: any(ind in x for ind in residential_indicators)
            ).astype(int)
            df['HasCommercialIndicator'] = address_str.apply(
                lambda x: any(ind in x for ind in commercial_indicators)
            ).astype(int)
            
            # Business concentration features
            address_key = df.apply(
                lambda x: f"{x['BorrowerAddress']}_{x['BorrowerCity']}_{x['BorrowerState']}",
                axis=1
            )
            address_counts = address_key.value_counts()
            df['BusinessesAtAddress'] = address_key.map(address_counts)
            
            # Loan amount pattern features
            max_amounts = [20832, 20833, 20834]
            df['IsExactMaxAmount'] = df['InitialApprovalAmount'].apply(
                lambda x: int(x) in max_amounts
            ).astype(int)
            
            # Business name pattern features
            name_str = df['BorrowerName'].astype(str).str.lower()
            suspicious_keywords = {
                'consulting', 'holdings', 'enterprise', 'solutions', 'services',
                'investment', 'trading', 'group', 'international', 'global'
            }
            df['HasSuspiciousKeyword'] = name_str.apply(
                lambda x: any(kw in x for kw in suspicious_keywords)
            ).astype(int)
            
            # Demographic completeness features
            demographic_fields = ['Race', 'Gender', 'Ethnicity']
            df['MissingDemographics'] = df[demographic_fields].isna().sum(axis=1)
            
            # Add and encode additional categorical features from the dataset
            df['Location'] = df.apply(
                lambda x: f"{str(x['BorrowerCity']).strip()}, {str(x['BorrowerState']).strip()}",
                axis=1
            )
            df['NAICSSector'] = df['NAICSCode'].astype(str).str[:2]
            df['OriginatingLender'] = df['OriginatingLender'].fillna('Unknown')
            
            # Handle categorical variables with encoding
            categorical_features = [
                'BusinessType', 'Race', 'Gender', 'Ethnicity', 'NAICSCode',
                'BorrowerState', 'OriginatingLender', 'Location', 'NAICSSector'
            ]
            
            # Store categorical features for later use
            self.categorical_features = categorical_features
            
            # Encode categorical variables
            for feature in categorical_features:
                df[feature] = df[feature].fillna('Unknown')
                df[feature] = pd.Categorical(df[feature]).codes
            
            # Store feature names
            self.feature_names = df.columns.tolist()
            
            return df
            
        except Exception as e:
            print(f"Error preparing features: {str(e)}")
            return df

    def analyze_multivariate(self, sus: pd.DataFrame, full: pd.DataFrame) -> None:
        try:
            print("\nEnhanced Multivariate Analysis via Logistic Regression")
            
            # Prepare full dataset with all enhanced features
            full_prepared = self.prepare_enhanced_features(full.copy())
            full_prepared["LoanNumber"] = full_prepared["LoanNumber"].astype(str).str.strip()
            sus["LoanNumber"] = sus["LoanNumber"].astype(str).str.strip()
            
            # Debugging output
            print(f"Suspicious LoanNumbers unique count: {sus['LoanNumber'].nunique()}")
            print(f"Full dataset LoanNumbers unique count: {full_prepared['LoanNumber'].nunique()}")
            print(f"Suspicious LoanNumbers sample: {sus['LoanNumber'].head().tolist()}")
            print(f"Full dataset LoanNumbers sample: {full_prepared['LoanNumber'].head().tolist()}")
            matched_loans = set(sus["LoanNumber"]).intersection(set(full_prepared["LoanNumber"]))
            print(f"Number of matched loan numbers: {len(matched_loans)}")
            
            full_prepared["Flagged"] = full_prepared["LoanNumber"].isin(matched_loans).astype(int)
            print(f"Class distribution - Suspicious: {full_prepared['Flagged'].sum()}, "
                f"Non-suspicious: {len(full_prepared) - full_prepared['Flagged'].sum()}")
            
            if full_prepared["Flagged"].sum() == 0:
                print("Error: No suspicious loans found in the full dataset after flagging.")
                return

            # Select features (corrected to match prepare_enhanced_features output)
            numerical_features = [
                "InitialApprovalAmount", "JobsReported", "NameLength", "WordCount",
                "AmountPerEmployee", "HasResidentialIndicator", "BusinessesAtAddress",  # Corrected names
                "IsExactMaxAmount", "IsRoundAmount", "HasCommercialIndicator",
                "HasSuspiciousKeyword", "MissingDemographics"
            ]
            categorical_features = ["BusinessType", "Race", "Gender", "Ethnicity"]
            
            # Verify feature existence
            missing_features = [f for f in numerical_features if f not in full_prepared.columns]
            if missing_features:
                print(f"Warning: Features {missing_features} not found in prepared data. Skipping them.")
                numerical_features = [f for f in numerical_features if f in full_prepared.columns]
            
            X = full_prepared[numerical_features].copy()
            for feature in categorical_features:
                dummies = pd.get_dummies(full_prepared[feature], prefix=feature)
                X = pd.concat([X, dummies], axis=1)
            
            y = full_prepared["Flagged"]
            
            # Handle class imbalance
            if len(y.unique()) >= 2:
                df_majority = X[y == 0]
                df_minority = X[y == 1]
                if len(df_minority) > 0:
                    df_minority_upsampled = resample(
                        df_minority, replace=True, n_samples=len(df_majority), random_state=42
                    )
                    X_balanced = pd.concat([df_majority, df_minority_upsampled])
                    y_balanced = pd.Series([0] * len(df_majority) + [1] * len(df_minority_upsampled))
                    
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X_balanced)
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_scaled, y_balanced, test_size=0.3, random_state=42, stratify=y_balanced
                    )
                    
                    # Use statsmodels for coefficient significance
                    X_train_sm = sm.add_constant(X_train)  # Add intercept
                    model_sm = sm.Logit(y_train, X_train_sm).fit(disp=0)  # disp=0 suppresses convergence output
                    
                    # Train sklearn model for predictions
                    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42, C=0.1)
                    model.fit(X_train, y_train)
                    
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    
                    print("\nClassification Report:")
                    print(classification_report(y_test, y_pred, digits=3))
                    
                    print("\nConfusion Matrix:")
                    print(confusion_matrix(y_test, y_pred))
                    
                    # Feature importance with significance
                    feature_names = ["const"] + numerical_features + \
                                [col for col in X_balanced.columns if col not in numerical_features]
                    coef_dict = dict(zip(feature_names, model_sm.params))
                    pval_dict = dict(zip(feature_names, model_sm.pvalues))
                    
                    # Inside analyze_multivariate, modify the feature importance print:
                    print("\nTop 30 Most Important Features (by absolute coefficient value) and All Race Categories:")
                    sorted_coefs = sorted(coef_dict.items(), key=lambda x: abs(x[1]), reverse=True)
                    top_30 = sorted_coefs[:30]
                    race_coefs = [(feat, coef) for feat, coef in sorted_coefs if feat.startswith("Race_")]

                    print("Note: Racial patterns reflect data distribution, not targeted focus.")
                    for feat, coef in top_30 + [r for r in race_coefs if r not in top_30]:  # Include all races
                        p_val = pval_dict.get(feat, 1.0)
                        sig_note = " (p < 0.05, significant)" if p_val < 0.05 else " (p >= 0.05, not significant)"
                        print(f"{feat}: {coef:.4f}{sig_note}")
                    
                    roc_auc = roc_auc_score(y_test, y_pred_proba)
                    print(f"\nROC-AUC Score: {roc_auc:.3f}")
                else:
                    print("No suspicious loans found for modeling.")
            else:
                print("Not enough classes for logistic regression.")
                        
        except Exception as e:
            print(f"Error in multivariate analysis: {str(e)}")

    def run_analysis(self) -> None:
        try:
            sus, full = self.load_data()
            print(
                f"\nStarting analysis of {len(sus):,} suspicious loans "
                f"out of {len(full):,} loans in the $5k-$22k range"
            )
            xgb_analyzer = XGBoostAnalyzer()

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
                ("XGBoost Analysis", lambda s, f: xgb_analyzer.analyze_with_xgboost(s, f, n_iter=5)),
                ("Feature Discrimination (AUPRC)", self.analyze_feature_discrimination)
            ]
            
            for title, func in analysis_steps:
                print(f"\nStarting {title} analysis...")
                try:
                    func(sus, full)
                except Exception as e:
                    print(f"Error in {title} analysis: {str(e)}")
            
            flag_rate = (len(sus) / len(full)) * 100
            print(
                f"\nOverall flagging rate: {flag_rate:.3f}% of loans in the "
                f"$5k-$22k range"
            )
            
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            raise

def main():
    """Main function to run the analysis."""
    # Lower process priority
    p = psutil.Process(os.getpid())
    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == 'nt' else 10)  # Windows or Unix
    
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