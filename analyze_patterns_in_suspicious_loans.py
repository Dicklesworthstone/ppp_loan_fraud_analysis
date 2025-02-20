import os
import psutil
import re
from typing import Tuple
import pandas as pd 
import numpy as np
from scipy import stats
from nameparser import HumanName
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.utils import resample
import xgboost as xgb
import shap
from joblib import Parallel, delayed, parallel_backend
from functools import lru_cache
from multiprocessing import Pool
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import dask.dataframe as dd

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
        return f"{test_name} p-value: {p_val:.6f}"
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
        self.category_feature_map = {}

    def _sanitize_feature_name(self, name: str) -> str:
        """Sanitize feature names to remove or replace characters not allowed by XGBoost."""
        # Replace invalid characters with underscores or remove them
        invalid_chars = r'[\[\]<>,;:]'
        sanitized = re.sub(invalid_chars, '_', name)
        # Remove multiple consecutive underscores and trim
        sanitized = re.sub(r'_+', '_', sanitized).strip('_')
        return sanitized
    
    def prepare_enhanced_features(self, df: pd.DataFrame, min_instances: int = 250) -> pd.DataFrame:
        """Prepare enhanced features for XGBoost, including useful categorical columns."""
        try:
            # Ensure unique indices and remove duplicates based on LoanNumber
            print("Starting prepare_enhanced_features - Initial shape:", df.shape)
            df = df.drop_duplicates(subset=["LoanNumber"], keep='first')
            df = df.reset_index(drop=True)
            print("After deduplication and index reset - Shape:", df.shape)

            # Basic numerical features with efficient types and proper NA handling
            df['JobsReported'] = (
                pd.to_numeric(df['JobsReported'].replace({pd.NA: np.nan}), errors='coerce')
                .fillna(0)
                .astype('int32')
            )

            df['InitialApprovalAmount'] = (
                pd.to_numeric(df['InitialApprovalAmount'].replace({pd.NA: np.nan}), errors='coerce')
                .fillna(0)
                .astype('float32')
            )
            df['NameLength'] = (
                df['BorrowerName']
                .fillna('')
                .astype(str)
                .str.len()
                .astype('int16')
            )

            df['WordCount'] = (
                df['BorrowerName']
                .fillna('')
                .astype(str)
                .apply(lambda x: len(x.split()))
                .astype('int8')
            )

            df['IsHubzone'] = (
                (df['HubzoneIndicator'].fillna('') == 'Y')
                .astype('uint8')
            )

            df['IsLMI'] = (
                (df['LMIIndicator'].fillna('') == 'Y')
                .astype('uint8')
            )

            df['IsNonProfit'] = (
                (df['NonProfit'].fillna('') == 'Y')
                .astype('uint8')
            )
            
            df['AmountPerEmployee'] = (
                df.apply(
                    lambda x: x['InitialApprovalAmount'] / x['JobsReported'] 
                    if x['JobsReported'] > 0 
                    else x['InitialApprovalAmount'],
                    axis=1
                )
                .fillna(0)
                .astype('float32')
            )
            
            df['IsRoundAmount'] = (
                pd.to_numeric(df['InitialApprovalAmount'], errors='coerce')
                .fillna(0)
                .apply(lambda x: x % 100 == 0)
            ).astype('uint8')

            # Fix NAType error by ensuring float conversion before int
            max_amounts = {20832, 20833, 20834}
            df['IsExactMaxAmount'] = (
                pd.to_numeric(df['InitialApprovalAmount'], errors='coerce')
                .fillna(0)
                .astype(float)  # Explicitly cast to float first
                .apply(lambda x: int(x) in max_amounts)  # Then convert to int
            ).astype('uint8')

            print("InitialApprovalAmount null values:", df['InitialApprovalAmount'].isna().sum())
            print("InitialApprovalAmount type:", df['InitialApprovalAmount'].dtype)

            # Boolean indicators (redundant code removed, already defined above)
            
            # Address-based features
            residential_indicators = {'apt', 'unit', 'suite', '#', 'po box', 'residence', 'residential', 'apartment', 'room', 'floor'}
            commercial_indicators = {'plaza', 'building', 'tower', 'office', 'complex', 'center', 'mall', 'commercial', 'industrial', 'park'}
            address_str = df['BorrowerAddress'].astype(str).str.lower()
            df['HasResidentialIndicator'] = address_str.apply(lambda x: any(ind in x for ind in residential_indicators)).astype('uint8')
            df['HasCommercialIndicator'] = address_str.apply(lambda x: any(ind in x for ind in commercial_indicators)).astype('uint8')
            
            # Business concentration
            address_key = (
                df[['BorrowerAddress', 'BorrowerCity', 'BorrowerState']]
                .fillna('')
                .astype(str)
                .agg('_'.join, axis=1)
            )
            address_counts = address_key.value_counts()
            df['BusinessesAtAddress'] = address_key.map(address_counts).fillna(0).astype('int16')

            # Business name pattern
            name_str = df['BorrowerName'].fillna('').astype(str).str.lower()
            suspicious_keywords = {'consulting', 'holdings', 'enterprise', 'solutions', 'services', 'investment', 'trading', 'group', 'international', 'global'}
            df['HasSuspiciousKeyword'] = name_str.apply(lambda x: any(kw in x for kw in suspicious_keywords)).astype('uint8')

            # Demographic completeness
            demographic_fields = ['Race', 'Gender', 'Ethnicity']
            df['MissingDemographics'] = df[demographic_fields].isna().sum(axis=1).astype('uint8')

            # Categorical features
            categorical_features = [
                'BusinessType', 'Race', 'Gender', 'Ethnicity', 'BorrowerState', 
                'BorrowerCity', 'NAICSCode', 'OriginatingLender'
            ]
            self.categorical_features = categorical_features
            self.category_feature_map = {}

            for feature in categorical_features:
                if feature not in df.columns or df[feature].isna().all():
                    print(f"Skipping {feature}: not present or all NaN")
                    continue
                
                df[feature] = df[feature].fillna('Unknown').astype(str)
                value_counts = df[feature].value_counts()
                common_categories = value_counts[value_counts >= min_instances].index
                print(f"{feature}: {len(value_counts)} total categories, {len(common_categories)} with >= {min_instances} instances")
                
                # Special handling for high-cardinality features
                if feature == 'OriginatingLender':
                    top_lenders = value_counts.head(20).index
                    df[feature] = df[feature].apply(lambda x: x if x in top_lenders else 'Other_Lender')
                    common_categories = top_lenders.union(['Other_Lender'])
                elif feature == 'BorrowerCity':
                    top_cities = value_counts.head(100).index
                    df[feature] = df[feature].apply(lambda x: x if x in top_cities else 'Other_City')
                    common_categories = top_cities.union(['Other_City'])
                elif feature == 'NAICSCode':
                    top_codes = value_counts.head(50).index
                    df[feature] = df[feature].apply(lambda x: x if x in top_codes else 'Other_NAICS')
                    common_categories = top_codes.union(['Other_NAICS'])
                
                df[feature] = df[feature].apply(lambda x: x if x in common_categories else f'Other_{feature}')
                dummies = pd.get_dummies(df[feature], dtype='uint8')
                
                # Sanitize dummy column names
                dummies.columns = [
                    self._sanitize_feature_name(
                        f"{feature}_{col}" if col != f'Other_{feature}' else f"{feature}_Other"
                    ) for col in dummies.columns
                ]
                
                for col in dummies.columns:
                    if any(c in col for c in ['[', ']', '<']):
                        print(f"Warning: Sanitized feature name still contains invalid characters: {col}")
                    self.category_feature_map[col] = feature
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(columns=[feature])

            # Drop unnecessary columns
            drop_cols = ['BorrowerName', 'BorrowerAddress', 'LoanNumber', 'Location']
            df = df.drop(columns=[col for col in drop_cols if col in df.columns])
            
            self.feature_names = [self._sanitize_feature_name(col) for col in df.columns.tolist()]
            df.columns = self.feature_names
            print(f"Prepared feature matrix shape: {df.shape}, Memory usage: {df.memory_usage().sum() / (1024**3):.2f} GiB")
            print("Index duplicates after preparation:", df.index.duplicated().sum())
            return df
            
        except Exception as e:
            print(f"Error preparing enhanced features for XGBoost analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return df
    
    def analyze_with_xgboost(self, sus: pd.DataFrame, full: pd.DataFrame, n_iter: int = 10, min_instances: int = 250) -> None:
        try:
            print("\nEnhanced XGBoost Analysis")
            full_prepared = self.prepare_enhanced_features(full.copy(), min_instances=min_instances)
            print("Full prepared shape after feature prep:", full_prepared.shape)
            
            # Debugging checks
            assert not full.index.duplicated().any(), "full has duplicate indices"
            assert not full_prepared.index.duplicated().any(), "full_prepared has duplicate indices"
            full_prepared["LoanNumber"] = full["LoanNumber"].astype(str).str.strip()
                        
            # Ensure LoanNumber consistency and reset index
            full_prepared["LoanNumber"] = full["LoanNumber"].astype(str).str.strip()
            sus["LoanNumber"] = sus["LoanNumber"].astype(str).str.strip()
            full_prepared = full_prepared.reset_index(drop=True)
            print("Full prepared index duplicates:", full_prepared.index.duplicated().sum())
            
            full_prepared["Flagged"] = full_prepared["LoanNumber"].isin(sus["LoanNumber"]).astype('uint8')
            print(f"Flagged loans: {full_prepared['Flagged'].sum()}, Total: {len(full_prepared)}")
            
            # Drop LoanNumber after flagging
            full_prepared = full_prepared.drop(columns=["LoanNumber"])
            
            # Filter feature columns
            feature_cols = [col for col in full_prepared.columns 
                            if col not in ["Flagged", "RiskScore", "RiskFlags"]
                            and full_prepared[col].dtype in ['int8', 'int16', 'int32', 'uint8', 'float32']]
                    
            X = full_prepared[feature_cols]
            y = full_prepared["Flagged"]
            print(f"Feature matrix ready. Shape: {X.shape}, Columns: {len(feature_cols)}")

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

            param_grid = {
                'max_depth': [3, 5],
                'learning_rate': [0.01, 0.1],
                'n_estimators': [100, 200],
                'min_child_weight': [1, 3],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'scale_pos_weight': [sum(y == 0) / sum(y == 1) if sum(y == 1) > 0 else 1]
            }
            
            xgb_clf = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='aucpr',
                random_state=42,
                tree_method='hist',
                n_jobs=12,
                max_bin=256
            )
            
            random_search = RandomizedSearchCV(
                xgb_clf,
                param_distributions=param_grid,
                n_iter=n_iter,
                scoring='average_precision',
                cv=2,
                random_state=42,
                n_jobs=9,
                verbose=2,
                error_score='raise'
            )
            
            with parallel_backend('loky', n_jobs=4):
                print("Performing hyperparameter tuning...")
                random_search.fit(X_train, y_train)
            
            best_params = random_search.best_params_
            print(f"Best parameters from tuning: {best_params}")
            
            self.model = xgb.XGBClassifier(
                **best_params,
                objective='binary:logistic',
                eval_metric='aucpr',
                random_state=42,
                tree_method='hist',
                n_jobs=12,
                max_bin=256
            )
            
            print("Fitting final model...")
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
            )
            
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            avg_precision_recall = average_precision_score(y_test, y_pred_proba)
            f1_score_result = f1_score(y_test, y_pred)
            
            print("\nModel Performance:")
            print(f"ROC-AUC Score: {roc_auc:.3f}")
            print(f"Area Under Precision Recall Curve: {avg_precision_recall:.3f}")
            print(f"F1 Score: {f1_score_result:.3f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, digits=3))
            
            self._analyze_feature_importance(X, feature_cols)
            self._analyze_shap_values(X_test, feature_cols)
            
            # Reset index before prediction to avoid duplicate label issues
            full_prepared = full_prepared.reset_index(drop=True)
            full_prepared['PredictedProbability'] = self.model.predict_proba(full_prepared[feature_cols])[:, 1]
            self._analyze_high_probability_patterns(full_prepared)
            
        except KeyboardInterrupt:
            print("\nXGBoost analysis interrupted by user")
            return
        except Exception as e:
            print(f"Error in XGBoost analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def _analyze_feature_importance(self, X: pd.DataFrame, feature_cols: list) -> None:
        try:
            importance_df = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': self.model.feature_importances_
            })
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            numerical_features = [f for f in feature_cols if f not in self.category_feature_map]
            categorical_features = [f for f in feature_cols if f in self.category_feature_map]
            
            print("\nTop Numerical Features:")
            num_importance = importance_df[importance_df['Feature'].isin(numerical_features)].head(15)
            for _, row in num_importance.iterrows():
                print(f"  {row['Feature']}: {row['Importance']:.4f}")
            
            categorical_by_base = {}
            for f in categorical_features:
                base = self.category_feature_map[f]
                if base not in categorical_by_base:
                    categorical_by_base[base] = []
                importance = importance_df[importance_df['Feature'] == f]['Importance'].values[0]
                categorical_by_base[base].append((f, importance))
            
            for base, feat_list in categorical_by_base.items():
                print(f"\nAll Categories for {base}:")
                if base == "Race":
                    print("  Note: Racial patterns reflect data distribution, not targeted focus.")
                sorted_feats = sorted(feat_list, key=lambda x: x[1], reverse=True)
                for i, (feat, imp) in enumerate(sorted_feats):
                    prefix = "**" if i == 0 else "  "
                    suffix = "**" if i == 0 else ""
                    print(f"    {prefix}{feat}: {imp:.4f}{suffix}")
                
        except Exception as e:
            print(f"Error analyzing feature importance: {str(e)}")

    def _analyze_shap_values(self, X_test: pd.DataFrame, feature_cols: list) -> None:
        try:
            print("\nAnalyzing feature interactions with SHAP values...")
            explainer = shap.TreeExplainer(self.model)
            
            # Ensure X_test has the same columns as used in training
            expected_features = self.model.feature_names if self.model.feature_names else feature_cols
            X_test_aligned = X_test[expected_features]  # Select only the expected features
            
            # Subsample X_test (e.g., 10,000 rows)
            sample_size = min(10000, len(X_test_aligned))  # Use 10k or total size if smaller
            X_test_sample = X_test_aligned.sample(n=sample_size, random_state=42)
            print(f"Using a subsample of {sample_size:,} rows for SHAP analysis (original size: {len(X_test):,})")
            
            # Debugging output to verify dimensions
            print(f"Expected features from model: {len(expected_features)}")
            print(f"X_test_sample shape: {X_test_sample.shape}")
            print(f"X_test_sample columns: {X_test_sample.columns.tolist()}")

            # Compute SHAP interaction values
            interaction_values = explainer.shap_interaction_values(X_test_sample)
            
            # Zero out diagonal (self-interactions)
            np.fill_diagonal(interaction_values, 0)
            interaction_sum = np.sum(np.abs(interaction_values), axis=(0, 1))
            
            # Create DataFrame of interaction strengths
            top_interactions = pd.DataFrame({
                'Feature': expected_features,  # Use the aligned feature names
                'Interaction_Strength': interaction_sum
            }).sort_values('Interaction_Strength', ascending=False)
            
            # Separate numerical and categorical features
            numerical_features = [f for f in expected_features if f not in self.category_feature_map]
            categorical_features = [f for f in expected_features if f in self.category_feature_map]
            
            # Print top numerical feature interactions
            print("\nTop Numerical Feature Interactions:")
            num_interactions = top_interactions[top_interactions['Feature'].isin(numerical_features)].head(5)
            for _, row in num_interactions.iterrows():
                print(f"  {row['Feature']}: {row['Interaction_Strength']:.4f}")
            
            # Group and print categorical feature interactions
            categorical_by_base = {}
            for f in categorical_features:
                base = self.category_feature_map[f]
                if base not in categorical_by_base:
                    categorical_by_base[base] = []
                strength = top_interactions[top_interactions['Feature'] == f]['Interaction_Strength'].values[0]
                categorical_by_base[base].append((f, strength))
            
            for base, feat_list in categorical_by_base.items():
                print(f"\nAll Categories for {base} Interactions:")
                if base == "Race":
                    print("  Note: Racial patterns reflect data distribution, not targeted focus.")
                sorted_feats = sorted(feat_list, key=lambda x: x[1], reverse=True)
                for i, (feat, strength) in enumerate(sorted_feats):
                    prefix = "**" if i == 0 else "  "
                    suffix = "**" if i == 0 else ""
                    print(f"    {prefix}{feat}: {strength:.4f}{suffix}")
                    
        except Exception as e:
            print(f"Error analyzing SHAP values: {str(e)}")
            raise  # Re-raise the exception for further debugging if needed

    def _analyze_high_probability_patterns(self, df: pd.DataFrame) -> None:
        try:
            high_prob_threshold = df['PredictedProbability'].quantile(0.99)
            high_prob_loans = df[df['PredictedProbability'] >= high_prob_threshold]
            
            print("\nAnalysis of High-Probability Suspicious Loans:")
            print(f"Number of high-probability loans: {len(high_prob_loans)}")
            
            numerical_features = ['InitialApprovalAmount', 'JobsReported', 'AmountPerEmployee', 'BusinessesAtAddress']
            print("\nNumerical Feature Patterns:")
            for feature in numerical_features:
                mean_all = df[feature].mean()
                mean_high_prob = high_prob_loans[feature].mean()
                print(f"{feature}:")
                print(f"  All loans mean: {mean_all:.3f}")
                print(f"  High-prob loans mean: {mean_high_prob:.3f}")
                print(f"  Ratio: {mean_high_prob/mean_all:.2f}x")
            
            if self.categorical_features:
                print("\nCategorical Feature Patterns in High-Probability Loans:")
                print("Note: Patterns reflect data distribution, not targeted focus on any group.")
                for base_feature in self.categorical_features:
                    print(f"\n{base_feature} Distribution:")
                    if base_feature == "Race":
                        print("  Note: Racial patterns reflect data distribution, not targeted focus.")
                    dummy_cols = [col for col in df.columns if col.startswith(f"{base_feature}_")]
                    category_counts = {}
                    for col in dummy_cols:
                        category = col.replace(f"{base_feature}_", "")
                        count = high_prob_loans[col].sum()
                        if count > 0:
                            category_counts[category] = count
                    total = len(high_prob_loans)
                    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
                        percentage = (count / total) * 100
                        print(f"  {category}: {count:,} ({percentage:.3f}%)")
                        
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
        print("Loading suspicious loans data...")
        try:
            # Define columns to load for suspicious loans
            sus_cols = [
                "LoanNumber", "BorrowerName", "RiskScore", "RiskFlags", "BorrowerAddress", "BorrowerCity", "BorrowerState",
                "BorrowerZip", "Latitude", "Longitude", "Census Tract Code", "OriginatingLender",
                "InitialApprovalAmount", "BusinessType", "Race", "Gender", "Ethnicity", "NAICSCode",
                "JobsReported", "HubzoneIndicator", "LMIIndicator", "NonProfit"
            ]
            sus_dtypes = {
                "LoanNumber": str,
                "BorrowerName": str,
                "RiskScore": "float32",
                "RiskFlags": str,
                "BorrowerAddress": str,
                "BorrowerCity": str,
                "BorrowerState": str,
                "BorrowerZip": str,
                "Latitude": "float32",
                "Longitude": "float32",
                "Census Tract Code": str,
                "OriginatingLender": str,
                "InitialApprovalAmount": "float32",
                "BusinessType": str,
                "Race": str,
                "Gender": str,
                "Ethnicity": str,
                "NAICSCode": str,
                "JobsReported": "Int32",
                "HubzoneIndicator": str,
                "LMIIndicator": str,
                "NonProfit": str,
            }
            # Load suspicious data with pandas, handling missing columns
            try:
                sus = pd.read_csv(
                    self.suspicious_file,
                    engine='pyarrow',
                    dtype=sus_dtypes,
                    usecols=sus_cols
                )
            except ValueError as e:
                print(f"Warning: Some columns are missing in suspicious loans data: {str(e)}")
                available_cols = [col for col in sus_cols if col in pd.read_csv(self.suspicious_file, nrows=0).columns]
                sus = pd.read_csv(
                    self.suspicious_file,
                    engine='pyarrow',
                    dtype={col: sus_dtypes[col] for col in available_cols if col in sus_dtypes},
                    usecols=available_cols
                )
            sus = sus.reset_index(drop=True)  # Ensure unique index

        except Exception as e:
            print(f"Error loading suspicious data: {str(e)}")
            raise

        print("Loading full loan dataset with Dask...")
        try:
            full_cols = [
                "LoanNumber", "BorrowerName", "BorrowerAddress", "BorrowerCity", "BorrowerState",
                "BorrowerZip", "Latitude", "Longitude", "Census Tract Code", "OriginatingLender",
                "InitialApprovalAmount", "BusinessType", "Race", "Gender", "Ethnicity", "NAICSCode",
                "JobsReported", "HubzoneIndicator", "LMIIndicator", "NonProfit"
            ]
            full_dtypes = {
                "LoanNumber": str,
                "BorrowerName": str,
                "BorrowerAddress": str,
                "BorrowerCity": str,
                "BorrowerState": str,
                "BorrowerZip": str,
                "Latitude": "float32",
                "Longitude": "float32",
                "Census Tract Code": str,
                "OriginatingLender": str,
                "InitialApprovalAmount": "float32",
                "BusinessType": str,
                "Race": str,
                "Gender": str,
                "Ethnicity": str,
                "NAICSCode": str,
                "JobsReported": "Int32",
                "HubzoneIndicator": str,
                "LMIIndicator": str,
                "NonProfit": str,
            }
            full_dd = dd.read_csv(
                self.full_data_file,
                usecols=full_cols,
                dtype=full_dtypes,
                blocksize="64MB"
            )
            full_dd = full_dd[
                (full_dd["InitialApprovalAmount"] >= 5000) &
                (full_dd["InitialApprovalAmount"] < 22000)
            ]
            full_dd["JobsReported"] = full_dd["JobsReported"].fillna(0)
            full = full_dd.compute(scheduler='processes', num_workers=32)
            full = full.reset_index(drop=True)  # Ensure unique index

        except Exception as e:
            print(f"Error loading full data: {str(e)}")
            raise

        # Debugging checks
        print(f"Suspicious dataset shape: {sus.shape}")
        print(f"Full dataset shape after filtering: {full.shape}")
        print(f"Suspicious LoanNumbers unique: {sus['LoanNumber'].nunique()}")
        print(f"Full LoanNumbers unique: {full['LoanNumber'].nunique()}")

        self.sus_data = sus.copy()
        self.full_data = full.copy()
        print(f"Loaded {len(sus):,} suspicious loans and {len(full):,} total loans in range.")
        return sus, full

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
            print(f"Chi-square test p-value: {p_chi2 if p_chi2 is not None else 'N/A' :.6f}")
            
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
        """Analyze feature discrimination using AUPRC with meaningful category names."""
        try:
            print("\nFeature Discrimination Analysis using AUPRC")
            
            # Prepare enhanced features with unique indices
            full_prepared = self.prepare_enhanced_features(full.copy())
            print("Full prepared shape after feature prep:", full_prepared.shape)
            full_prepared = full_prepared.drop_duplicates(subset=["LoanNumber"], keep='first')
            full_prepared = full_prepared.reset_index(drop=True)
            print("After deduplication and index reset - Shape:", full_prepared.shape)
            
            # Debugging checks
            assert not full.index.duplicated().any(), "full has duplicate indices"
            assert not full_prepared.index.duplicated().any(), "full_prepared has duplicate indices"
            full_prepared["LoanNumber"] = full["LoanNumber"].astype(str).str.strip()
                        
            full_prepared["LoanNumber"] = full["LoanNumber"].astype(str).str.strip()
            sus["LoanNumber"] = sus["LoanNumber"].astype(str).str.strip()
            
            full_prepared["Flagged"] = full_prepared["LoanNumber"].isin(sus["LoanNumber"]).astype(int)
            print(f"Flagged loans: {full_prepared['Flagged'].sum()}, Total: {len(full_prepared)}")
            
            # Drop LoanNumber after flagging
            full_prepared = full_prepared.drop(columns=["LoanNumber"])
            
            # Define features to analyze
            features_to_analyze = [
                "InitialApprovalAmount", "JobsReported", "NameLength", "WordCount",
                "AmountPerEmployee", "HasResidentialIndicator", "HasCommercialIndicator",
                "BusinessesAtAddress", "IsExactMaxAmount", "IsRoundAmount",
                "HasSuspiciousKeyword", "MissingDemographics", "IsHubzone", "IsLMI", "IsNonProfit",
                "BusinessType", "Race", "Gender", "Ethnicity"
            ]
            
            y_true = full_prepared["Flagged"]
            feature_auprc = {}
            category_auprc_dict = {}
            
            # Calculate AUPRC and significance for each feature
            for feature in features_to_analyze:
                if feature not in full_prepared.columns and feature not in full.columns:
                    continue
                    
                if feature in ['BusinessType', 'Race', 'Gender', 'Ethnicity']:
                    original_col = full[feature].fillna('Unknown').astype(str)
                    value_counts = original_col.value_counts()
                    common_categories = value_counts[value_counts >= 5].index
                    temp_df = original_col.apply(lambda x: x if x in common_categories else 'Other')
                    
                    X = pd.get_dummies(temp_df)
                    X.columns = [f"{feature}_{col.replace(' ', '_')}" for col in X.columns]
                    X = X.reset_index(drop=True)  # Ensure X has unique indices
                    
                    if X.shape[1] > 1:
                        for col in X.columns:
                            single_feature = X[[col]]
                            model = LogisticRegression(max_iter=1000, random_state=42)
                            model.fit(single_feature, y_true)
                            y_scores = model.predict_proba(single_feature)[:, 1]
                            auprc = average_precision_score(y_true, y_scores)
                            p_val = stats.ttest_ind(
                                single_feature[col][y_true == 1],
                                single_feature[col][y_true == 0],
                                equal_var=False
                            ).pvalue
                            feature_auprc[col] = (auprc, p_val)
                            if feature not in category_auprc_dict:
                                category_auprc_dict[feature] = {}
                            category_auprc_dict[feature][col] = auprc
                    else:
                        y_scores = X.iloc[:, 0]
                        auprc = average_precision_score(y_true, y_scores)
                        p_val = stats.ttest_ind(
                            X.iloc[:, 0][y_true == 1],
                            X.iloc[:, 0][y_true == 0],
                            equal_var=False
                        ).pvalue
                        feature_auprc[feature] = (auprc, p_val)
                else:
                    X = full_prepared[[feature]].fillna(0)
                    X = X.reset_index(drop=True)  # Ensure X has unique indices
                    model = LogisticRegression(max_iter=1000, random_state=42)
                    model.fit(X, y_true)
                    y_scores = model.predict_proba(X)[:, 1]
                    auprc = average_precision_score(y_true, y_scores)
                    p_val = stats.ttest_ind(
                        X[feature][y_true == 1],
                        X[feature][y_true == 0],
                        equal_var=False
                    ).pvalue
                    feature_auprc[feature] = (auprc, p_val)
            
            # Sort and display results
            sorted_features = sorted(feature_auprc.items(), key=lambda x: x[1][0], reverse=True)
            
            print("\nDetailed Feature Analysis:")
            print("Note: Patterns reflect data distribution and statistical significance, not targeted focus on any group.")
            
            displayed_categories = set()
            for feature, (auprc, p_val) in sorted_features:
                sig_note = " (p < 0.05, significant)" if p_val < 0.05 else " (p >= 0.05, not significant)"
                print(f"{feature}: AUPRC = {auprc:.4f}{sig_note}")
                
                base_feature = feature.split('_')[0] if '_' in feature else feature
                if base_feature in category_auprc_dict and base_feature not in displayed_categories:
                    print(f"  All categories for {base_feature}:")
                    if base_feature == "Race":
                        print("  Note: Racial patterns reflect data distribution, not targeted focus.")
                    sorted_categories = sorted(
                        category_auprc_dict[base_feature].items(),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    for i, (cat, cat_auprc) in enumerate(sorted_categories):
                        prefix = "**" if i == 0 else "  "
                        suffix = "**" if i == 0 else ""
                        print(f"    {prefix}{cat}: AUPRC = {cat_auprc:.4f}{suffix}")
                    displayed_categories.add(base_feature)
            
            print("\nFeatures ranked by AUPRC (discriminative power):")
            for feature, (auprc, _) in sorted_features:
                print(f"{feature}: AUPRC = {auprc:.4f}")
            
            baseline_auprc = y_true.mean()
            print(f"\nBaseline AUPRC (random guessing): {baseline_auprc:.4f}")
            
        except Exception as e:
            print(f"Error in feature discrimination analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

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

            self.analyze_categorical_patterns(sus, full, "NonProfit", "Non-Profit Status Patterns")
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
            if df['InitialApprovalAmount'].isna().any():
                print("Warning: NaN values remain in InitialApprovalAmount")
                df['InitialApprovalAmount'] = df['InitialApprovalAmount'].fillna(0)            
                        
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
        """Prepare enhanced features for multivariate logit analysis with safe NA handling."""
        try:
            df = df.copy()
            print(f"Starting prepare_enhanced_features - Initial shape: {df.shape}")

            # Numeric columns
            df['JobsReported'] = pd.to_numeric(df['JobsReported'].replace({pd.NA: np.nan}), errors='coerce').fillna(0)
            df['InitialApprovalAmount'] = pd.to_numeric(df['InitialApprovalAmount'].replace({pd.NA: np.nan}), errors='coerce').fillna(0)

            # BorrowerName features
            df['BorrowerName'] = df['BorrowerName'].fillna('')
            df['NameLength'] = df['BorrowerName'].astype(str).str.len()
            df['WordCount'] = df['BorrowerName'].str.split().str.len().fillna(0).astype(int)

            # Boolean columns
            df['IsHubzone'] = (df['HubzoneIndicator'].fillna('') == 'Y').astype('uint8')
            df['IsLMI'] = (df['LMIIndicator'].fillna('') == 'Y').astype('uint8')
            df['IsNonProfit'] = (df['NonProfit'].fillna('') == 'Y').astype('uint8')

            # AmountPerEmployee
            df['AmountPerEmployee'] = np.where(
                df['JobsReported'] > 0,
                df['InitialApprovalAmount'] / df['JobsReported'],
                df['InitialApprovalAmount']
            )

            # IsRoundAmount
            df['IsRoundAmount'] = (
                pd.to_numeric(df['InitialApprovalAmount'], errors='coerce')
                .fillna(0)
                .apply(lambda x: x % 100 == 0)
            ).astype('uint8')

            # Address features
            df['BorrowerAddress'] = df['BorrowerAddress'].fillna('')
            df['BorrowerCity'] = df['BorrowerCity'].fillna('')
            df['BorrowerState'] = df['BorrowerState'].fillna('')

            residential_indicators = {'apt', 'unit', 'suite', '#', 'po box', 'residence', 'residential', 'apartment', 'room', 'floor'}
            commercial_indicators = {'plaza', 'building', 'tower', 'office', 'complex', 'center', 'mall', 'commercial', 'industrial', 'park'}
            address_str = df['BorrowerAddress'].astype(str).str.lower()

            residential_pattern = '|'.join(map(re.escape, residential_indicators))
            commercial_pattern = '|'.join(map(re.escape, commercial_indicators))

            df['HasResidentialIndicator'] = address_str.str.contains(residential_pattern, case=False, na=False).astype('uint8')
            df['HasCommercialIndicator'] = address_str.str.contains(commercial_pattern, case=False, na=False).astype('uint8')

            # HasMultipleBusinesses
            address_counts = (df.groupby(['BorrowerAddress', 'BorrowerCity', 'BorrowerState'])
                            .size()
                            .reset_index(name='Count'))
            df = df.merge(address_counts, on=['BorrowerAddress', 'BorrowerCity', 'BorrowerState'], how='left')
            df['Count'] = df['Count'].fillna(0).astype(int)
            df['HasMultipleBusinesses'] = (df['Count'] > 1).astype(int)
            df = df.drop(columns=['Count'])

            # Exact maximum amounts
            max_amounts = [20832, 20833, 20834]
            df['IsExactMaxAmount'] = df['InitialApprovalAmount'].apply(lambda x: int(x) in max_amounts).astype(int)

            # Categorical variables
            df['BusinessType'] = df['BusinessType'].fillna('Unknown')
            df['Race'] = df['Race'].fillna('Unknown')
            df['Gender'] = df['Gender'].fillna('Unknown')
            df['Ethnicity'] = df['Ethnicity'].fillna('Unknown')

            # MissingDemographics
            demographic_fields = ['Race', 'Gender', 'Ethnicity']
            df['MissingDemographics'] = (df[demographic_fields] == 'Unknown').sum(axis=1).astype('uint8')
            df = df.reset_index(drop=True)
            return df

        except Exception as e:
            print(f"Error preparing enhanced features for multivariate logit analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def analyze_multivariate(self, sus: pd.DataFrame, full: pd.DataFrame) -> None:
        """Perform multivariate analysis using logistic regression with meaningful category names."""
        try:
            print("\nEnhanced Multivariate Analysis via Logistic Regression")
            
            print("Preparing enhanced features for multivariate analysis...")
            full_prepared = self.prepare_enhanced_features(full.copy())

            # Remove duplicates
            full_prepared = full_prepared.drop_duplicates(subset=["LoanNumber"], keep='first')

            # Fix: Reset index of full to ensure no duplicates and proper alignment
            full = full.reset_index(drop=True)
            full_prepared["LoanNumber"] = full["LoanNumber"].astype(str).str.strip()
            sus["LoanNumber"] = sus["LoanNumber"].astype(str).str.strip()

            full_prepared = self.prepare_enhanced_features(full.copy()).reset_index(drop=True)

            # Debugging output
            print(f"Suspicious LoanNumbers unique count: {sus['LoanNumber'].nunique()}")
            print(f"Full dataset LoanNumbers unique count: {full_prepared['LoanNumber'].nunique()}")
            matched_loans = set(sus["LoanNumber"]).intersection(set(full_prepared["LoanNumber"]))
            print(f"Number of matched loan numbers: {len(matched_loans)}")
            
            full_prepared["Flagged"] = full_prepared["LoanNumber"].isin(matched_loans).astype(int)
            print(f"Class distribution - Suspicious: {full_prepared['Flagged'].sum()}, "
                f"Non-suspicious: {len(full_prepared) - full_prepared['Flagged'].sum()}")

            if full_prepared["Flagged"].sum() == 0:
                print("Error: No suspicious loans found in the full dataset after flagging.")
                return

            # Define features - reduced set to target ~30 columns
            numerical_features = [
                "InitialApprovalAmount", "JobsReported", "NameLength", "WordCount",
                "AmountPerEmployee", "HasResidentialIndicator", "HasCommercialIndicator",
                "BusinessesAtAddress", "IsExactMaxAmount", "IsRoundAmount",
                "HasSuspiciousKeyword", "MissingDemographics", "IsHubzone", "IsLMI", "IsNonProfit"
            ]
            categorical_features = ["BusinessType", "Race", "Gender", "Ethnicity"]  # Reduced from 8 to 4

            # Filter available features
            numerical_features = [f for f in numerical_features if f in full_prepared.columns]
            categorical_features = [f for f in categorical_features if f in full.columns]

            # Drop LoanNumber now that flagging is complete
            if "LoanNumber" in full_prepared.columns:
                print("Dropping LoanNumber column...")
                full_prepared = full_prepared.drop(columns=["LoanNumber"])

            # Initial feature matrix with numerical features
            print("Creating initial feature matrix X...")
            X = full_prepared[numerical_features].copy()
            y = full_prepared["Flagged"]
            print(f"Initial X shape: {X.shape}, Columns: {X.columns.tolist()}")

            # Diagnose and clean numerical features
            print("\nDiagnosing feature variance and correlations...")
            variances = X.var()
            low_variance_cols = variances[variances < 0.01].index.tolist()  # Tightened from 0.005 to 0.01
            if low_variance_cols:
                print(f"Removing low-variance numerical features: {low_variance_cols}")
                X = X.drop(columns=low_variance_cols)
                numerical_features = [f for f in numerical_features if f not in low_variance_cols]

            # Address multicollinearity explicitly
            corr_matrix = X.corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            high_corr_pairs = [((row_idx, col_idx), value) for (row_idx, col_idx), value in upper_tri.stack().items() if value > 0.85]  # Tightened from 0.9 to 0.85
            if high_corr_pairs:
                print("High correlations detected (r > 0.85):")
                to_drop = set()
                for (row_idx, col_idx), corr_value in high_corr_pairs:
                    print(f"  {row_idx} - {col_idx}: {corr_value:.3f}")
                    to_drop.add(col_idx)  # Remove the second column in highly correlated pairs
                print(f"Removing correlated features: {to_drop}")
                X = X.drop(columns=to_drop)
                numerical_features = [f for f in numerical_features if f not in to_drop]

            # Handle categorical features with meaningful names
            category_feature_map = {}
            for feature in categorical_features:
                print(f"\nProcessing categorical feature: {feature}")
                value_counts = full[feature].value_counts()
                print(f"  Unique categories: {len(value_counts)}, Min count: {value_counts.min()}")
                
                min_occurrences = max(50, int(len(full) * 0.001))
                common_categories = value_counts[value_counts >= min_occurrences].index
                temp_df = full[feature].fillna('Unknown').astype(str).apply(
                    lambda x: x if x in common_categories else 'Other'
                )
                
                dummies = pd.get_dummies(temp_df)
                dummies.columns = [f"{feature}_{col.replace(' ', '_')}" for col in dummies.columns]
                
                dummy_variances = dummies.var()
                low_variance_dummies = dummy_variances[dummy_variances < 0.001].index.tolist()
                if low_variance_dummies:
                    print(f"  Removing low-variance dummy columns: {low_variance_dummies}")
                    dummies = dummies.drop(columns=low_variance_dummies)
                
                for col in dummies.columns:
                    if dummies[col].var() > 0:
                        category_feature_map[col] = feature
                
                if dummies.shape[1] > 0:
                    print(f"  Concatenating dummies for {feature}...")
                    X = pd.concat([X, dummies], axis=1)
                    print(f"  After concat shape: {X.shape}")

                # Final feature matrix check
                print(f"\nFinal feature matrix shape: {X.shape}")
                if X.shape[1] == 0:
                    print("Error: No valid features remain after preprocessing.")
                    exit()

                # Ensure X is float64 for VIF compatibility
                X = X.astype(float)

                # Handle NaNs and infinities
                if X.isna().any().any():
                    print("Warning: NaNs found in X. Filling with 0.")
                    X = X.fillna(0)
                if np.isinf(X).any().any():
                    print("Warning: Infinities found in X. Replacing with large finite numbers.")
                    X = X.replace([np.inf, -np.inf], [1e10, -1e10])

                # Remove zero-variance features
                variances = X.var()
                zero_variance_cols = variances[variances == 0].index.tolist()
                if zero_variance_cols:
                    print(f"Removing zero-variance features: {zero_variance_cols}")
                    X = X.drop(columns=zero_variance_cols)
                    # Update feature lists as needed (e.g., numerical_features, category_feature_map)

                print("\nChecking for multicollinearity using Correlation Matrix and Targeted VIF...")

                # Step 1: Compute correlation matrix
                corr_matrix = X.corr().abs()

                # Step 2: Identify features with high correlations (|corr| > 0.85)
                high_corr_threshold = 0.85
                high_corr_features = set()
                for i in range(len(corr_matrix.columns)):
                    for j in range(i + 1, len(corr_matrix.columns)):
                        if corr_matrix.iloc[i, j] > high_corr_threshold:
                            high_corr_features.add(corr_matrix.columns[i])
                            high_corr_features.add(corr_matrix.columns[j])

                # Step 3: Targeted VIF calculation
                vif_threshold = 10
                features_to_check = list(high_corr_features)
                while len(features_to_check) > 1:  # Stop if 1 or fewer features remain
                    X_subset = X[features_to_check]
                    vif_data = pd.DataFrame()
                    vif_data["feature"] = features_to_check
                    vif_data["VIF"] = [variance_inflation_factor(X_subset.values, i) 
                                    for i in range(len(features_to_check))]
                    high_vif = vif_data[vif_data["VIF"] > vif_threshold]
                    
                    if high_vif.empty:
                        print("No multicollinearity issues in targeted features (all VIFs ≤ 10).")
                        break
                    else:
                        # Remove feature with highest VIF
                        feature_to_remove = high_vif.sort_values("VIF", ascending=False).iloc[0]["feature"]
                        print(f"Removing {feature_to_remove} (VIF: {high_vif[high_vif['feature'] == feature_to_remove]['VIF'].values[0]:.2f})")
                        X = X.drop(columns=[feature_to_remove])
                        features_to_check.remove(feature_to_remove)
                        # Update other feature lists (e.g., numerical_features, category_feature_map) as needed

                # Handle case where only one feature remains
                if len(features_to_check) == 1:
                    print(f"Only one feature remains in targeted set: {features_to_check[0]}. No VIF calculation needed.")

                print(f"Feature matrix after multicollinearity check: {X.shape}, Columns: {X.columns.tolist()}")

                # Proceed with scaling and modeling
                if X.shape[1] == 0:
                    print("Error: No valid features remain after multicollinearity check.")
                    return
                                
            # Handle class imbalance and scaling
            if len(y.unique()) >= 2:
                df_majority = X[y == 0]
                df_minority = X[y == 1]
                if len(df_minority) > 0:
                    print(f"Upsampling minority class from {len(df_minority):,} to {len(df_majority):,}")
                    df_minority_upsampled = resample(
                        df_minority, replace=True, n_samples=len(df_majority), random_state=42
                    )
                    X_balanced = pd.concat([df_majority, df_minority_upsampled])
                    y_balanced = pd.Series([0] * len(df_majority) + [1] * len(df_minority_upsampled))
                    
                    # Enhanced scaling to prevent overflow
                    print("Scaling features...")
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X_balanced)
                    X_scaled = np.clip(X_scaled, -10, 10)
                    
                    print("Splitting data into training and test sets...")
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_scaled, y_balanced, test_size=0.3, random_state=42, stratify=y_balanced
                    )
                    
                    # Scikit-learn model with adjusted parameters
                    model_sk = LogisticRegression(
                        max_iter=2000, class_weight="balanced", random_state=42,
                        penalty='l2', C=0.1, solver='lbfgs'
                    )
                    print("Fitting Scikit-learn logistic regression (to be used as a backup in case Statsmodels fails)...")
                    model_sk.fit(X_train, y_train)
                    
                    # Statsmodels with L2 regularization and increased iterations
                    X_train_sm = sm.add_constant(X_train.astype(float))
                    print("Fitting Statsmodels logistic regression...")
                    try:
                        model_sm = sm.Logit(y_train, X_train_sm).fit(
                            method='bfgs',        # Limited-memory BFGS, good for large datasets
                            maxiter=1000,          # Increase iterations for better convergence
                            pgtol=1e-8,            # Gradient tolerance for convergence
                            factr=1e6,             # Factor for function value tolerance
                            disp=0,                # Suppress convergence messages
                            cov_type='opg'         # Use OPG for covariance estimation
                        )
                        # Check convergence explicitly
                        if model_sm.mle_retvals.get('converged', False) and not model_sm.mle_retvals.get('warnflag', 1) == 0:
                            print("Statsmodels logistic regression converged successfully.")
                        else:
                            raise ValueError("Convergence questionable based on mle_retvals.")
                    except Exception as e:
                        print(f"Statsmodels failed or convergence unreliable: {str(e)}. Falling back to sklearn model.")
                        model_sm = None
                    
                    print("Predicting on test set...")
                    y_pred = model_sk.predict(X_test)
                    y_pred_proba = model_sk.predict_proba(X_test)[:, 1]
                    
                    print("\nClassification Report:")
                    print(classification_report(y_test, y_pred, digits=3))
                    
                    print("\nConfusion Matrix:")
                    print(confusion_matrix(y_test, y_pred))
                    
                    feature_names = ["const"] + X.columns.tolist() if model_sm else X.columns.tolist()
                    if model_sm:
                        coef_dict = dict(zip(feature_names, model_sm.params))
                        pval_dict = dict(zip(feature_names, model_sm.pvalues))
                        print("\nFeature Importance (Coefficients):")
                        sorted_coefs = sorted(coef_dict.items(), key=lambda x: abs(x[1]), reverse=True)
                        
                        numerical_coefs = [(feat, coef) for feat, coef in sorted_coefs if feat not in category_feature_map and feat != "const"]
                        categorical_coefs = [(feat, coef) for feat, coef in sorted_coefs if feat in category_feature_map]
                        
                        print("Top Numerical Features:")
                        for feat, coef in numerical_coefs[:15]:
                            p_val = pval_dict.get(feat, 1.0)
                            sig_note = " (p < 0.05, significant)" if p_val < 0.05 else " (p >= 0.05, not significant)"
                            print(f"  {feat}: {coef:.4f}{sig_note}")
                        
                        categorical_by_base = {}
                        for feat, coef in categorical_coefs:
                            base = category_feature_map[feat]
                            if base not in categorical_by_base:
                                categorical_by_base[base] = []
                            categorical_by_base[base].append((feat, coef, pval_dict.get(feat, 1.0)))
                        
                        for base, coef_list in categorical_by_base.items():
                            print(f"\nAll Categories for {base}:")
                            if base == "Race":
                                print("  Note: Racial patterns reflect data distribution, not targeted focus.")
                            sorted_cat_coefs = sorted(coef_list, key=lambda x: abs(x[1]), reverse=True)
                            for i, (feat, coef, p_val) in enumerate(sorted_cat_coefs):
                                prefix = "**" if i == 0 else "  "
                                suffix = "**" if i == 0 else ""
                                sig_note = " (p < 0.05, significant)" if p_val < 0.05 else " (p >= 0.05, not significant)"
                                print(f"    {prefix}{feat}: {coef:.4f}{sig_note}{suffix}")
                    else:
                        print("\nFeature Importance (sklearn coefficients):")
                        for feat, coef in sorted(zip(X.columns, model_sk.coef_[0]), key=lambda x: abs(x[1]), reverse=True)[:30]:
                            print(f"  {feat}: {coef:.4f}")
                    
                    roc_auc = roc_auc_score(y_test, y_pred_proba)
                    auprc = average_precision_score(y_test, y_pred_proba)
                    print(f"\nROC-AUC Score: {roc_auc:.3f}")
                    print(f"AUPRC Score: {auprc:.3f}")
                else:
                    print("No suspicious loans found for modeling.")
            else:
                print("Not enough classes for logistic regression.")
                                            
        except Exception as e:
            print(f"Error in multivariate analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
            
    def run_analysis(self) -> None:
        try:
            sus, full = self.load_data()
            print(
                f"\nStarting analysis of {len(sus):,} suspicious loans "
                f"out of {len(full):,} loans in the $5k-$22k range"
            )
            xgb_analyzer = XGBoostAnalyzer()

            analysis_steps = [
                # ("Geographic Clusters", self.analyze_geographic_clusters),
                # ("Lender Patterns", self.analyze_lender_patterns),
                # ("Business Patterns", self.analyze_business_patterns),
                # ("Name Patterns", self.analyze_name_patterns),
                # ("Business Name Patterns", self.analyze_business_name_patterns),
                # ("Demographic Patterns", self.analyze_demographic_patterns),
                # ("Risk Flags Analysis", lambda s, f: self.analyze_risk_flags(s)),
                # ("Loan Amount Distribution", self.analyze_loan_amount_distribution),
                # ("Jobs Reported Patterns", self.analyze_jobs_reported_patterns),
                # ("Risk Score Distribution", lambda s, f: self.analyze_risk_score_distribution(s)),
                # ("Risk Flags Count Distribution", lambda s, f: self.analyze_flag_count_distribution(s)),
                # ("Correlations", lambda s, f: self.analyze_correlations(f)),
                ("Multivariate Analysis", self.analyze_multivariate),
                ("Feature Discrimination (AUPRC)", self.analyze_feature_discrimination),
                ("XGBoost Analysis", lambda s, f: xgb_analyzer.analyze_with_xgboost(s, f, n_iter=15)),
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
        suspicious_file="suspicious_loans_sorted.csv",
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