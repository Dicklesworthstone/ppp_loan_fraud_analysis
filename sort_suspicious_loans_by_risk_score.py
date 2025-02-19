import pandas as pd
import logging
from termcolor import colored
import os
import numpy as np

def setup_logging() -> logging.Logger:
    """Set up logging with colored output"""
    logger = logging.getLogger('SuspiciousLoanSorter')
    logger.setLevel(logging.INFO)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    class ColoredFormatter(logging.Formatter):
        COLORS = {
            'WARNING': 'yellow',
            'INFO': 'white',
            'DEBUG': 'blue',
            'CRITICAL': 'red',
            'ERROR': 'red'
        }
        
        def format(self, record):
            if hasattr(record, 'msg'):
                levelname = record.levelname
                if levelname in self.COLORS:
                    record.msg = colored(record.msg, self.COLORS[levelname])
            return super().format(record)
    
    formatter = ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s', 
                               datefmt='%Y-%m-%d %H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def create_ascii_histogram(data: pd.Series, width: int = 50, num_bins: int = 10) -> str:
    """Create an ASCII histogram with percentile-based bins"""
    if data.empty or data.isna().all():
        return "No data to display in histogram"
    
    # Calculate percentile edges
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(data, percentiles)
    # Remove duplicate edges (can happen if data has many identical values)
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return f"All values are approximately {data.iloc[0]:.1f}"
    
    hist, edges = np.histogram(data, bins=bin_edges)
    max_count = hist.max()
    if max_count == 0:
        return "No data to display in histogram"
    
    result = ["Risk Score Distribution (Percentile-Based):"]
    for i in range(len(hist)):
        count = hist[i]
        bar_length = int((count / max_count) * width) if max_count > 0 else 0
        bar = '█' * bar_length
        if i == len(hist) - 1:
            range_str = f"{edges[i]:.0f}+"
        else:
            range_str = f"{edges[i]:.0f}-{edges[i+1]:.0f}"
        percentile_range = f"{percentiles[i]:.1f}-{percentiles[i+1]:.1f}%"
        result.append(f"{range_str:>12} ({percentile_range}): {bar} ({count:,})")
    
    # Add basic statistics
    result.append("\nHistogram Statistics:")
    result.append(f"  Mean: {data.mean():.1f}")
    result.append(f"  Median: {data.median():.1f}")
    result.append(f"  Min: {data.min():.1f}")
    result.append(f"  Max: {data.max():.1f}")
    
    return "\n".join(result)

def sort_suspicious_loans(input_file: str = "suspicious_loans.csv", 
                         output_file: str = "suspicious_loans_sorted.csv",
                         min_risk_cutoff: float = 0.0) -> None:
    """
    Sort suspicious loans by risk score in descending order with minimum risk cutoff.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to save the sorted CSV file
        min_risk_cutoff (float): Minimum risk score threshold (default: 0.0)
    """
    logger = setup_logging()
    
    try:
        if not os.path.exists(input_file):
            logger.error(f"Input file '{input_file}' not found!")
            return
        
        logger.info(f"Reading suspicious loans from {input_file}")
        
        # Define dtypes based on the PPP loan processor
        dtypes = {
            'Accuracy Score': float, 'Accuracy Type': str, 'BorrowerAddress': str, 'BorrowerCity': str,
            'BorrowerName': str, 'BorrowerState': str, 'BorrowerZip': str, 'BusinessAgeDescription': str,
            'BusinessType': str, 'CD': str, 'Census Block Code': str, 'Census Block Group': str,
            'Census Tract Code': str, 'Census Year': str, 'City': str, 'Combined Statistical Area Code': str,
            'Combined Statistical Area Name': str, 'Country': str, 'County': str, 'County FIPS': str,
            'CurrentApprovalAmount': float, 'DateApproved': str, 'DEBT_INTEREST_PROCEED': float,
            'Ethnicity': str, 'ForgivenessAmount': float, 'ForgivenessDate': str, 'FranchiseName': str,
            'Full FIPS (block)': str, 'Full FIPS (tract)': str, 'Gender': str, 'HEALTH_CARE_PROCEED': float,
            'HubzoneIndicator': str, 'InitialApprovalAmount': float, 'JobsReported': float, 'Latitude': float,
            'LMIIndicator': str, 'LoanNumber': str, 'LoanStatus': str, 'LoanStatusDate': str, 'Longitude': float,
            'Metro/Micro Statistical Area Code': str, 'Metro/Micro Statistical Area Name': str,
            'Metro/Micro Statistical Area Type': str, 'Metropolitan Division Area Code': str,
            'Metropolitan Division Area Name': str, 'MORTGAGE_INTEREST_PROCEED': float, 'NAICSCode': str,
            'NonProfit': str, 'Number': str, 'OriginatingLender': str, 'OriginatingLenderCity': str,
            'OriginatingLenderLocationID': str, 'OriginatingLenderState': str, 'PAYROLL_PROCEED': float,
            'Place FIPS': str, 'Place Name': str, 'ProcessingMethod': str, 'ProjectCity': str,
            'ProjectCountyName': str, 'ProjectState': str, 'ProjectZip': str, 'Race': str,
            'REFINANCE_EIDL_PROCEED': float, 'RENT_PROCEED': float, 'RuralUrbanIndicator': str,
            'SBAGuarantyPercentage': float, 'SBAOfficeCode': str, 'ServicingLenderAddress': str,
            'ServicingLenderCity': str, 'ServicingLenderLocationID': str, 'ServicingLenderName': str,
            'ServicingLenderState': str, 'ServicingLenderZip': str, 'Source': str, 'State': str,
            'State FIPS': str, 'Street': str, 'Term': float, 'UndisbursedAmount': float, 'Unit Number': str,
            'Unit Type': str, 'UTILITIES_PROCEED': float, 'Veteran': str, 'Zip': str,
            'RiskScore': float, 'RiskFlags': str
        }
        
        df = pd.read_csv(input_file, dtype=dtypes, low_memory=False)
        
        if 'RiskScore' not in df.columns:
            logger.error("Error: RiskScore column not found in the CSV file!")
            return
            
        initial_rows = len(df)
        logger.info(f"Found {initial_rows:,} suspicious loans to process")
        
        # Apply minimum risk cutoff and calculate statistics
        logger.info(f"Filtering loans with RiskScore >= {min_risk_cutoff}")
        df_filtered = df[df['RiskScore'] >= min_risk_cutoff]
        filtered_rows = len(df_filtered)
        filtered_out = initial_rows - filtered_rows
        filtered_percent = (filtered_out / initial_rows * 100) if initial_rows > 0 else 0
        
        if filtered_rows == 0:
            logger.warning("No loans remain after applying the risk cutoff!")
            return
            
        logger.info("Filtering Statistics:")
        logger.info(f"  Total loans before filtering: {initial_rows:,}")
        logger.info(f"  Loans filtered out: {filtered_out:,} ({filtered_percent:.1f}%)")
        logger.info(f"  Loans remaining: {filtered_rows:,} ({100-filtered_percent:.1f}%)")
        
        # Create and display histogram of remaining loans
        logger.info("\n" + create_ascii_histogram(df_filtered['RiskScore']))
        
        # Sort by RiskScore in descending order
        df_sorted = df_filtered.sort_values(by='RiskScore', ascending=False)
        
        # Save sorted data
        df_sorted.to_csv(output_file, index=False)
        logger.info(f"\nSuccessfully sorted {filtered_rows:,} loans by risk score")
        logger.info(f"Sorted results saved to: {output_file}")
        
        # Display top 100 highest risk scores (or all if less than 100)
        display_count = min(100, filtered_rows)
        logger.info(f"\nTop {display_count} highest risk scores:")
        top_n = df_sorted.head(display_count)
        for _, loan in top_n.iterrows():
            logger.info(
                f"Risk Score: {loan['RiskScore']:.1f} | "
                f"Borrower: {loan['BorrowerName']} | "
                f"Amount: ${float(loan['InitialApprovalAmount']):,.2f} | "
                f"Location: {loan['BorrowerCity']}, {loan['BorrowerState']} | "
                f"Jobs: {loan['JobsReported']} | "
                f"Lender: {loan['OriginatingLender']} | "
                f"Flags: {loan['RiskFlags']}"
            )
            
    except pd.errors.EmptyDataError:
        logger.error(f"Error: The file '{input_file}' is empty!")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    sort_suspicious_loans(min_risk_cutoff=140.0)