import pandas as pd
import logging
from termcolor import colored
import os

def setup_logging() -> logging.Logger:
    """Set up logging with colored output"""
    logger = logging.getLogger('SuspiciousLoanSorter')
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
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

def sort_suspicious_loans(input_file: str = "suspicious_loans.csv", 
                         output_file: str = "suspicious_loans_sorted.csv") -> None:
    """
    Sort suspicious loans by risk score in descending order.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to save the sorted CSV file
    """
    logger = setup_logging()
    
    try:
        # Check if input file exists
        if not os.path.exists(input_file):
            logger.error(f"Input file '{input_file}' not found!")
            return
        
        logger.info(f"Reading suspicious loans from {input_file}")
        
        # Define dtypes for key columns to prevent mixed types warning
        dtypes = {
            'LoanNumber': str,
            'BorrowerName': str,
            'BorrowerAddress': str,
            'BorrowerCity': str,
            'BorrowerState': str,
            'BorrowerZip': str,
            'RiskScore': float,
            'InitialApprovalAmount': float,
            'JobsReported': float,
            'NAICSCode': str,
            'BusinessType': str,
            'Race': str,
            'Gender': str,
            'Ethnicity': str,
            'LoanStatus': str,
            'OriginatingLender': str,
            'OriginatingLenderLocationID': str,
            'RiskLevel': str,
            'RiskFlags': str
        }
        
        # Read CSV with specified dtypes and low_memory=False
        df = pd.read_csv(input_file, dtype=dtypes, low_memory=False)
        
        if 'RiskScore' not in df.columns:
            logger.error("Error: RiskScore column not found in the CSV file!")
            return
            
        initial_rows = len(df)
        logger.info(f"Found {initial_rows:,} suspicious loans to sort")
        
        # Sort by RiskScore in descending order
        df_sorted = df.sort_values(by='RiskScore', ascending=False)
        
        # Save sorted data
        df_sorted.to_csv(output_file, index=False)
        logger.info(f"Successfully sorted {initial_rows:,} loans by risk score")
        logger.info(f"Sorted results saved to: {output_file}")
        
        # Display top 500 highest risk scores
        logger.info("\nTop 500 highest risk scores:")
        top_500 = df_sorted.head(500)
        for _, loan in top_500.iterrows():
            logger.info(
                f"Risk Score: {loan['RiskScore']:.1f} | "
                f"Borrower: {loan['BorrowerName']} | "
                f"Amount: ${float(loan['InitialApprovalAmount']):,.2f}"
            )
            
    except pd.errors.EmptyDataError:
        logger.error(f"Error: The file '{input_file}' is empty!")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    sort_suspicious_loans()