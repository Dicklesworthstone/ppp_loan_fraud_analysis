import os
import re
import zipfile
import asyncio
import aiofiles
import httpx
import logging
from math import log
import pandas as pd
import numpy as np
from tqdm import tqdm
from termcolor import colored
from typing import List, Set, Tuple
from collections import defaultdict

CSV_FILENAME = "ppp-full.csv"
ZIP_URL = "https://releases.geocod.io/public/PPP_full_geocodio.zip"
ZIP_FILENAME = "PPP_full_geocodio.zip"

class ColoredFormatter(logging.Formatter):
    COLORS = {'WARNING': 'yellow', 'INFO': 'white', 'DEBUG': 'blue', 'CRITICAL': 'red', 'ERROR': 'red'}
    def format(self, record):
        if hasattr(record, 'msg'):
            if not isinstance(record.msg, str) or '\033[' not in str(record.msg):
                levelname = record.levelname
                if levelname in self.COLORS:
                    record.msg = colored(record.msg, self.COLORS[levelname])
        return super().format(record)

def setup_logging() -> logging.Logger:
    logger = logging.getLogger('PPPFraudDetector')
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

if 'profile' not in globals():
    def profile(func):
        """A no-op decorator for when line_profiler is not active."""
        return func
    
async def download_and_extract_csv() -> None:
    logger = setup_logging()
    if not os.path.exists(CSV_FILENAME):
        logger.info(f"CSV file '{CSV_FILENAME}' not found. Downloading from {ZIP_URL}...")
        async with httpx.AsyncClient() as client:
            resp = await client.get(ZIP_URL)
            resp.raise_for_status()
        async with aiofiles.open(ZIP_FILENAME, "wb") as f:
            logger.debug(f"Writing ZIP archive to {ZIP_FILENAME}...")
            await f.write(resp.content)
        logger.info("Download complete. Extracting the CSV file...")
        with zipfile.ZipFile(ZIP_FILENAME, "r") as z:
            csv_names = [n for n in z.namelist() if n.lower().endswith("ppp-full.csv")]
            if not csv_names:
                logger.error("ERROR: Could not find ppp-full.csv in the downloaded ZIP!")
                raise SystemExit(1)
            z.extract(csv_names[0], ".")
            if os.path.basename(csv_names[0]) != csv_names[0]:
                os.replace(csv_names[0], CSV_FILENAME)
        logger.info(f"Extraction complete. {CSV_FILENAME} is ready.")
    else:
        logger.info(f"Found {CSV_FILENAME} in the project root. No download needed.")

class PPPLoanProcessor:
    def __init__(self, input_file: str, output_file: str, risk_threshold: float = 50, chunk_size: int = 10000):
        self.input_file = input_file
        self.output_file = output_file
        self.risk_threshold = risk_threshold
        self.chunk_size = chunk_size
        self.logger = setup_logging()
        self.seen_loans = set()
        self.borrower_loans = {}
        self.seen_addresses = {}
        self.daily_zip_counts = defaultdict(lambda: defaultdict(int))
        self.daily_lender_counts = defaultdict(lambda: defaultdict(int))
        self.total_days = set()        
        self.stats = {
            'processed': 0,
            'suspicious': 0,
            'total_risk_score': 0,
            'high_risk_lenders': set(),
            'suspicious_patterns': {},
            'duplicate_loans': 0
        }
        self.RISK_WEIGHTS = {
            'business_age': 0.10,
            'address': 0.15,
            'loan_amount': 0.10,
            'jobs': 0.20,
            'forgiveness': 0.05,
            'business_type': 0.10,
            'naics': 0.10,
            'lender': 0.10,
            'demographics': 0.05,
            'name_pattern': 0.05,
            'geospatial_cluster': 0.05,
            'sba_office_cluster': 0.05 
        }
        self.HIGH_RISK_LENDERS = {
            'Celtic Bank Corporation': 0.8,
            'Cross River Bank': 0.7,
            'Customers Bank': 0.7,
            'Kabbage, Inc.': 0.8,
            'Capital Plus Financial': 0.9,
            'Harvest Small Business Finance': 0.8
        }
        self.LEGITIMATE_BUSINESS_TYPES = {
            'Corporation',
            'Limited Liability Company(LLC)',
            'Subchapter S Corporation',
            'Partnership',
            'Non-Profit Organization'
        }
        self.HIGH_RISK_BUSINESS_TYPES = {
            'Sole Proprietorship': 0.85,
            'Self-Employed Individuals': 0.7,
            'Independent Contractors': 0.5
        }
        self.GENERIC_NAICS = {
            '541990': 0.7,
            '541618': 0.7,
            '541611': 0.6,
            '453998': 0.8,
            '454390': 0.8,
            '541213': 0.8,
            '812111': 0.8,
            '812113': 0.8,
        }
        self.SUSPICIOUS_PATTERNS = {
            r'wakanda': 0.95,
            r'thanos': 0.95,
            r'black\s*panther': 0.95,
            r'vibranium': 0.95,
            r'asgard': 0.95,
            r'iron\s*man': 0.95,
            r'groot': 0.95,
            r'avatar': 0.95,
            r'matrix': 0.95,
            r'gotham': 0.95,
            r'krypton': 0.95,
            r'multiverse': 0.95,
            r'hellcat': 0.9,
            r'demon\s*hemi': 0.9,
            r'red\s*bottom': 0.9,
            r'gucci': 0.85,
            r'bentley': 0.85,
            r'lambo': 0.9,
            r'rolex': 0.85,
            r'versace': 0.85,
            r'louis\s*v': 0.85,
            r'fendi': 0.85,
            r'balenciaga': 0.85,
            r'yacht': 0.85,
            r'maybach': 0.9,
            r'rolls\s*royce': 0.9,
            r'private\s*jet': 0.9,
            r'covid': 0.9,
            r'crab\s*leg': 0.8,
            r'seafood\s*king': 0.8,
            r'wing\s*king': 0.8,
            r'kitchen\s*king': 0.8,
            r'food\s*empire': 0.8,
            r'gourmet\s*empire': 0.8,
            r'cuisine\s*empire': 0.8,
            r'free\s*money': 0.95,
            r'get\s*paid': 0.95,
            r'get\s*money': 0.95,
            r'secure\s*bag': 0.95,
            r'bag\s*secured': 0.95,
            r'quick\s*cash': 0.9,
            r'easy\s*money': 0.9,
            r'fast\s*cash': 0.9,
            r'money\s*printer': 0.95,
            r'cash\s*king': 0.9,
            r'money\s*team': 0.9,
            r'reparation': 0.95,
            r'wealth\s*plug': 0.95,
            r'money\s*plug': 0.95,
            r'connect\s*plug': 0.95,
            r'luxury\s*lifestyle': 0.85,
            r'rich\s*life': 0.85,
            r'wealth\s*empire': 0.85,
            r'money\s*empire': 0.9,
            r'success\s*empire': 0.85,
            r'wealth\s*builder': 0.8,
            r'money\s*maker': 0.85,
            r'boss\s*life': 0.85,
            r'ceo\s*mindset': 0.85,
            r'billionaire\s*mindset': 0.9,
            r'millionaire\s*lifestyle': 0.9,
            r'entrepreneur\s*empire': 0.8,
            r'success\s*blueprint': 0.85,
            r'wealth\s*secrets': 0.9,
            r'money\s*blueprint': 0.9,
            r'ppp\s*king': 0.95,
            r'loan\s*king': 0.95,
            r'grant\s*master': 0.95,
            r'stimulus': 0.9,
            r'pandemic\s*profit': 0.95,
            r'covid\s*cash': 0.95,
            r'forgiven': 0.9,
            r'blessed\s*by\s*gov': 0.95,
            r'uncle\s*sam\s*bless': 0.95,
            r'government\s*cheese': 0.95,
            r'bread\s*winner': 0.85,
            r'get\s*this\s*bread': 0.9,
            r'paper\s*chase': 0.9,
            r'hustl[ea]': 0.85,
            r'grind': 0.8,
            r'trap': 0.85,
            r'plug': 0.9,
            r'connect': 0.8,
            r'baller': 0.9,
            r'high\s*roller': 0.9,
            r'cash\s*flow': 0.3,
            r'investment\s*group': 0.4,
            r'consulting\s*llc': 0.3,
            r'holdings\s*corp': 0.3,
            r'ventures': 0.3,
            r'capital': 0.3,
            r'enterprises': 0.3,
            r'management': 0.2,
            r'solutions': 0.2,
            r'lucky\s*charm': 0.85,
            r'blessed': 0.7,
            r'deez\snutz': 0.95,
            r'\bno\s*money\s*no\s*honey\b': 0.9,
            r"\bfool's\s*gold\b": 0.95,
            r'\bquick\s*buck\b': 0.9,
            r'\bget\s*rich\s*quick\b': 0.9,
            r'\binstant\s*wealth\b': 0.95,
            r'\bno\s*credit\s*needed\b': 0.85,
            r'\bcon\s*game\b': 0.9,
            r'\bchump\s*change\b': 0.85,
            r'relief': 0.9,
            r'grant': 0.9,
            r'stimulus\s*helper': 0.95,
            r'ppp\s*assist': 0.95,
            r'loan\s*help': 0.95,
            r'pandemic': 0.9,
            r'covid\s*relief': 0.95,
            r'emergency\s*fund': 0.9,
            r'disaster\s*loan': 0.9,
            r'drop\s*ship': 0.9,
            r'resell': 0.9,
            r'flip': 0.9,
            r'broker': 0.8,
            r'middle\s*man': 0.9,
            r'referral': 0.85,
            r'affiliate': 0.85,
            r'marketing\s*guru': 0.9,
            r'business\s*coach': 0.9,
            r'life\s*coach': 0.9,
            r'mentor': 0.85,
            r'mastermind': 0.9,
            r'stonks': 0.95,
            r'diamond\s*hands': 0.95,
            r'yolo': 0.95,
            r'lmao': 0.95,
            r'finesse': 0.95,
            r'sauce': 0.9,
            r'vibes': 0.9,
            r'lit': 0.9,
            r'fire': 0.9,
            r'wave': 0.9,
            r'mood': 0.9,
            r'energy': 0.9,
            r'no\s*cap': 0.95,
            r'fr\s*fr': 0.95,
            r'gang': 0.95,
            r'squad': 0.9,
            r'drip': 0.9,
            r'designer': 0.85,
            r'luxury': 0.85,
            r'exclusive': 0.85,
            r'elite': 0.85,
            r'premium': 0.8,
            r'lifestyle': 0.85,
            r'luxury\s*car': 0.9,
            r'exotic\s*car': 0.9,
            r'foreign\s*car': 0.9,
            r'stimulus\s*check': 0.95,
            r'money\s*moves': 0.95,
            r'cash\s*app': 0.95,
            r'venmo': 0.95,
            r'zelle': 0.95,
            r'paypal': 0.9,
            r'swipe': 0.9,
            r'invest\s*guru': 0.9,
            r'money\s*magnet': 0.95,
            r'passive\s*income': 0.9,
            r'income\s*stream': 0.9,
            r'money\s*rain': 0.95,
            r'get\s*the\s*bag': 0.95,
            r'money\s*bag': 0.95,
            r'money\s*machine': 0.95,
            r'onlyfans': 0.95,
            r'clubhouse': 0.9,
            r'tiktok': 0.9,
            r'snapchat': 0.9,
            r'instagram': 0.9,
        }
        # Precompile suspicious patterns for faster reuse.
        pattern_list = [f'(?P<pat{i}>{pat})' for i, pat in enumerate(self.SUSPICIOUS_PATTERNS.keys())]
        pattern_str = '|'.join(pattern_list)
        self.compiled_suspicious_pattern = re.compile(pattern_str, re.IGNORECASE)
        # Map group names back to original patterns and weights
        self.pattern_weights = {f'pat{i}': weight for i, (pat, weight) in enumerate(self.SUSPICIOUS_PATTERNS.items())}
        self.pattern_to_original = {f'pat{i}': pat for i, pat in enumerate(self.SUSPICIOUS_PATTERNS.keys())}        
        self.LEGITIMATE_KEYWORDS = {
            'consulting', 'services', 'solutions', 'associates',
            'partners', 'group', 'inc', 'llc', 'ltd', 'corporation',
            'management', 'enterprises', 'international', 'systems'
        }
        self.known_businesses = set()
        self.date_patterns = defaultdict(lambda: defaultdict(list))
        self.lender_loan_sequences = defaultdict(list)
        self.ZIP_CLUSTER_THRESHOLD = 5
        self.SEQUENCE_THRESHOLD = 5
        self.address_to_businesses = defaultdict(set)
        self.business_to_addresses = defaultdict(set)
        self.lender_batches = defaultdict(list)
        self.lender_sequences = defaultdict(list)
        self.RESIDENTIAL_INDICATORS = {
            'apt': 0.8, 'unit': 0.7, '#': 0.7, 'suite': 0.4, 'floor': 0.3,
            'po box': 0.9, 'p.o.': 0.9, 'box': 0.8, 'residence': 0.9,
            'residential': 0.9, 'apartment': 0.9, 'house': 0.8, 'condo': 0.8,
            'room': 0.9
        }
        self.COMMERCIAL_INDICATORS = {
            'plaza': -0.7, 'building': -0.5, 'tower': -0.6, 'office': -0.7,
            'complex': -0.5, 'center': -0.5, 'mall': -0.8, 'commercial': -0.8,
            'industrial': -0.8, 'park': -0.4, 'warehouse': -0.8, 'factory': -0.8,
            'store': -0.7, 'shop': -0.6
        }
        self.business_name_patterns = defaultdict(int)
        self.SUSPICIOUS_NAME_PATTERNS = {
            r'consulting.*llc': 0.1,
            r'holdings.*llc': 0.1,
            r'enterprise.*llc': 0.1,
            r'solutions.*llc': 0.1,
            r'services.*llc': 0.05,
            r'investment.*llc': 0.05,
        }
        # Precompiled patterns for validate_address
        self.address_fake_patterns = [
            re.compile(r'\d{1,3}\s*[a-z]+\s*(st|street|ave|avenue|rd|road)', re.IGNORECASE),
            re.compile(r'p\.?o\.?\s*box\s*\d+', re.IGNORECASE),
            re.compile(r'general\s*delivery', re.IGNORECASE)
        ]
        # Precompiled patterns for validate_business_name
        self.business_name_personal = re.compile(r'^[a-z]+\s+[a-z]+$', re.IGNORECASE)
        self.business_name_suspicious_chars = re.compile(r'[@#$%^&*]')
        # Precompiled patterns for analyze_address_type
        self.address_range = re.compile(r'\d+\s*-\s*\d+')
        self.address_street_end = re.compile(r'(st|street|ave|avenue|road|rd)\s*$', re.IGNORECASE)        
        # Precompile suspicious name patterns.
        self.compiled_suspicious_name_patterns = {re.compile(pat, re.IGNORECASE): weight for pat, weight in self.SUSPICIOUS_NAME_PATTERNS.items()}
        # Track geospatial coordinates
        self.seen_coordinates = defaultdict(set)  # (lat, lon) -> set of loan numbers
        # New: Track SBA office code volumes by date
        self.daily_office_counts = defaultdict(lambda: defaultdict(int))  # date -> office_code ->
        self.UNANSWERED_SET = {'unanswered', 'unknown'}
        self.INVALID_ADDRESS_SET = {'n/a', 'none', '', 'nan'}

    def validate_address(self, address: str) -> tuple[bool, List[str]]:
        flags = []
        address_str = str(address).lower().strip()
        if pd.isna(address) or address_str in self.INVALID_ADDRESS_SET:
            return False, ['Invalid/missing address']
        
        # Combine indicators into one regex
        residential_pattern = re.compile('|'.join(self.RESIDENTIAL_INDICATORS.keys()), re.IGNORECASE)
        commercial_pattern = re.compile('|'.join(self.COMMERCIAL_INDICATORS.keys()), re.IGNORECASE)
        
        residential_score = sum(weight for ind, weight in self.RESIDENTIAL_INDICATORS.items() if ind in address_str)
        commercial_score = sum(weight for ind, weight in self.COMMERCIAL_INDICATORS.items() if ind in address_str)
        total_score = residential_score + commercial_score
        
        if total_score > 0.7:
            flags.append('Residential address')
        if len(address_str) < 10:
            flags.append('Suspiciously short address')
        if any(pat.search(address_str) for pat in self.address_fake_patterns):
            flags.append('Potentially fake address pattern')
        return (len(flags) == 0), flags

    def validate_business_name(self, name: str) -> tuple[float, List[str]]:
        flags = []
        risk_score = 0
        name_str = str(name).lower().strip()
        if name_str in self.known_businesses:
            return 0, []
        if pd.isna(name) or name_str in ('n/a', 'none', ''):
            return 1.0, ['Invalid/missing business name']
        for cre, weight in self.compiled_suspicious_patterns.items():
            if cre.search(name_str):
                risk_score += weight
                flags.append(f'Suspicious name pattern: {cre.pattern}') 
        legitimate_count = sum(1 for keyword in self.LEGITIMATE_KEYWORDS if keyword in name_str)
        risk_score -= (legitimate_count * 0.2)
        if self.business_name_personal.match(name_str):
            risk_score += 0.4
            flags.append('Personal name only')
        if self.business_name_suspicious_chars.search(name_str):
            risk_score += 0.5
            flags.append('Suspicious characters in name')
        return max(0, min(1, risk_score)), flags

    @profile
    def check_multiple_applications(self, chunk: pd.DataFrame) -> tuple[pd.Series, List[List[str]]]:
        chunk = chunk.copy()
        # Precompute keys as lists to avoid pandas operations later
        exact_keys = [f"{row['BorrowerName']}_{row['BorrowerCity']}_{row['BorrowerState']}_{row['InitialApprovalAmount']}" 
                    for _, row in chunk.iterrows()]
        name_keys = [f"{row['BorrowerName']}_{row['BorrowerCity']}_{row['BorrowerState']}" 
                    for _, row in chunk.iterrows()]
        address_keys = [f"{row['BorrowerAddress']}_{row['BorrowerCity']}_{row['BorrowerState']}" 
                        for _, row in chunk.iterrows()]

        # Update global state
        self.seen_loans.update(exact_keys)
        name_counts_dict = {key: self.borrower_loans.get(key, 0) + 1 for key in name_keys}
        self.borrower_loans.update({key: self.borrower_loans.get(key, 0) + sum(1 for k in name_keys if k == key) 
                                    for key in set(name_keys)})
        address_counts_dict = {key: self.seen_addresses.get(key, 0) + 1 for key in address_keys}
        self.seen_addresses.update({key: self.seen_addresses.get(key, 0) + sum(1 for k in address_keys if k == key) 
                                    for key in set(address_keys)})

        # Initialize outputs
        scores = pd.Series(0.0, index=chunk.index)  # Keep scores as Series for risk scoring
        flags = [[] for _ in range(len(chunk))]  # Plain list of lists for flags

        # Process each row manually
        for i, (exact_key, name_key, address_key) in enumerate(zip(exact_keys, name_keys, address_keys)):
            # Exact duplicates
            if exact_key in self.seen_loans and exact_key in exact_keys[:i]:  # Check if seen before this row
                scores.iloc[i] += 30
                flags[i].append('Duplicate exact loan detected')
            
            # Multiple loans
            name_count = name_counts_dict[name_key]
            if name_count > 1:
                scores.iloc[i] += 30
                flags[i].append(f'Multiple loans ({name_count}) for same borrower')
            
            # Multi address
            address_count = address_counts_dict[address_key]
            if address_count > 2:
                scores.iloc[i] += 30
                flags[i].append(f'Address used {address_count} times')

        self.stats['duplicate_loans'] += sum(1 for key in exact_keys if exact_keys.count(key) > 1)
        return scores, flags

    @profile
    def analyze_time_patterns(self, loan: pd.Series) -> tuple[float, List[str]]:
        risk_score = 0
        flags = []
        date = str(loan['DateApproved'])  # Date only, e.g., "2020-04-15"
        zip_code = str(loan['BorrowerZip'])[:5]
        lender = str(loan['OriginatingLender'])
        loan_number = str(loan['LoanNumber'])
        
        # Store loan info with date (no timestamp needed)
        loan_info = {
            'loan_number': loan_number,
            'business_type': loan['BusinessType'],
            'amount': loan['InitialApprovalAmount'],
            'lender': lender
        }
        self.date_patterns[date][zip_code].append(loan_info)
        
        # Existing ZIP cluster analysis
        zip_cluster = self.date_patterns[date][zip_code]
        if len(zip_cluster) >= self.ZIP_CLUSTER_THRESHOLD:
            amounts = [l['amount'] for l in zip_cluster]
            business_types = [l['business_type'] for l in zip_cluster]
            if max(amounts) - min(amounts) < min(amounts) * 0.1:
                risk_score += 10
                flags.append(f'Part of cluster: {len(zip_cluster)} similar loans in ZIP {zip_code} on {date}')
            if len(set(business_types)) == 1:
                risk_score += 15
                flags.append(f'Cluster of identical business types in ZIP {zip_code}')
        
        # Track total loans per day by ZIP and lender
        self.daily_zip_counts[date][zip_code] += 1
        self.daily_lender_counts[date][lender] += 1
        
        # Calculate cluster intensity
        self.total_days.add(date)
        days_processed = len(self.total_days)
        
        if days_processed > 1:
            # ZIP-based intensity
            total_zip_loans = sum(self.daily_zip_counts[date].values())
            avg_daily_zip_loans = total_zip_loans / days_processed
            zip_loans_today = self.daily_zip_counts[date][zip_code]
            MIN_LOANS_FOR_CLUSTER = 5
            if zip_loans_today >= MIN_LOANS_FOR_CLUSTER and avg_daily_zip_loans >= 1:
                zip_intensity = zip_loans_today / avg_daily_zip_loans
                if zip_intensity > 5:
                    base_score = 20
                    cluster_score = base_score + 10 * log(max(zip_loans_today - MIN_LOANS_FOR_CLUSTER + 1, 1), 2)
                    risk_score += cluster_score
                    flags.append(f'Unusual ZIP cluster intensity: {zip_loans_today:,} loans vs {avg_daily_zip_loans:.1f} avg on {date} (score: {cluster_score:.1f})')
                        
            # Lender-based intensity
            total_lender_loans = sum(self.daily_lender_counts[date].values())
            avg_daily_lender_loans = total_lender_loans / days_processed
            lender_loans_today = self.daily_lender_counts[date][lender]
            MIN_LOANS_FOR_CLUSTER = 5
            if lender_loans_today >= MIN_LOANS_FOR_CLUSTER and avg_daily_lender_loans >= 1:
                lender_intensity = lender_loans_today / avg_daily_lender_loans
                if lender_intensity > 5:
                    intensity_score = min(20 * (lender_intensity / 5), 40)
                    risk_score += intensity_score
                    flags.append(f'Unusual lender cluster intensity: {lender_loans_today:,} loans vs {avg_daily_lender_loans:.1f} avg on {date}')

        # Existing sequential check
        self.lender_loan_sequences[lender].append(loan_number)
        recent_loans = self.lender_loan_sequences[lender][-self.SEQUENCE_THRESHOLD:]
        if len(recent_loans) >= self.SEQUENCE_THRESHOLD and self.is_roughly_sequential(recent_loans):
            risk_score += 25
            flags.append(f'Sequential loan numbers from {lender}')
        
        return risk_score, flags

    def is_roughly_sequential(self, loan_numbers: List[str]) -> bool:
        # Minimize regex overhead by pre-filtering and extracting in one pass
        numbers = np.array([int(m[-1]) for num in loan_numbers if (m := re.findall(r'\d+', num))], dtype=np.int64)
        if len(numbers) < 2:
            return False
        np.sort(numbers, kind='quicksort')  # In-place sort is slightly faster
        gaps = np.diff(numbers)
        result = (gaps.mean() < 10) & np.all(gaps < 20)  # Use bitwise & for consistency
        return result

    def analyze_address_type(self, address: str) -> float:
        address_str = str(address).lower()
        score = sum(weight for ind, weight in self.RESIDENTIAL_INDICATORS.items() if ind in address_str)
        score += sum(weight for ind, weight in self.COMMERCIAL_INDICATORS.items() if ind in address_str)
        if self.address_range.search(address_str):
            score += 0.6
        if self.address_street_end.search(address_str):
            score += 0.4
        return score

    def analyze_name_patterns(self, business_names: Set[str]) -> Tuple[float, List[str]]:
        self.logger.debug("Analyzing name patterns for businesses")
        score = 0
        flags = []
        name_patterns = defaultdict(int)
        names = [str(name).lower() for name in business_names]
        for name in names:
            for cre, weight in self.compiled_suspicious_name_patterns.items():
                if cre.search(name):
                    name_patterns[cre.pattern] += 1
        for pattern, count in name_patterns.items():
            if count >= 2:
                pattern_score = self.SUSPICIOUS_NAME_PATTERNS[pattern] * count
                score += pattern_score
                flags.append(f"Found {count} businesses matching pattern: {pattern}")
        lengths = [len(name) for name in names]
        if len(lengths) >= 3:
            if max(lengths) - min(lengths) <= 2:
                score += 10
                flags.append("Multiple businesses with suspiciously similar name lengths")
        self.logger.debug("Name pattern analysis complete")
        return score, flags

    def check_sequential_pattern(self, loans: List[dict]) -> Tuple[float, List[str]]:
        self.logger.debug("Checking sequential patterns in lender loans")
        if len(loans) < 2:
            return 0, []
        score = 0
        flags = []
        number_pairs = []
        business_names = set()
        for loan in loans:
            matches = re.findall(r'\d+', loan['loan_number'])
            if matches:
                number_pairs.append((int(matches[-1]), loan['business_name'], pd.to_datetime(loan['date'])))
                business_names.add(loan['business_name'])
        if len(number_pairs) < 2 or len(business_names) < 2:
            return 0, []
        number_pairs.sort()
        suspicious_sequences = 0
        for i in range(len(number_pairs) - 1):
            number_gap = number_pairs[i+1][0] - number_pairs[i][0]
            time_gap = (number_pairs[i+1][2] - number_pairs[i][2]).total_seconds()
            if number_gap <= 10:
                if time_gap < 3600:
                    suspicious_sequences += 1
                    score += 10
                elif time_gap < 86400:
                    suspicious_sequences += 1
                    score += 5
        if suspicious_sequences > 0:
            flags.append(f"Found {suspicious_sequences} suspicious sequential loan number patterns")
            pattern_score, pattern_flags = self.analyze_name_patterns(business_names)
            score += pattern_score
            flags.extend(pattern_flags)
        self.logger.debug("Sequential pattern check complete")
        return score, flags

    @profile
    def analyze_networks(self, loan: pd.Series) -> tuple[float, List[str]]:
        self.logger.debug("Analyzing networks for loan")
        risk_score = 0
        flags = []  # Start with a clean list
        
        # Address-based network analysis
        if pd.notna(loan['BorrowerAddress']):
            address_key = f"{loan['BorrowerAddress']}_{loan['BorrowerCity']}_{loan['BorrowerState']}"
            business_name = loan['BorrowerName']
            self.address_to_businesses[address_key].add(business_name)
            self.business_to_addresses[business_name].add(address_key)
            residential_score = self.analyze_address_type(loan['BorrowerAddress'])
            businesses_at_address = len(self.address_to_businesses[address_key])
            
            if businesses_at_address >= 2:
                if residential_score > 0.5:
                    risk_score += 15 * businesses_at_address
                    flags.append(f"Multiple businesses at residential address ({businesses_at_address})")
                else:
                    risk_score += 8 * businesses_at_address
                flags.append(f"Shared address with {businesses_at_address-1} other businesses")
                
                connected_businesses = set()
                for addr in self.business_to_addresses[business_name]:
                    connected_businesses.update(self.address_to_businesses[addr])
                if len(connected_businesses) > 2:
                    pattern_score, pattern_flags = self.analyze_name_patterns(connected_businesses)
                    risk_score += pattern_score
                    flags.extend(pattern_flags)
                    risk_score += 5 * len(connected_businesses)
                    flags.append(f"Connected to {len(connected_businesses)-1} other businesses")
        
        # Lender batch analysis
        lender_key = f"{loan['OriginatingLender']}_{loan['OriginatingLenderLocationID']}_{loan['DateApproved']}"
        batch_info = {
            'loan_number': loan['LoanNumber'],
            'amount': loan['InitialApprovalAmount'],
            'business_name': loan['BorrowerName'],
            'timestamp': loan['DateApproved']
        }
        self.lender_batches[lender_key].append(batch_info)
        current_batch = self.lender_batches[lender_key]
        
        if len(current_batch) >= 5:
            amounts = [l['amount'] for l in current_batch]
            if max(amounts) - min(amounts) < min(amounts) * 0.1:
                batch_names = {l['business_name'] for l in current_batch}
                pattern_score, pattern_flags = self.analyze_name_patterns(batch_names)
                risk_score += 15 + pattern_score
                flags.append(f"Part of suspicious batch: {len(current_batch)} similar loans from same lender")
                flags.extend(pattern_flags)
        
        # Lender sequence analysis
        lender_base = f"{loan['OriginatingLender']}_{loan['OriginatingLenderLocationID']}"
        self.lender_sequences[lender_base] = self.lender_sequences[lender_base][-50:] + [{
            'loan_number': loan['LoanNumber'],
            'business_name': loan['BorrowerName'],
            'date': loan['DateApproved']
        }]
        recent_loans = self.lender_sequences[lender_base][-5:]
        sequence_score, sequence_flags = self.check_sequential_pattern(recent_loans)
        risk_score += sequence_score
        flags.extend(sequence_flags)  # Extend without adding semicolons here
        
        self.logger.debug("Network analysis complete")
        return risk_score, flags
    
    @profile
    def calculate_risk_scores(self, chunk: pd.DataFrame) -> pd.DataFrame:
        self.logger.debug("Calculating risk scores for chunk")
        
        INTERACTION_RULES = {
            ("High amount per employee", "Residential address"): 1.05,
            ("Exact maximum loan amount detected", "High-risk lender"): 1.05,
            ("Multiple businesses at residential address", "Cluster of identical business types"): 1.15,
            ("Missing all demographics", "High-risk business type"): 1.05,
            ("Geospatial cluster", "Sequential loan numbers"): 1.05,
            ("SBA office cluster", "Sequential loan numbers"): 1.05  
        }

        chunk = chunk.copy()
        chunk['BorrowerName_lower'] = chunk['BorrowerName'].str.lower().fillna('')
        chunk['BorrowerAddress_lower'] = chunk['BorrowerAddress'].str.lower().fillna('')
        chunk['BusinessAgeDescription_lower'] = chunk['BusinessAgeDescription'].str.lower().fillna('')
        chunk['JobsReported'] = chunk['JobsReported'].fillna(0).astype(float)
        chunk['InitialApprovalAmount'] = chunk['InitialApprovalAmount'].astype(float)
        chunk['Latitude'] = chunk['Latitude'].fillna(0).astype(float)
        chunk['Longitude'] = chunk['Longitude'].fillna(0).astype(float)
        chunk['SBAOfficeCode'] = chunk['SBAOfficeCode'].fillna('')
        
        risk_scores = pd.DataFrame(index=chunk.index)
        risk_scores['RiskScore'] = 0.0
        risk_scores['RiskFlags'] = [[] for _ in range(len(chunk))]  # Still a list of lists
        
        multi_scores, multi_flags = self.check_multiple_applications(chunk)
        risk_scores['RiskScore'] += multi_scores
        # Add multi_flags directly as lists
        for i, flag_list in enumerate(multi_flags):
            risk_scores['RiskFlags'].iloc[i].extend(flag_list)
        
        # Rest of the method unchanged, just adjust indexing where needed
        invalid_addr_mask = chunk['BorrowerAddress_lower'].isin(self.INVALID_ADDRESS_SET)
        residential_scores = pd.Series(0.0, index=chunk.index)
        for indicator, weight in self.RESIDENTIAL_INDICATORS.items():
            residential_scores += chunk['BorrowerAddress_lower'].str.contains(indicator, na=False) * weight
        for indicator, weight in self.COMMERCIAL_INDICATORS.items():
            residential_scores += chunk['BorrowerAddress_lower'].str.contains(indicator, na=False) * weight
        residential_mask = residential_scores > 0.7
        risk_scores.loc[invalid_addr_mask, 'RiskScore'] += 10
        risk_scores.loc[invalid_addr_mask, 'RiskFlags'] = risk_scores.loc[invalid_addr_mask, 'RiskFlags'].apply(
            lambda x: x + ['Invalid/missing address']
        )
        risk_scores.loc[residential_mask, 'RiskFlags'] = risk_scores.loc[residential_mask, 'RiskFlags'].apply(
            lambda x: x + ['Residential address']
        )
        
        def extract_pattern_matches(name):
            if pd.isna(name) or not name:
                return 0.0, []
            score = 0.0
            flags = []
            matches = self.compiled_suspicious_pattern.finditer(name)
            for match in matches:
                matched_group = next(k for k, v in match.groupdict().items() if v)
                weight = self.pattern_weights[matched_group]
                original_pat = self.pattern_to_original[matched_group]
                score += 30 * weight
                flags.append(f'Suspicious pattern in name: {original_pat}')
            return score, flags
        
        results = chunk['BorrowerName_lower'].apply(extract_pattern_matches)
        risk_scores['RiskScore'] += results.apply(lambda x: x[0])
        for i, flag_list in enumerate(results.apply(lambda x: x[1])):
            risk_scores['RiskFlags'].iloc[i].extend(flag_list)
        
        per_employee = chunk['InitialApprovalAmount'] / chunk['JobsReported'].replace(0, np.nan)
        high_per_emp_mask = (chunk['JobsReported'] > 0) & (per_employee > 12000)
        very_high_per_emp_mask = (chunk['JobsReported'] > 0) & (per_employee > 14000)
        one_emp_mask = chunk['JobsReported'] == 1
        few_emp_mask = (chunk['JobsReported'] < 3) & (chunk['JobsReported'] > 0)
        risk_scores.loc[high_per_emp_mask, 'RiskScore'] += 15
        risk_scores.loc[high_per_emp_mask, 'RiskFlags'] = risk_scores.loc[high_per_emp_mask, 'RiskFlags'].apply(
            lambda x, idx=high_per_emp_mask[high_per_emp_mask].index: x + [f'High amount per employee: ${per_employee[idx[0]]:,.2f}'] if len(idx) > 0 else x
        )
        risk_scores.loc[very_high_per_emp_mask, 'RiskScore'] += 15
        risk_scores.loc[very_high_per_emp_mask, 'RiskFlags'] = risk_scores.loc[very_high_per_emp_mask, 'RiskFlags'].apply(
            lambda x: x + ['Extremely high amount per employee']
        )
        risk_scores.loc[one_emp_mask, 'RiskScore'] += 20
        risk_scores.loc[one_emp_mask, 'RiskFlags'] = risk_scores.loc[one_emp_mask, 'RiskFlags'].apply(
            lambda x: x + ['Only one employee reported: Very high fraud risk']
        )
        risk_scores.loc[few_emp_mask, 'RiskScore'] += 10
        risk_scores.loc[few_emp_mask, 'RiskFlags'] = risk_scores.loc[few_emp_mask, 'RiskFlags'].apply(
            lambda x: x + ['Fewer than three employees reported: Increased fraud risk']
        )
        
        exact_max_mask = chunk['InitialApprovalAmount'].isin([20832, 20833])
        risk_scores.loc[exact_max_mask, 'RiskScore'] += 25
        risk_scores.loc[exact_max_mask, 'RiskFlags'] = risk_scores.loc[exact_max_mask, 'RiskFlags'].apply(
            lambda x: x + ['Exact maximum loan amount detected']
        )
        
        lender_has_flags = risk_scores['RiskFlags'].apply(len) > 0
        high_risk_lender_mask = chunk['OriginatingLender'].isin(self.HIGH_RISK_LENDERS) & lender_has_flags
        lender_risk_scores = chunk['OriginatingLender'].map(lambda x: 15 * self.HIGH_RISK_LENDERS.get(x, 0)).astype(float).fillna(0.0)
        risk_scores.loc[high_risk_lender_mask, 'RiskScore'] += lender_risk_scores[high_risk_lender_mask]
        risk_scores.loc[high_risk_lender_mask, 'RiskFlags'] = risk_scores.loc[high_risk_lender_mask, 'RiskFlags'].apply(
            lambda x: x + ['High-risk lender']
        )
        
        demo_fields = ['Race', 'Gender', 'Ethnicity']
        missing_all_demo = chunk[demo_fields].apply(lambda x: x.str.lower().isin(self.UNANSWERED_SET)).all(axis=1) & lender_has_flags
        risk_scores.loc[missing_all_demo, 'RiskScore'] += 10
        risk_scores.loc[missing_all_demo, 'RiskFlags'] = risk_scores.loc[missing_all_demo, 'RiskFlags'].apply(
            lambda x: x + ['Missing all demographics']
        )
        
        high_risk_bt_mask = chunk['BusinessType'].isin(self.HIGH_RISK_BUSINESS_TYPES) & lender_has_flags
        bt_risk_scores = chunk['BusinessType'].map(lambda x: 15 * self.HIGH_RISK_BUSINESS_TYPES.get(x, 0)).astype(float).fillna(0.0)
        risk_scores.loc[high_risk_bt_mask, 'RiskScore'] += bt_risk_scores[high_risk_bt_mask]
        risk_scores.loc[high_risk_bt_mask, 'RiskFlags'] = risk_scores.loc[high_risk_bt_mask, 'RiskFlags'].apply(
            lambda x: x + ['High-risk business type']
        )
        
        paid_mask = chunk['LoanStatus'] == 'Paid in Full'
        risk_scores.loc[paid_mask, 'RiskScore'] -= 15
        
        for idx, loan in chunk.iterrows():
            time_score, time_flags = self.analyze_time_patterns(loan)
            net_score, net_flags = self.analyze_networks(loan)
            risk_scores.at[idx, 'RiskScore'] += time_score + net_score
            risk_scores.at[idx, 'RiskFlags'].extend(time_flags + net_flags)
        
        new_business_mask = (chunk['BusinessAgeDescription_lower'].isin(['new', 'existing less than 2 years']) &
                            (chunk['InitialApprovalAmount'] > 15000) & (chunk['JobsReported'] <= 2))
        estab_business_mask = chunk['BusinessAgeDescription_lower'].isin(['existing 2+ years', 'established'])
        risk_scores.loc[new_business_mask, 'RiskScore'] += 20
        risk_scores.loc[new_business_mask, 'RiskFlags'] = risk_scores.loc[new_business_mask, 'RiskFlags'].apply(
            lambda x: x + ['New business with high loan amount and few jobs']
        )
        risk_scores.loc[estab_business_mask, 'RiskScore'] -= 10
        risk_scores.loc[estab_business_mask, 'RiskFlags'] = risk_scores.loc[estab_business_mask, 'RiskFlags'].apply(
            lambda x: x + ['Established business - lower risk']
        )
        
        valid_coords = (chunk['Latitude'] != 0) & (chunk['Longitude'] != 0)
        if valid_coords.any():
            coord_keys = pd.Series([
                (round(lat, 4), round(lon, 4), str(date)) if valid else None
                for lat, lon, date, valid in zip(chunk['Latitude'], chunk['Longitude'], chunk['DateApproved'], valid_coords)
            ], index=chunk.index)
            for idx in chunk[valid_coords].index:
                coord_key = coord_keys[idx]
                self.seen_coordinates[coord_key].add(chunk.at[idx, 'LoanNumber'])
                cluster_size = len(self.seen_coordinates[coord_key])
                if cluster_size > 5:
                    cluster_score = 3 * log(max(cluster_size - 2, 1), 2)
                    risk_scores.at[idx, 'RiskScore'] += cluster_score
                    risk_scores.at[idx, 'RiskFlags'].append(
                        f'Geospatial cluster: {cluster_size:,} loans at coordinates {coord_key[:2]} on {coord_key[2]} (score: {cluster_score:.1f})'
                    )
        
        valid_office = (chunk['SBAOfficeCode'] != '') & chunk['DateApproved'].notna()
        if valid_office.any():
            for idx in chunk[valid_office].index:
                office_code = str(chunk.at[idx, 'SBAOfficeCode'])
                date = str(chunk.at[idx, 'DateApproved'])
                self.daily_office_counts[date][office_code] += 1
                self.total_days.add(date)
                days_processed = len(self.total_days)
                if days_processed > 1:
                    total_office_loans = sum(self.daily_office_counts[date].values())
                    office_loans_today = self.daily_office_counts[date][office_code]
                    if office_loans_today >= 5:
                        avg_daily_office_loans = total_office_loans / days_processed
                        if avg_daily_office_loans >= 5:
                            office_intensity = office_loans_today / avg_daily_office_loans
                            if office_intensity > 15:
                                cluster_score = 3 * log(max(office_loans_today - 3 + 1, 1), 2)
                                risk_scores.at[idx, 'RiskScore'] += cluster_score
                                risk_scores.at[idx, 'RiskFlags'].append(
                                    f'SBA office cluster: {office_loans_today:,} loans vs {avg_daily_office_loans:.1f} avg from office {office_code} on {date} (score: {cluster_score:.1f})'
                                )

        def clean_flags(flags):
            if not flags or not any(flags):
                return ''
            cleaned = [flag.strip() for flag in flags if flag.strip()]
            return '; '.join(cleaned)

        risk_scores['RiskFlags'] = risk_scores['RiskFlags'].apply(clean_flags)
        
        seq_mask = risk_scores['RiskFlags'].str.contains("Sequential loan numbers") & (risk_scores['RiskFlags'].str.count(';') > 1)
        risk_scores.loc[seq_mask, 'RiskScore'] *= 1.5
        
        for (flag1, flag2), multiplier in INTERACTION_RULES.items():
            interaction_mask = risk_scores['RiskFlags'].str.contains(flag1) & risk_scores['RiskFlags'].str.contains(flag2)
            risk_scores.loc[interaction_mask, 'RiskScore'] *= multiplier
            for idx in interaction_mask[interaction_mask].index:
                current_flags = risk_scores.at[idx, 'RiskFlags']
                if current_flags:
                    risk_scores.at[idx, 'RiskFlags'] = f"{current_flags}; Interaction: {flag1} + {flag2} (x{multiplier})"
                else:
                    risk_scores.at[idx, 'RiskFlags'] = f"Interaction: {flag1} + {flag2} (x{multiplier})"
        
        no_exact_max = ~risk_scores['RiskFlags'].str.contains("Exact maximum loan amount detected")
        few_flags = risk_scores['RiskFlags'].str.count(';') < 2
        cap_mask = no_exact_max & few_flags
        risk_scores.loc[cap_mask, 'RiskScore'] = np.minimum(risk_scores.loc[cap_mask, 'RiskScore'], 49)
        
        combined = pd.concat([chunk[['LoanNumber', 'BorrowerName']], risk_scores], axis=1)
        self.logger.debug("Risk score calculation complete for chunk")
        return combined 

    def process_chunks(self) -> None:
        total_rows = self.count_lines()  # Assumes optimized count_lines from earlier
        self.log_message('info', f"Starting to process {total_rows:,} loans...", 'green')
        
        # Define dtypes for all 77 columns to avoid type inference overhead
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
            'Unit Type': str, 'UTILITIES_PROCEED': float, 'Veteran': str, 'Zip': str
        }
        
        chunks = pd.read_csv(
            self.input_file,
            chunksize=self.chunk_size,
            dtype=dtypes,
            engine='c',  # Faster C engine for parsing
            low_memory=False
        )
        
        first_chunk = True
        pbar = tqdm(total=total_rows, unit='loans', desc='Processing loans')
        try:
            for chunk_number, chunk in enumerate(chunks, 1):
                target_loans = chunk[
                    (chunk['InitialApprovalAmount'] >= 5000) &
                    (chunk['InitialApprovalAmount'] < 21000)
                ].copy()
                if len(target_loans) > 0:
                    high_risk_loans = self.process_chunk(target_loans)
                    if len(high_risk_loans) > 0:
                        validated_loans = self.validate_high_risk_loans(high_risk_loans)
                        if len(validated_loans) > 0:
                            if first_chunk:
                                validated_loans.to_csv(self.output_file, index=False, compression='gzip')
                                first_chunk = False
                            else:
                                validated_loans.to_csv(self.output_file, mode='a', header=False, index=False, compression='gzip')
                            self.log_suspicious_findings(validated_loans)
                self.stats['processed'] += len(chunk)
                pbar.update(len(chunk))
                if chunk_number % 5 == 0:
                    self.log_interim_stats()
        except Exception as e:
            self.log_message('error', f"Error processing chunk: {str(e)}", 'red')
            raise
        finally:
            pbar.close()
            self.log_final_stats()

    @profile
    def validate_high_risk_loans(self, loans: pd.DataFrame) -> pd.DataFrame:
        self.logger.debug("Validating high risk loans")
        def validate_loan(loan: pd.Series) -> bool:
            if loan['BorrowerName'] in self.known_businesses:
                return False
            validation_score = 0
            flags = loan['RiskFlags'].split('; ')  # Split string into list of flags
            if len(flags) < 2 or (len(flags) == 1 and not flags[0]):  # Check for empty or single empty flag
                return False
            if loan['JobsReported'] > 0:
                amount_per_job = loan['InitialApprovalAmount'] / loan['JobsReported']
                if amount_per_job < 8000:
                    validation_score -= 20
            if (loan['BusinessType'] in self.LEGITIMATE_BUSINESS_TYPES and 
                str(loan['NAICSCode']) not in self.GENERIC_NAICS):
                validation_score -= 15
            if str(loan['OriginatingLender']) not in self.HIGH_RISK_LENDERS:
                validation_score -= 10
            if loan['LoanStatus'] == 'Paid in Full':
                validation_score -= 15
            if not all(str(loan.get(field, '')).lower() in self.UNANSWERED_SET for field in ['Race', 'Gender', 'Ethnicity']):
                validation_score -= 10
            if not pd.isna(loan['BorrowerCity']) and not pd.isna(loan['BorrowerState']):
                validation_score -= 5
            return validation_score > -30
        valid_mask = loans.apply(validate_loan, axis=1)
        self.logger.debug("High risk loan validation complete")
        return loans[valid_mask]

    @profile
    def process_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        self.logger.debug("Processing a chunk of loans")
        risk_scores = self.calculate_risk_scores(chunk)
        high_risk_mask = risk_scores['RiskScore'] >= self.risk_threshold
        high_risk_scores = risk_scores[high_risk_mask]
        if len(high_risk_scores) > 0:
            high_risk_loans = pd.merge(
                high_risk_scores,
                chunk[high_risk_mask],
                on=['LoanNumber', 'BorrowerName']
            )
            self.stats['suspicious'] += len(high_risk_loans)
            self.logger.debug("High risk loans found in chunk")
            return high_risk_loans
        self.logger.debug("No high risk loans in this chunk")
        return pd.DataFrame()

    def count_lines(self) -> int:
        self.log_message('info', "Counting total lines in file...", 'cyan')
        try:
            with open(self.input_file, 'rb') as f:
                buf_size = 1024 * 1024  # 1MB buffer
                lines = 0
                buffer = f.read(buf_size)
                while buffer:
                    lines += buffer.count(b'\n')
                    buffer = f.read(buf_size)
                lines -= 1  # Subtract header
            self.log_message('info', f"Found {lines:,} lines to process", 'cyan')
            return lines
        except FileNotFoundError:
            self.log_message('error', f"Input file '{self.input_file}' not found!", 'red')
            raise

    def log_suspicious_findings(self, suspicious_loans: pd.DataFrame) -> None:
        for _, loan in suspicious_loans.iterrows():
            risk_details = [
                "SUSPICIOUS LOAN DETECTED",
                f"Name: {loan['BorrowerName']}",
                f"Amount: ${loan['InitialApprovalAmount']:,.2f}",
                f"Risk Score: {loan['RiskScore']:.1f}",
                f"Location: {loan['BorrowerCity']}, {loan['BorrowerState']}",
                f"Business Type: {loan['BusinessType']}",
                f"Jobs Reported: {loan['JobsReported']}",
                f"Demographics: {loan['Race']} | {loan['Gender']} | {loan['Ethnicity']}",
                f"Lender: {loan['OriginatingLender']}",
                f"Status: {loan['LoanStatus']}",
                f"Risk Flags: {loan['RiskFlags']}"
            ]
            self.log_message('warning', "\n    " + "\n    ".join(filter(None, risk_details)), 'yellow')

    def log_message(self, level: str, message: str, color: str = None) -> None:
        msg = colored(message, color) if color else message
        getattr(self.logger, level.lower())(msg)

    def log_interim_stats(self) -> None:
        suspicious_pct = (self.stats['suspicious'] / self.stats['processed']) * 100
        self.log_message('info',
            f"\nInterim Stats:\nProcessed: {self.stats['processed']:,} loans\nSuspicious: {self.stats['suspicious']:,} loans ({suspicious_pct:.2f}%)",
            'cyan')

    def log_final_stats(self) -> None:
        suspicious_pct = (self.stats['suspicious'] / self.stats['processed']) * 100
        self.log_message('info',
            f"\nFinal Results:\nTotal Processed: {self.stats['processed']:,} loans\nTotal Suspicious: {self.stats['suspicious']:,} loans ({suspicious_pct:.2f}%)\nResults saved to: {self.output_file}",
            'green')

    def extra_debug_info(self) -> None:
        self.log_message('debug', f"Address to businesses mapping count: {len(self.address_to_businesses)}", 'blue')
        self.log_message('debug', f"Business to addresses mapping count: {len(self.business_to_addresses)}", 'blue')
        self.log_message('debug', f"Lender batches count: {len(self.lender_batches)}", 'blue')
        self.log_message('debug', f"Lender sequences count: {len(self.lender_sequences)}", 'blue')
        self.log_message('debug', f"Total loans processed: {self.stats['processed']}", 'blue')
        self.log_message('debug', f"Total suspicious loans detected: {self.stats['suspicious']}", 'blue')

def main():
    if not os.path.exists(CSV_FILENAME):
        asyncio.run(download_and_extract_csv())
    input_file = CSV_FILENAME
    output_file = "suspicious_loans.csv"
    risk_threshold = 100
    chunk_size = 50000
    processor = PPPLoanProcessor(input_file, output_file, risk_threshold, chunk_size)
    try:
        processor.process_chunks()
        processor.extra_debug_info()
    except KeyboardInterrupt:
        processor.log_message('warning', "\nProcessing interrupted by user", 'yellow')
    except Exception as e:
        processor.log_message('error', f"\nError: {str(e)}", 'red')

if __name__ == "__main__":
    main()
