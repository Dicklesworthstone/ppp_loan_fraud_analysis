import os
import re
import zipfile
import asyncio
import aiofiles
import httpx
import logging
import pandas as pd
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
            'name_pattern': 0.05
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
            r'stack[sz]': 0.9,
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
            r'lucky': 0.8,
            r'jackpot': 0.9,
            r'lottery': 0.9,
            r'winner': 0.8,
            r'casino': 0.85,
            r'vegas': 0.8,
            r'cards': 0.7,
            r'dice': 0.8,
            r'lucky\s*charm': 0.85,
            r'blessed': 0.7,
            r'deez\snutz': 0.95,
            r'\bno\s*money\s*no\s*honey\b': 0.9,
            r'\bscam\s*central\b': 0.95,
            r"\bfool's\s*gold\b": 0.95,
            r'\bsketchy\s*inc\b': 0.9,
            r'\bshady\s*business\b': 0.95,
            r'\bquick\s*buck\b': 0.9,
            r'\bget\s*rich\s*quick\b': 0.9,
            r'\binstant\s*wealth\b': 0.95,
            r'\bno\s*credit\s*needed\b': 0.85,
            r'\bcon\s*game\b': 0.9,
            r'\bchump\s*change\b': 0.85,
            r'relief': 0.9,
            r'aid': 0.9,
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
            r'ice': 0.85,
            r'drip': 0.9,
            r'designer': 0.85,
            r'luxury': 0.85,
            r'vip': 0.85,
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
            r'metaverse': 0.95,
            r'crypto': 0.95,
            r'bitcoin': 0.95,
            r'nft': 0.95,
            r'web3': 0.95,
            r'dogecoin': 0.95,
            r'meme': 0.9,
            r'viral': 0.9
        }
        self.LEGITIMATE_KEYWORDS = {
            'consulting', 'services', 'solutions', 'associates',
            'partners', 'group', 'inc', 'llc', 'ltd', 'corporation',
            'management', 'enterprises', 'international', 'systems'
        }
        self.known_businesses = set()
        self.date_patterns = defaultdict(lambda: defaultdict(list))
        self.lender_loan_sequences = defaultdict(list)
        self.ZIP_CLUSTER_THRESHOLD = 3
        self.SEQUENCE_THRESHOLD = 3
        # New attributes for network analysis
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
            r'consulting.*llc': 0.4,
            r'holdings.*llc': 0.4,
            r'enterprise.*llc': 0.4,
            r'solutions.*llc': 0.4,
            r'services.*llc': 0.3,
            r'investment.*llc': 0.4,
            r'\d+.*consulting': 0.6,
            r'[A-Z]\s*&\s*[A-Z]\s+': 0.5,
            r'[A-Z]{3,}': 0.4
        }

    def validate_address(self, address: str) -> tuple[bool, List[str]]:
        flags = []
        address = str(address).lower().strip()
        if pd.isna(address) or address in ('n/a', 'none', ''):
            return False, ['Invalid/missing address']
        residential_indicators = {'apt': 0.7, 'unit': 0.5, 'suite': 0.3, '#': 0.4, 'po box': 0.8, 'box': 0.7, 'residential': 0.9, 'house': 0.8}
        commercial_indicators = {'plaza', 'building', 'tower', 'office', 'complex', 'center', 'mall', 'commercial', 'industrial', 'park'}
        residential_score = 0
        for indicator, weight in residential_indicators.items():
            if indicator in address:
                residential_score += weight
        for indicator in commercial_indicators:
            if indicator in address:
                residential_score -= 0.5
        if residential_score > 0.7:
            flags.append('Residential address')
        if len(address) < 10:
            flags.append('Suspiciously short address')
        fake_patterns = [r'\d{1,3}\s*[a-z]+\s*(st|street|ave|avenue|rd|road)', r'p\.?o\.?\s*box\s*\d+', r'general\s*delivery']
        if any(re.search(pattern, address) for pattern in fake_patterns):
            flags.append('Potentially fake address pattern')
        return (len(flags) == 0), flags

    def validate_business_name(self, name: str) -> tuple[float, List[str]]:
        flags = []
        risk_score = 0
        name = str(name).lower().strip()
        if name in self.known_businesses:
            return 0, []
        if pd.isna(name) or name in ('n/a', 'none', ''):
            return 1.0, ['Invalid/missing business name']
        for pattern, weight in self.SUSPICIOUS_PATTERNS.items():
            if re.search(pattern, name):
                risk_score += weight
                flags.append(f'Suspicious name pattern: {pattern}')
        legitimate_count = sum(1 for keyword in self.LEGITIMATE_KEYWORDS if keyword in name)
        risk_score -= (legitimate_count * 0.2)
        if re.match(r'^[a-z]+\s+[a-z]+$', name):
            risk_score += 0.4
            flags.append('Personal name only')
        if re.search(r'[@#$%^&*]', name):
            risk_score += 0.5
            flags.append('Suspicious characters in name')
        return max(0, min(1, risk_score)), flags

    def check_multiple_applications(self, loan: pd.Series) -> tuple[bool, List[str]]:
        flags = []
        name_key = f"{loan['BorrowerName']}_{loan['BorrowerCity']}_{loan['BorrowerState']}"
        address_key = f"{loan['BorrowerAddress']}_{loan['BorrowerCity']}_{loan['BorrowerState']}"
        exact_key = f"{name_key}_{loan['InitialApprovalAmount']}"
        if exact_key in self.seen_loans:
            flags.append("Duplicate exact loan detected")
            self.stats['duplicate_loans'] += 1
        self.borrower_loans[name_key] = self.borrower_loans.get(name_key, 0) + 1
        if self.borrower_loans[name_key] > 1:
            flags.append(f"Multiple loans ({self.borrower_loans[name_key]}) for same borrower")
        self.seen_addresses[address_key] = self.seen_addresses.get(address_key, 0) + 1
        if self.seen_addresses[address_key] > 2:
            flags.append(f"Address used {self.seen_addresses[address_key]} times")
        self.seen_loans.add(exact_key)
        return len(flags) > 0, flags

    def analyze_time_patterns(self, loan: pd.Series) -> tuple[float, List[str]]:
        risk_score = 0
        flags = []
        date = str(loan['DateApproved'])
        zip_code = str(loan['BorrowerZip'])[:5]
        lender = str(loan['OriginatingLender'])
        loan_number = str(loan['LoanNumber'])
        self.date_patterns[date][zip_code].append({
            'loan_number': loan_number,
            'business_type': loan['BusinessType'],
            'amount': loan['InitialApprovalAmount'],
            'lender': lender
        })
        zip_cluster = self.date_patterns[date][zip_code]
        if len(zip_cluster) >= self.ZIP_CLUSTER_THRESHOLD:
            amounts = [l['amount'] for l in zip_cluster]  # noqa: E741
            business_types = [l['business_type'] for l in zip_cluster]  # noqa: E741
            if max(amounts) - min(amounts) < min(amounts) * 0.1:
                risk_score += 20
                flags.append(f"Part of cluster: {len(zip_cluster)} similar loans in ZIP {zip_code} on {date}")
            if len(set(business_types)) == 1:
                risk_score += 15
                flags.append(f"Cluster of identical business types in ZIP {zip_code}")
        self.lender_loan_sequences[lender].append(loan_number)
        if len(self.lender_loan_sequences[lender]) >= 2:
            recent_loans = self.lender_loan_sequences[lender][-self.SEQUENCE_THRESHOLD:]
            if self.is_roughly_sequential(recent_loans):
                risk_score += 25
                flags.append(f"Sequential loan numbers from {lender}")
        return risk_score, flags

    def is_roughly_sequential(self, loan_numbers: List[str]) -> bool:
        if len(loan_numbers) < 2:
            return False
        numbers = []
        for loan_num in loan_numbers:
            matches = re.findall(r'\d+', loan_num)
            if matches:
                numbers.append(int(matches[-1]))
        if len(numbers) < 2:
            return False
        numbers.sort()
        gaps = [numbers[i+1] - numbers[i] for i in range(len(numbers)-1)]
        avg_gap = sum(gaps) / len(gaps)
        return avg_gap < 10 and all(gap < 20 for gap in gaps)

    def analyze_address_type(self, address: str) -> float:
        address = address.lower()
        score = 0
        for indicator, weight in self.RESIDENTIAL_INDICATORS.items():
            if indicator in address:
                score += weight
        for indicator, weight in self.COMMERCIAL_INDICATORS.items():
            if indicator in address:
                score += weight
        if re.search(r'\d+\s*-\s*\d+', address):
            score += 0.6
        if re.search(r'(st|street|ave|avenue|road|rd)\s*$', address):
            score += 0.4
        return score

    def analyze_name_patterns(self, business_names: Set[str]) -> Tuple[float, List[str]]:
        score = 0
        flags = []
        name_patterns = defaultdict(int)
        names = [name.lower() for name in business_names]
        for name in names:
            for pattern, weight in self.SUSPICIOUS_NAME_PATTERNS.items():
                if re.search(pattern, name):
                    name_patterns[pattern] += 1
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
        return score, flags

    def check_sequential_pattern(self, loans: List[dict]) -> Tuple[float, List[str]]:
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
        return score, flags

    def analyze_networks(self, loan: pd.Series) -> tuple[float, List[str]]:
        risk_score = 0
        flags = []
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
                    flags.append("Multiple businesses at residential address")
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
            amounts = [l['amount'] for l in current_batch]  # noqa: E741
            if max(amounts) - min(amounts) < min(amounts) * 0.1:
                batch_names = {l['business_name'] for l in current_batch}  # noqa: E741
                pattern_score, pattern_flags = self.analyze_name_patterns(batch_names)
                risk_score += 15 + pattern_score
                flags.append(f"Part of suspicious batch: {len(current_batch)} similar loans from same lender")
                flags.extend(pattern_flags)
        lender_base = f"{loan['OriginatingLender']}_{loan['OriginatingLenderLocationID']}"
        self.lender_sequences[lender_base] = self.lender_sequences[lender_base][-50:] + [{
            'loan_number': loan['LoanNumber'],
            'business_name': loan['BorrowerName'],
            'date': loan['DateApproved']
        }]
        recent_loans = self.lender_sequences[lender_base][-5:]
        sequence_score, sequence_flags = self.check_sequential_pattern(recent_loans)
        if sequence_score > 0:
            risk_score += sequence_score
            flags.extend(sequence_flags)
        return risk_score, flags

    def calculate_risk_scores(self, chunk: pd.DataFrame) -> pd.DataFrame:
        def score_loan(loan: pd.Series) -> pd.Series:
            score = 0
            flags = []
            has_duplicates, duplicate_flags = self.check_multiple_applications(loan)
            if has_duplicates:
                score += 30
                flags.extend(duplicate_flags)
            valid, address_flags = self.validate_address(loan['BorrowerAddress'])
            if not valid:
                score += 10
            flags.extend(address_flags)
            name = str(loan['BorrowerName']).lower()
            for pattern, weight in self.SUSPICIOUS_PATTERNS.items():
                if re.search(pattern, name):
                    score += 30 * weight
                    flags.append(f'Suspicious pattern in name: {pattern}')
            if loan['JobsReported'] > 0:
                amount = float(loan['InitialApprovalAmount'])
                per_employee = amount / loan['JobsReported']
                if per_employee > 12000:
                    score += 15
                    flags.append(f'High amount per employee: ${per_employee:,.2f}')
                    if per_employee > 14000:
                        score += 15
                        flags.append('Extremely high amount per employee')
                if loan['JobsReported'] == 1:
                    score += 20
                    flags.append("Only one employee reported: Very high fraud risk")
                elif loan['JobsReported'] < 3:
                    score += 10
                    flags.append("Fewer than three employees reported: Increased fraud risk")
            if loan['InitialApprovalAmount'] == 20832 or loan['InitialApprovalAmount'] == 20833:
                score += 25
                flags.append("Exact maximum loan amount detected")
            lender = str(loan['OriginatingLender'])
            if lender in self.HIGH_RISK_LENDERS and len(flags) > 0:
                score += 15 * self.HIGH_RISK_LENDERS[lender]
                flags.append('High-risk lender')
            demographic_fields = ['Race', 'Gender', 'Ethnicity']
            missing_demographics = sum(1 for field in demographic_fields if str(loan.get(field, '')).lower() in ['unanswered', 'unknown'])
            if missing_demographics == len(demographic_fields) and len(flags) > 1:
                score += 10
                flags.append('Missing all demographics')
            business_type = str(loan['BusinessType'])
            if business_type in self.HIGH_RISK_BUSINESS_TYPES:
                if len(flags) > 1:
                    score += 15 * self.HIGH_RISK_BUSINESS_TYPES[business_type]
                    flags.append('High-risk business type')
            if loan['LoanStatus'] == 'Paid in Full':
                score -= 15
            time_score, time_flags = self.analyze_time_patterns(loan)
            score += time_score
            flags.extend(time_flags)
            network_score, network_flags = self.analyze_networks(loan)
            score += network_score
            flags.extend(network_flags)
            if any("Sequential loan numbers" in flag for flag in flags) and len(flags) > 1:
                score = score * 1.5
            final_score = score
            if "Exact maximum loan amount detected" not in flags and len(flags) < 2:
                final_score = min(49, final_score)
            return pd.Series({
                'RiskScore': final_score,
                'RiskLevel': 'Very High Risk' if final_score >= 75 else 'High Risk' if final_score >= 50 else 'Medium Risk' if final_score >= 25 else 'Low Risk',
                'RiskFlags': '; '.join(flags)
            })
        results = chunk.apply(score_loan, axis=1)
        return pd.concat([chunk[['LoanNumber', 'BorrowerName']], results], axis=1)

    def process_chunks(self) -> None:
        total_rows = self.count_lines()
        self.log_message('info', f"Starting to process {total_rows:,} loans...", 'green')
        chunks = pd.read_csv(
            self.input_file,
            chunksize=self.chunk_size,
            dtype={
                'LoanNumber': str,
                'BorrowerName': str,
                'BorrowerAddress': str,
                'BorrowerCity': str,
                'BorrowerState': str,
                'InitialApprovalAmount': float,
                'JobsReported': float,
                'ForgivenessAmount': float,
                'NAICSCode': str,
                'BusinessType': str,
                'Race': str,
                'Gender': str,
                'Ethnicity': str,
                'Veteran': str,
                'OriginatingLender': str,
                'OriginatingLenderLocationID': str,
                'BusinessAgeDescription': str,
                'LoanStatus': str,
                'DateApproved': str,
                'BorrowerZip': str
            },
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
                                validated_loans.to_csv(self.output_file, index=False)
                                first_chunk = False
                            else:
                                validated_loans.to_csv(self.output_file, mode='a', header=False, index=False)
                            self.log_suspicious_findings(validated_loans)
                self.stats['processed'] += len(chunk)
                pbar.update(len(chunk))
                if chunk_number % 10 == 0:
                    self.log_interim_stats()
        except Exception as e:
            self.log_message('error', f"Error processing chunk: {str(e)}", 'red')
            raise
        finally:
            pbar.close()
            self.log_final_stats()

    def validate_high_risk_loans(self, loans: pd.DataFrame) -> pd.DataFrame:
        def validate_loan(loan: pd.Series) -> bool:
            if loan['BorrowerName'] in self.known_businesses:
                return False
            validation_score = 0
            flags = str(loan['RiskFlags']).split(';')
            if len(flags) < 2:
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
            if not all(str(loan.get(field, '')).lower() in ['unanswered', 'unknown'] for field in ['Race', 'Gender', 'Ethnicity']):
                validation_score -= 10
            if not pd.isna(loan['BorrowerCity']) and not pd.isna(loan['BorrowerState']):
                validation_score -= 5
            return validation_score > -30
        valid_mask = loans.apply(validate_loan, axis=1)
        return loans[valid_mask]

    def process_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
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
            return high_risk_loans
        return pd.DataFrame()

    def count_lines(self) -> int:
        self.log_message('info', "Counting total lines in file...", 'cyan')
        try:
            with open(self.input_file, 'rb') as f:
                count = sum(1 for _ in f) - 1
            self.log_message('info', f"Found {count:,} lines to process", 'cyan')
            return count
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

def main():
    if not os.path.exists(CSV_FILENAME):
        asyncio.run(download_and_extract_csv())
    input_file = CSV_FILENAME
    output_file = "suspicious_loans.csv"
    risk_threshold = 90
    chunk_size = 10000
    processor = PPPLoanProcessor(input_file, output_file, risk_threshold, chunk_size)
    try:
        processor.process_chunks()
    except KeyboardInterrupt:
        processor.log_message('warning', "\nProcessing interrupted by user", 'yellow')
    except Exception as e:
        processor.log_message('error', f"\nError: {str(e)}", 'red')

if __name__ == "__main__":
    main()
