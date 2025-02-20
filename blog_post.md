# PPP Loan Fraud: A Data Science Detective Story

![PPP Illustration](https://github.com/Dicklesworthstone/ppp_loan_fraud_analysis/raw/main/ppp_illustration.webp)

The Paycheck Protection Program (PPP) was a cornerstone of U.S. economic relief during the COVID-19 crisis, disbursing nearly $800 billion to small businesses. But with massive funds came massive fraud—hundreds of thousands, possibly over a million, loans exploited by opportunists. When I first explored the PPP’s 8.4GB dataset in a few days ago, I anticipated uncovering fraud. What I didn't expect was that I would end up with a system that systematically uncovered fraud networks and patterns across millions of loans, implicating hundreds of thousands of fraudulent borrowers and hundreds of corrupt lenders and agents. 

This project comprises three scripts working in tandem:

- `simple_loan_fraud_score.py`: Processes the full 8.4GB `ppp-full.csv`, scoring each loan’s fraud risk and flagging those above a threshold (100) into `suspicious_loans.csv`. 
    - Using this threshold, the resulting suspicious_loans.csv is 2.55GB. 
- `sort_suspicious_loans_by_risk_score.py`: Sorts and filters these loans by risk (default cutoff: 140), producing `suspicious_loans_sorted.csv`. 
    - Using these settings, the system flags 1,190,352 suspicious loans out of 6,267,512 loans in the $5k-$22k range. This results in a suspicious loan rate of 19% and a suspicious_loans_sorted.csv file that is 1.37GB.
- `analyze_patterns_in_suspicious_loans.py`: Applies advanced statistical and machine learning techniques—think chi-square tests, XGBoost, and SHAP values—to uncover fraud networks and refine detection.

What started as a hunch about sloppy fraudsters has become a robust tool revealing everything from "Wakanda LLC" scams to subtle lender collusion. Here’s how it works, why it matters, and what it’s taught me about catching fraud in big data. If you'd like a more technical, formal explanation of the system, you should read the readme.md file in the repo [here](https://github.com/Dicklesworthstone/ppp_loan_fraud_analysis/blob/main/README.md), and of course, you can read the code itself. It's a total of 3,281 lines of code across 3 files (1,113 + 184 + 1,984), so it's fairly readable if you know Python and know a bit about statistics and data analysis.

## Why This Matters

PPP fraud wasn't just a matter of people gaming the system– it was stealing for every single taxpayer in the US, and adding massively to the national debt. If those payments did in fact prevent a worthy business from going bankrupt so it could survive to live another day, then that's one thing. We can debate and independently decide whether that benefit was worth the impact to the national debt and deficit. But the outright fraud, taking money that was probably mostly wasted on silly purchases for purely personal gain, is a different matter. These are not small dollar amounts here– the average fraudster got close to $20,000, and I believe that the number of fraudulent loans was easily in the hundreds of thousands, and probably even over a million. We should all be able to agree that this was a disgrace and that the people responsible should be held accountable. 

In particular, the corrupt lenders and agents who facilitated this should be prosecuted to the fullest extent of the law. I suspect that in many cases they actively recruited fraudulant borrowers and guided them step by step about how to commit the fraud and what to write on their applications, knowing full well that the information was fake but that the loans would be approved anyway because of the extreme urgency and lack of proper controls and systems. Not only that, I believe many of these agents and lenders received payments from the SBA for originating and processing these loans. If a corrupt lender originated 100 fake loans-- and I believe that a very large number of them did at least that many-- then we could be talking about $20k*100 = $2 million in fraud from just one bad actor. It takes a lot of honest taxpayers to pay for that level of fraud, and it happened in cities across the country at a scale that is simply staggering.

## The Upshot

If you just want to see the final results of running the initial fraud risk scoring system and then processing that with the analysis system, you can find the final results [here](https://raw.githubusercontent.com/Dicklesworthstone/ppp_loan_fraud_analysis/refs/heads/main/final_output_of_analysis_step_in_ppp_loan_fraud_analysis.txt). 

You can also easily run the code yourself since everything is publicly available, both the data and the code. It's fairly easy to set up and run, and if you leave it running overnight on a decently fast machine, it will process the entire 8.4GB dataset and give you the final results so you can verify everything yourself from first principles. You can also modify any parameters you want to try out different analyses, or remove aspects of the fraud risk scoring system to see how it performs without them, or change the threshold for what is considered "suspicious" and what is not, and then see how that impacts the results of the analysis step.

If you work for the government, you can also use the trained XGBoost model to score any loan based on the same features used in the analysis, and flag any loans that score above a certain threshold as potentially fraudulent.

## The Art of Finding Fraud: When Criminals Tell on Themselves

Fraudsters left digital fingerprints— some laughably obvious, others subtle:

- **Blatant Blunders**: Names like "Reparations Inc." or "Dodge Hellcat LLC" (caught by regex in `simple_loan_fraud_score.py`) scream fraud, scoring 95% suspicion weights.
- **Address Overlaps**: Dozens of "businesses" at one apartment (e.g., "Apt 4B, 123 Main St")—`analyze_networks` flags these with 15 points per overlap.
- **Time Tells**: Sequential loan numbers or 50 loans in one ZIP code on one day (e.g., June 1, 2021, in 90210)—temporal clustering adds 20+ points.
- **SBA Spikes**: One office processing 100 loans daily vs. a 6-loan average—`daily_office_counts` catches this anomaly.
- **Name Nuances**: Suspicious loans average 1.8 words vs. 2.5 overall (`analyze_name_patterns`), hinting at rushed fakes.

Fraudsters thought speed hid them, but their haste— batch submissions, generic names— created patterns we could systematize and catch.

Many PPP fraudsters seemed to operate under the assumption that no one would ever actually look at their applications. The patterns I discovered while building this detection system ranged from the blindingly obvious to the surprisingly subtle.

### Blatant Fraud Examples

Imagine applying for a federal loan and naming your business "Fake Business LLC" or "PPP Loan Kingz." Surely no one would be that obvious, right? Wrong. During my analysis, I found loan applications from businesses with names that are just blatantly fake and ridiculous. Some personal favorites:

- Companies named after fictional places like "Wakanda"
- Businesses with names including phrases like "Free Money" and "Get Paid"
- A "Reparations LLC" that is perhaps making a political statement about the intent of the fraudulently obtained loan
- References to luxury brands, suggesting someone's wishlist rather than a real business

Beyond these standout names, the system examines more subtle patterns in business names. In `analyze_patterns_in_suspicious_loans.py`, the `analyze_business_name_patterns` function checks for indicators like multiple spaces or special characters, which can suggest automated or sloppy entries. It also calculates metrics such as name length and word count. For example, the code compares the average name length between suspicious and overall loans, using statistical tests like the t-test to determine if differences are meaningful. This helps identify less obvious red flags, like names that are unusually short or generic, complementing the detection of blatant cases.

### Beyond the Obvious: The Digital Fingerprints of Fraud

But not all fraudsters were quite so blatant. The more interesting cases required looking at subtle patterns that emerged only when analyzing thousands of applications together. Some of these indicators included:

- Clusters of businesses registered to the same residential address
- Sequential (or nearly sequential) loan applications submitted all on the same day, as measured by the digits of the loan number
- Identical loan amounts for supposedly different businesses
- Clusters of similar looking loans in the same ZIP code on the same day
- Clusters of loans all originating from the same lender in the same area and on the same day, at a scale that makes them suspicious even if they don't have any other obvious red flags
- Clusters of loans all originating from the same SBA office in the same area and on the same day

What makes these patterns fascinating is that they often reveal how fraudsters, in trying to avoid detection, actually create new patterns that make them stand out. Their attempts to hide create even clearer patterns.

The real challenge – and what drove me to build this detection system – was finding ways to automatically identify these patterns across millions of records. When you're dealing with an 8.4GB CSV file, you can't exactly skim through it looking for suspicious entries. You need a systematic approach that can catch both the obvious cases and the more subtle patterns that emerge only when you look at the data as a whole.

### Digging Deeper: How the Code Spots the Sneaky Stuff

Let’s investigate further how the system turns these fraudster slip-ups into hard data. In `simple_loan_fraud_score.py`, the `SUSPICIOUS_PATTERNS` dictionary contains a list of over 100 regex patterns— like `r'quick\s*cash'` or `r'pandemic\s*profit'`— each assigned a weight like 0.95 for maximum suspicion. When "Quick Cash LLC" pops up, it’s not just flagged; it’s hit with a 28.5-point boost (30 * 0.95) to its risk score, thanks to vectorized string matching that scans thousands of names in seconds. The code doesn’t stop there— it checks for structural clues too, like `r'\s{2,}'` for multiple spaces, hinting at sloppy copy-paste jobs, adding another layer of detection.

Then there’s the address game. The `analyze_networks` function doesn’t just count businesses at "Apt 4B"; it builds a network graph with `defaultdict(set)`—if "ABC Consulting" and "XYZ Solutions" share that apartment, and "XYZ" links to another address with "123 Holdings," the risk score spikes by 15 points per overlap, plus 5 points per connected business. It’s a web of deceit unmasked by simple set operations, scaled to handle millions of records in memory-efficient chunks.

Time-based patterns reveal even more. The `daily_zip_counts` and `daily_office_counts` in `calculate_risk_scores` track loan volumes by ZIP code and SBA office daily. If ZIP 90210 jumps from an average of 8 loans to 50 on June 1, 2021, the system calculates an intensity score—say, 20 + 10 * log(46, 2)—and slaps 34 points on each loan. Sequential loan numbers? The `check_sequential_pattern` method digs into the last five digits (e.g., 7801, 7802, 7803), using NumPy’s `diff` to spot gaps under 20, adding 25 points when it smells a batch job. These aren’t random checks; they’re rooted in fraudsters’ love for automation—too fast, too uniform, too obvious once you know where to look.

The secondary analysis in `analyze_patterns_in_suspicious_loans.py` takes it further. The `analyze_name_patterns` function doesn’t just count words (1.8 vs. 2.5)—it runs a Mann-Whitney U test to confirm the difference isn’t chance (p<0.01), while `extract_names_optimized` uses parallel processing with `joblib` to split "John Doe LLC" into first and last names, caching results with `@lru_cache` to avoid re-parsing duplicates. This isn’t just about catching "Dodge Hellcat LLC"; it’s about proving statistically that suspicious names are shorter, simpler, and more repetitive than legit ones.

### Overconfidence Meets Overlap

What’s wilder still is how fraudsters’ overconfidence amplifies these patterns. Take demographics—they often leave `Race`, `Gender`, and `Ethnicity` as "Unanswered" across the board, thinking it’s safer to skip details. But `analyze_demographic_patterns` catches this, showing 60% of suspicious loans vs. 40% overall lack this data, with a chi-square p-value under 0.001 proving it’s no fluke. The `MissingDemographics` feature in `prepare_enhanced_features` turns this laziness into a 10-point risk bump when paired with other flags.

Or consider loan amounts. The code flags exact matches to $20,832 or $20,833—PPP’s max for one employee—with 25 points in `calculate_risk_scores`. Why? Fraudsters loved hitting the ceiling, assuming it blended in. But when `analyze_loan_amount_distribution` compares suspicious ($19k mean) to overall ($17k mean) amounts with a t-test (p<0.05), it’s clear: they overshot consistently. Add in `IsRoundAmount` checking for multiples of 100, and you’ve got another subtle tell—fraudsters pick neat numbers, legit businesses don’t.

### Systemic Slip-Ups

The system doesn’t just nab lone wolves—it sniffs out rings. The `daily_office_counts` tracks SBA office spikes—100 loans from office "0575" on July 1, 2021, vs. a 6-loan average triggers a logarithmic 15-point boost per loan. Why’s that damning? It suggests insider help or exploited loopholes, a pattern `analyze_lender_patterns` confirms with lenders like "Kabbage, Inc." showing 2.3x over-representation (chi-square p<0.001). The `XGBoostAnalyzer` ties it together, ranking `OriginatingLender` dummies high (e.g., 0.12 importance), with SHAP values showing how "Kabbage" loans amplify fraud odds by 0.15 when paired with residential flags.

The range of clues—from obvious fake names to hidden SBA patterns—reveals how fraudsters operate: bold yet sloppy, clever yet rushed. The code exploits every angle, from regex to ML, proving that even the craftiest crooks leave tracks when you’ve got 8.4GB of data and the right tools to sift it.

## How It Works: The Technical Details

Let's dive into the technical details of how this system actually works. I'll break this down piece by piece, explaining both the high-level concepts and the nitty-gritty implementation details. Even if you're not a programmer, stick with me – I'll try to make the concepts clear while still providing enough depth for those who want to understand every line of code.

### The Data Size Challenge

When you're dealing with an 8.4GB CSV file containing millions of loan records, you can't just load it into memory and start analyzing. A naive approach would crash most computers. Here's how we handle this beast efficiently:

```python
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
            'InitialApprovalAmount': float,
            'JobsReported': float,
            'ForgivenessAmount': float,
            'NAICSCode': str,
            # ... other column definitions
        },
        engine='c',  # Faster C engine for parsing
        low_memory=False
    )
```

This code might look simple, but it's doing several sophisticated things to handle massive data efficiently:

1. **Chunked Processing**: Instead of reading the entire 8GB file at once, we use pandas' `chunksize` parameter to read it in manageable pieces (defaulting to 50,000 rows at a time).

2. **Memory-Efficient Data Types**: Notice how we explicitly define the data type for each column? This is crucial for memory optimization:
   - `LoanNumber` as string prevents pandas from trying to convert it to integers (which would break with leading zeros)
   - `InitialApprovalAmount` as float is more memory-efficient than the default float64
   - `NAICSCode` as string prevents pandas from treating industry codes as numbers

3. **Progress Tracking**: We count total rows upfront so we can show a progress bar:
```python
pbar = tqdm(total=total_rows, unit='loans', desc='Processing loans')
try:
    for chunk_number, chunk in enumerate(chunks, 1):
        # Process chunk
        pbar.update(len(chunk))
```

4. **Targeted Processing**: We only fully analyze loans in our target range:
```python
target_loans = chunk[
    (chunk['InitialApprovalAmount'] >= 5000) &
    (chunk['InitialApprovalAmount'] < 21000)
].copy()
```

This filtering step means we spend processing power only on the loans we care about - those between $5,000 and $21,000 where fraud is most common.

5. **Efficient Output Handling**: Instead of keeping all results in memory, we write them to disk as we go:
```python
if first_chunk:
    validated_loans.to_csv(self.output_file, index=False)
    first_chunk = False
else:
    validated_loans.to_csv(self.output_file, mode='a', header=False, index=False)
```

Handling large datasets also requires optimizing speed. In `analyze_patterns_in_suspicious_loans.py`, the `extract_names_optimized` function uses parallel processing with the `multiprocessing.Pool` to parse business names across multiple CPU cores, reducing computation time. Additionally, the `@lru_cache` decorator caches results of repeated name parsing, avoiding redundant work when the same names appear multiple times. These techniques ensure the system can efficiently analyze millions of loans, catching patterns like multiple businesses at one address without delays.

**For non-programmers:** Handling an 8.4GB dataset is like organizing a warehouse of records without getting buried. Here’s how we do it:  
1. Break it into smaller, workable batches (chunked processing).  
2. Focus only on the records that matter, skipping the rest (targeted processing).  
3. Save our notes as we go, so nothing piles up in memory (efficient output).  
4. Track our progress to know how much we’ve covered (progress tracking).  

This approach lets us analyze millions of loans on a reasonably fast computer without running out of memory or taking days to complete.

---

### Name Analysis: Unmasking Fraudsters Through Clever Patterns

When hunting for fraud in PPP loan data, one of the first places to look is the business name. It might sound straightforward—spot the obvious fakes like "Wakanda Enterprises" and call it a day—but the reality is far more nuanced and fascinating. The code digs deep into business names, using a mix of suspicious keyword detection, structural analysis, and pattern matching to catch fraudsters who think they’ve outsmarted the system.

#### Suspicious Keywords: The Red Flags You Can’t Ignore

The heart of the name analysis lies in a massive dictionary of suspicious patterns, defined in `self.SUSPICIOUS_PATTERNS`. Here’s a taste of what it’s looking for:

```python
self.SUSPICIOUS_PATTERNS = {
    r'wakanda': 0.95,
    r'quick\s*cash': 0.9,
    r'money\s*printer': 0.95,
    r'get\s*paid': 0.95,
    r'secure\s*bag': 0.95,
    r'covid\s*cash': 0.95,
    r'blessed\s*by\s*gov': 0.95,
    r'stonks': 0.95,
    r'diamond\s*hands': 0.95,
    # ... and over 200 more patterns!
}
```
Each pattern comes with a weight between 0 and 1, reflecting its suspiciousness. A name like "Wakanda LLC" scores a hefty 0.95—practically screaming fraud—while "Quick Cash Solutions" gets a still-concerning 0.9. These weights aren’t arbitrary; they’re tuned to flag names tied to pop culture (e.g., "wakanda," "thanos"), get-rich-quick schemes (e.g., "money printer," "fast cash"), or pandemic opportunism (e.g., "covid cash," "stimulus helper").

What’s clever here is the use of regular expressions (regex). Take `r'quick\s*cash'`: the `\s*` means "zero or more whitespace characters," so it catches sneaky variations like:
- "Quick Cash"
- "QuickCash"
- "Quick    Cash"
- "QUICK   CASH"

The code precompiles these patterns into a single regex for efficiency (see `self.compiled_suspicious_pattern`), then scans every business name in `calculate_risk_scores`. If a match is found, it adds a risk score bump (30 times the pattern’s weight) and logs a flag like "Suspicious pattern in name: quick\s*cash". Fraudsters might think they’re slick with extra spaces or capitalization tricks, but this system doesn’t miss a beat.

#### Beyond Keywords: Structural Clues in the Name

Suspicious keywords are just the start. The `validate_business_name` method digs into the structure of the name itself, looking for signs of fraud that go beyond specific words:

```python
def validate_business_name(self, name: str) -> tuple[float, List[str]]:
    flags = []
    risk_score = 0
    name_str = str(name).lower().strip()
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
```
Here’s what’s happening:
- **Missing or Invalid Names**: If the name is blank or nonsense (e.g., "n/a"), it’s an instant 1.0 risk score—game over.
- **Suspicious Patterns**: Matches against the keyword list above add to the risk score, weighted by their suspiciousness.
- **Legitimate Keywords**: Words like "consulting," "llc," or "corporation" (from `self.LEGITIMATE_KEYWORDS`) reduce the risk by 0.2 per match, rewarding names that sound professional.
- **Personal Names**: If the name looks like "John Smith" (matched via `r'^[a-z]+\s+[a-z]+$'`), it gets a 0.4 risk boost—legitimate businesses rarely use just a person’s name without qualifiers.
- **Odd Characters**: Special characters like "@" or "#" (via `r'[@#$%^&*]'`) add 0.5 to the risk, hinting at attempts to game the system.

The result? A risk score between 0 and 1, plus a list of flags explaining why. For example, "John@Cash" might score 0.9 with flags like "Personal name only" and "Suspicious characters in name."

#### Clustering and Context: The Bigger Picture

The analysis doesn’t stop at individual names. Methods like `analyze_name_patterns` and `analyze_networks` look for patterns across multiple businesses:
- **Similar Name Lengths**: If multiple businesses at the same address have names of nearly identical length (e.g., "CashKing," "LoanKing," "FastKing"), `analyze_name_patterns` adds a 10-point risk boost. This screams auto-generated fraud.
- **Generic Patterns**: `self.SUSPICIOUS_NAME_PATTERNS` flags names like "Consulting LLC" or "Holdings LLC" with low weights (0.05–0.1), but if several pop up together, the risk escalates.
- **Network Ties**: `analyze_networks` checks if businesses sharing an address have suspicious name patterns. A residential address with five "MoneyMaker"-style names? That’s a 15-point risk hike per business, plus extra for the cluster.

#### Why It Works

For non-programmers, think of this as a detective analyzing handwriting. Keywords are like obvious misspellings—easy red flags. But the structural checks are like spotting forged signatures through shaky lines or odd flourishes. By combining these layers—keywords, structure, and clustering—the code catches both the blatant fraudsters ("Stonks Inc.") and the subtler ones ("John Smith Consulting LLC" with no legit traits).

The real magic? It’s all vectorized in `calculate_risk_scores` using pandas for speed, processing thousands of names in seconds. Each match adds to a cumulative risk score, and multiple flags trigger multipliers (e.g., 1.5x for sequential loan numbers plus other issues). So, a name like "Quick Cash LLC" at a residential address with a high loan amount? That’s not just suspicious—it’s a neon sign pointing to fraud.

---

### Network Analysis: Unraveling Fraud in the Web of Connections

Catching obvious red flags in business names is satisfying, but the real detective work—and the most revealing fraud detection—happens when we dive into the networks of relationships. Sophisticated fraudsters might dodge scrutiny with polished, legitimate-sounding names, but their schemes often unravel when we trace the threads tying businesses, addresses, and lenders together. This is where the `analyze_networks` method shines, exposing patterns that lone red flags can’t reveal.

Let’s break down the core of this network analysis:

```python
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
```

#### The Network Web: Mapping the Connections

At its foundation, this code builds two critical data structures:
- **`address_to_businesses`**: A dictionary mapping each unique address (e.g., "123 Main St_Dallas_TX") to a set of businesses registered there.
- **`business_to_addresses`**: A reverse mapping, linking each business name to all addresses it’s associated with.

For non-programmers: Imagine a giant corkboard with pins and strings. Each pin is an address or business, and the strings connect them—showing which businesses share locations. This code automates that process, tracking thousands of connections in real time as it processes each loan.

#### Beyond Simple Mapping: Scoring the Suspicion

The method doesn’t stop at building the web—it analyzes it for fraud signals:
- **Shared Addresses**: If multiple businesses (two or more) use the same address, the risk ticks up. At a residential address (detected via `analyze_address_type` with a score > 0.5), it’s 15 points per business—think five "consulting" firms in a one-bedroom apartment. At a commercial address, it’s a milder 8 points per business, but still flagged as "Shared address with X other businesses."
- **Connected Networks**: It then traces deeper connections. If "ABC Corp" at one address links to other businesses at different addresses, it builds a set of `connected_businesses`. More than two connections? That’s another 5 points per business, plus a call to `analyze_name_patterns` to check if names like "CashKing" and "LoanKing" cluster suspiciously (adding even more risk if they do).
- **Batch and Sequence Checks**: Beyond the snippet above, the full method also examines lender patterns—batches of similar loans on the same day (15+ points if five or more look fishy) and sequential loan numbers (via `check_sequential_pattern`), piling on risk for organized schemes.

#### Why It’s Powerful

This isn’t just about spotting duplicates—it’s about uncovering intent. A single business at a residential address might be a small startup. Five businesses with similar names at the same apartment? That’s a fraud factory. The risk score compounds with each layer—address sharing, name patterns, lender anomalies—making it harder for fraudsters to hide behind scale or subtlety.

For the tech-curious: This runs per loan in `simple_loan_fraud_score.py`, feeding into a pandas-powered pipeline that processes chunks of data efficiently. The result? A fraud detection net that catches not just the obvious, but the cunning— Transforming messy data into clear fraud patterns.

---

#### The Residential Address Red Flag

One of the standout fraud signals in PPP loan data is when multiple businesses claim the same address—especially if it’s a residential one. The `analyze_networks` method zeroes in on this, sniffing out suspicious setups like five "consulting firms" crammed into a single apartment. Here’s how it works:

```python
# Inside analyze_networks
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
```

#### How It Spots a Residential Address

The magic happens in `analyze_address_type`, which scores an address based on clues about its nature:

```python
def analyze_address_type(self, address: str) -> float:
    address_str = str(address).lower()
    score = sum(weight for ind, weight in self.RESIDENTIAL_INDICATORS.items() if ind in address_str)
    score += sum(weight for ind, weight in self.COMMERCIAL_INDICATORS.items() if ind in address_str)
    if self.address_range.search(address_str):
        score += 0.6
    if self.address_street_end.search(address_str):
        score += 0.4
    return score

# Defined earlier in the class
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
```

Here’s the breakdown:
- **Residential Clues**: Words like "apt" (0.8), "po box" (0.9), or "apartment" (0.9) push the score up, signaling a home-like address. A high score means it’s more likely residential.
- **Commercial Clues**: Terms like "plaza" (-0.7), "office" (-0.7), or "mall" (-0.8) pull the score down, hinting at a business-friendly location.
- **Extra Hints**: An address range (e.g., "100-110 Main St") adds 0.6—common in apartments—while a street ending (e.g., "Main St") adds 0.4, refining the guess.

If the final `residential_score` exceeds 0.5, it’s flagged as residential, and the risk jumps—15 points per business at that address. Otherwise, it’s treated as commercial, with a gentler 8-point bump per business.

#### Why It Matters

For non-programmers: Think of this as a property inspector with a checklist. Residential signs (apartment numbers, PO boxes) score positive points, while commercial signs (plazas, offices) score negative. The total tells us if it’s more likely a house or a storefront. Finding multiple businesses at "123 Apt B" isn’t just odd—it’s a red flag screaming fraud, especially when each gets a 15-point penalty. A legit shared office at "456 Plaza Dr" might still raise an eyebrow (8 points each), but it’s less alarming.

This nuanced scoring handles edge cases—like mixed-use buildings—better than a simple yes/no check. It’s not just about the address; it’s about the mismatch between location and activity, making it a powerful tool in the fraud detection arsenal.

---

#### Connected Components: Uncovering Fraud Networks

Spotting suspicious addresses is a great start, but the real detective work kicks in when we trace networks of connected businesses. The `analyze_networks` method doesn’t just flag isolated oddities—it reveals webs of potential fraud, linking applications that might seem unrelated at first glance. Here’s how it digs deeper:

```python
# Inside analyze_networks
connected_businesses = set()
for addr in self.business_to_addresses[business_name]:
    connected_businesses.update(self.address_to_businesses[addr])
if len(connected_businesses) > 2:
    pattern_score, pattern_flags = self.analyze_name_patterns(connected_businesses)
    risk_score += pattern_score
    flags.extend(pattern_flags)
    risk_score += 5 * len(connected_businesses)
    flags.append(f"Connected to {len(connected_businesses)-1} other businesses")
```

#### How It Builds the Network

This snippet is doing something clever—think of it as a fraud sleuth mapping out a conspiracy:
1. **Start with a Business**: For a given `business_name` (e.g., "ABC Corp"), it grabs every address linked to it from `self.business_to_addresses`.
2. **Expand the Web**: For each of those addresses, it pulls all other businesses registered there from `self.address_to_businesses`, building a set of `connected_businesses`.
3. **Analyze the Cluster**: If this set grows beyond two businesses, it’s a red flag. The code then:
   - Calls `analyze_name_patterns` to check for suspicious naming trends (e.g., "CashKing," "LoanKing") among the group, adding a `pattern_score` if found.
   - Tacks on 5 points per connected business, amplifying the risk as the network grows.
   - Logs flags like "Connected to 4 other businesses" to highlight the web’s size.

#### The Non-Programmer’s Analogy

Imagine you’re playing Six Degrees of Kevin Bacon, but with shady businesses instead of actors. "ABC Corp" shares an apartment with "XYZ Solutions," which shares a PO Box with "QuickCash LLC." Even if "ABC Corp" and "QuickCash LLC" never directly share an address, they’re linked through this chain. The code spots that connection, treating it like a fraud family tree. If the names start looking similar—like a gang of "King"-themed businesses—the suspicion skyrockets.

#### Why It’s Useful

This isn’t just about catching duplicates at one address—it’s about exposing organized schemes. A lone business might be a fluke, but a network of five tied through multiple addresses, especially with dodgy names? That’s a coordinated hustle. The compounding risk score (5 points per business plus name pattern penalties) makes it harder for fraudsters to hide behind a web of shell companies, turning subtle links into glaring warning signs.

---

#### Temporal Patterns: The Time Dimension of Fraud

Timing can be a fraudster’s Achilles’ heel. Humans struggle to fake randomness, and when fraudsters churn out multiple PPP loan applications, they often leave behind temporal fingerprints—patterns in when and where loans are submitted. The code doesn’t have a standalone `analyze_time_patterns` method, but it weaves this logic into `calculate_risk_scores`, catching these telltale signs with precision. Let’s explore how it works:

```python
# Inside calculate_risk_scores
zip_groups = chunk.groupby(['DateApproved', 'BorrowerZip'])
for (date, zip_code), group in zip_groups:
    date = str(date)
    zip_code = str(zip_code)[:5]
    loan_info_list = group[['LoanNumber', 'BusinessType', 'InitialApprovalAmount', 'OriginatingLender']].rename(
        columns={'LoanNumber': 'loan_number', 'BusinessType': 'business_type', 
                 'InitialApprovalAmount': 'amount', 'OriginatingLender': 'lender'}
    ).to_dict('records')
    self.date_patterns[date][zip_code].extend(loan_info_list)
    self.daily_zip_counts[date][zip_code] += len(group)

# Later in the same method
for (date, zip_code), indices in zip_groups.groups.items():
    date = str(date)
    zip_code = str(zip_code)[:5]
    total_loans = len(self.date_patterns[date][zip_code])
    if total_loans >= self.ZIP_CLUSTER_THRESHOLD:
        amounts = [loan['amount'] for loan in self.date_patterns[date][zip_code]]
        business_types = [loan['business_type'] for loan in self.date_patterns[date][zip_code]]
        if max(amounts) - min(amounts) < min(amounts) * 0.1:
            flag = f'Part of cluster: {total_loans} similar loans in ZIP {zip_code} on {date}'
            for idx in indices:
                risk_scores.at[idx, 'RiskFlags'].append(flag)
            risk_scores.loc[indices, 'RiskScore'] += 10
        if len(set(business_types)) == 1:
            flag = f'Cluster of identical business types in ZIP {zip_code}'
            for idx in indices:
                risk_scores.at[idx, 'RiskFlags'].append(flag)
            risk_scores.loc[indices, 'RiskScore'] += 15
```

#### How It Spots Temporal Clues

This code builds a rich timeline of loan activity:
1. **Tracking by Date and ZIP**: For each loan, it logs the approval date and ZIP code (first five digits) in `self.date_patterns`, storing details like loan number, business type, amount, and lender. It also tallies daily counts in `self.daily_zip_counts`.
2. **Clustering Check**: If a ZIP code has at least `ZIP_CLUSTER_THRESHOLD` (set to 5) loans on a single day, it digs deeper:
   - **Similar Amounts**: If the loan amounts are suspiciously close (max - min < 10% of min), it adds 10 points and flags it as a cluster of "similar loans."
   - **Identical Business Types**: If all businesses in the cluster share the same type (e.g., all "Sole Proprietorships"), it adds 15 points, marking a "cluster of identical business types."
3. **Broader Intensity**: Beyond this snippet, the code also checks for unusual spikes—comparing daily loan counts per ZIP or lender against historical averages, adding up to 40 points for extreme bursts (e.g., 50 loans in a day vs. a 5-loan average).

#### The Non-Programmer’s Analogy

Picture a map with pins dropping as loan applications roll in. A few pins scattered across a city on different days? Normal. But if five pins land in the same neighborhood on the same day, that’s curious. If they’re all requesting nearly identical amounts—like $20,832 each—that’s suspicious. And if they’re all "consulting LLCs"? That’s not coincidence; it’s a neon sign of fraud. This code turns that map into a fraud radar, spotting clusters that real businesses wouldn’t naturally form.

#### Why It’s Revealing

Fraudsters often batch their applications, rushing to exploit a loophole before it closes. Legit businesses don’t apply in lockstep—same day, same ZIP, same story. By catching these unnatural rhythms, the code exposes coordinated schemes, piling on risk scores that make these patterns impossible to ignore.

---

#### Sequential Loan Numbers: Catching the Assembly Line

Sequential loan numbers are a dead giveaway for fraud. Legitimate applications from different businesses should have fairly random loan numbers, but when fraudsters pump out bulk submissions, those numbers often line up like cars off an assembly line. The code catches this with a mix of precision and flexibility—here’s how it works:

```python
def is_roughly_sequential(self, numbers: List[int]) -> bool:
    if len(numbers) < 2:
        return False
    numbers = np.array(numbers, dtype=np.int64)
    np.sort(numbers, kind='quicksort')  # In-place sort
    gaps = np.diff(numbers)
    return (gaps.mean() < 10) and np.all(gaps < 20)

# In calculate_risk_scores
for lender in chunk['OriginatingLender'].unique():
    recent = self.lender_loan_sequences[lender][-self.SEQUENCE_THRESHOLD:]
    if len(recent) >= self.SEQUENCE_THRESHOLD and self.is_roughly_sequential(recent):
        flag = f'Sequential loan numbers from {lender}'
        mask = chunk['OriginatingLender'] == lender
        risk_scores.loc[mask, 'RiskScore'] += 25
        for idx in chunk[mask].index:
            risk_scores.at[idx, 'RiskFlags'].append(flag)
```

#### How It Spots the Pattern

The `is_roughly_sequential` method is smarter than a simple sequence check:
1. **Extract and Sort**: It takes a list of loan numbers (stored as integers in `self.lender_loan_sequences`), converts them to a NumPy array, and sorts them fast with `quicksort`.
2. **Measure Gaps**: It calculates the differences (`gaps`) between consecutive numbers using `np.diff`.
3. **Flexible Detection**: It flags a sequence if the average gap is under 10 and no gap exceeds 20. This catches "roughly" sequential runs—like 157, 159, 162—not just perfect ones like 157, 158, 159.
4. **Trigger in Context**: In `calculate_risk_scores`, it checks the last `SEQUENCE_THRESHOLD` (set to 5) loan numbers per lender. If five or more are roughly sequential, it adds 25 points and flags it.

---

#### Beyond Sequences: Batch Analysis

The code doesn’t stop there. In `analyze_networks`, it also looks at lender batches for suspicious timing and similarity:

```python
# In analyze_networks
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
```

Here, if five or more loans from the same lender on the same day have amounts within 10% of each other, it adds 15 points (plus extra from name patterns), flagging a "suspicious batch."

#### The Non-Programmer’s Analogy

Think of loan numbers like deli counter tickets. Real customers grab tickets at random—57, 72, 89. But if you see 157, 159, 162 from different "businesses," it’s like someone’s hogging the machine. Add in nearly identical loan amounts—like five $20,832 tickets—and it’s not a deli; it’s a fraud factory. The code’s like a sharp-eyed clerk spotting the cheaters, even if they skip a number or two.

#### Why It’s Damning

Fraudsters often automate or rush their applications, leaving these orderly traces—sequential numbers or tight batches—that legit businesses don’t mimic. By catching both the sequence (25 points) and the batch similarity (15+ points), the code nails bulk fraud, turning an innocent-looking stream of loans into a glaring red flag.

---

#### Lender Batch Analysis: Spotting the Fraud Factory

Sequential loan numbers can snag some fraudsters, but the craftier ones might dodge that trap by scrambling their numbers. That’s where lender batch analysis steps in—it sniffs out suspicious patterns even when loan numbers seem random, catching fraud rings by their telltale habits. The `analyze_networks` method does this with finesse—here’s how:

```python
# Inside analyze_networks
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
```

#### How It Detects the Factory Line

This code zeroes in on a subtle but damning clue—batches of loans from the same lender, same location, same day, with eerily similar amounts:
1. **Batch Key**: It crafts a unique `lender_key` by combining the lender’s name, location ID, and approval date, grouping loans into daily batches per lender branch.
2. **Threshold Check**: If a batch hits 5 or more loans, it digs deeper into the amounts.
3. **Similarity Test**: If the loan amounts are tight—max minus min is less than 10% of the min (e.g., all around $20,000)—it flags it.
4. **Extra Layer**: It then runs `analyze_name_patterns` on the batch’s business names, adding a `pattern_score` if they look fishy (e.g., "CashKing," "LoanKing"). The base risk jumps 15 points, plus whatever the name check tacks on.

#### The Non-Programmer’s Analogy

Imagine you’re a bank teller, and five people stroll in on the same day, each asking for $20,832 from the same branch. Odd, right? Now picture them doing it daily, with names like "QuickCash LLC" and "FastCash Inc." It’s not a coincidence—it’s a fraud assembly line. Real businesses don’t line up like that with near-identical requests; this code’s like a hawk-eyed teller who spots the pattern, even if the ticket numbers aren’t in order.

#### Why It Works

Fraudsters might dodge sequential numbers, mix up addresses, or tweak names, but they’re awful at faking randomness. When mass-producing applications, they often stick to "safe" amounts—like $20,832—that slip through approval cracks. This check catches that habit. And when paired with suspicious name patterns (via `analyze_name_patterns`), the risk score spikes—15 points base, plus more for dodgy names—separating legit batches (like a franchise’s bulk apps) from coordinated scams. It’s a net for the clever ones who think they’ve outsmarted simpler traps.

---

#### The Early Warning System: High-Risk Lenders

One of the sharper tools in our fraud detection kit is tracking lenders with a history of processing dodgy loans. It’s a bit controversial—lenders aren’t thrilled to be on a watchlist—but it creates a powerful early warning system that flags emerging fraud patterns. Here’s the setup:

```python
self.HIGH_RISK_LENDERS = {
    'Celtic Bank Corporation': 0.8,
    'Cross River Bank': 0.7,
    'Customers Bank': 0.7,
    'Kabbage, Inc.': 0.8,
    'Capital Plus Financial': 0.9,
    'Harvest Small Business Finance': 0.8
}
```

Each lender gets a risk weight between 0 and 1, reflecting their track record. In `calculate_risk_scores`, if a loan comes from one of these high-risk players and already has other red flags, the score gets a boost:

```python
# In calculate_risk_scores
lender_has_flags = risk_scores['RiskFlags'].apply(len) > 0
high_risk_lender_mask = chunk['OriginatingLender'].isin(self.HIGH_RISK_LENDERS) & lender_has_flags
lender_risk_scores = chunk['OriginatingLender'].map(lambda x: 15 * self.HIGH_RISK_LENDERS.get(x, 0)).astype(float).fillna(0.0)
risk_scores.loc[high_risk_lender_mask, 'RiskScore'] += lender_risk_scores[high_risk_lender_mask]
risk_scores.loc[high_risk_lender_mask, 'RiskFlags'] = risk_scores.loc[high_risk_lender_mask, 'RiskFlags'].apply(
    lambda x: x + ['High-risk lender']
)
```

#### How It Works

- **Weighted Risk**: A lender like "Capital Plus Financial" (0.9) adds 13.5 points (15 * 0.9) to the risk score if other flags—like suspicious names or addresses—are present.
- **Combo Trigger**: It only kicks in when there’s already a flag, avoiding unfair penalties for clean loans from these lenders.

For non-programmers: Picture a bank with security guards. Some guards have a rep for letting sketchy characters slip by. If you spot someone shady strolling past one of those guards, you double-check them. This system does the same—high-risk lenders raise the stakes when other clues are in play.

---

#### Time-Based Clustering: Catching the Mass Producers

Time is a fraudster’s worst enemy—and our best ally. When churning out multiple PPP loan applications, fraud rings often leave behind detectable time-based clusters that legitimate businesses don’t mimic. The code harnesses this in `calculate_risk_scores` and `analyze_networks`, spotting mass production patterns with surgical precision. Here’s how it tracks the clock:

```python
# In calculate_risk_scores
zip_groups = chunk.groupby(['DateApproved', 'BorrowerZip'])
for (date, zip_code), group in zip_groups:
    date = str(date)
    zip_code = str(zip_code)[:5]
    loan_info_list = group[['LoanNumber', 'BusinessType', 'InitialApprovalAmount', 'OriginatingLender']].rename(
        columns={'LoanNumber': 'loan_number', 'BusinessType': 'business_type', 
                 'InitialApprovalAmount': 'amount', 'OriginatingLender': 'lender'}
    ).to_dict('records')
    self.date_patterns[date][zip_code].extend(loan_info_list)
    self.daily_zip_counts[date][zip_code] += len(group)

for (date, zip_code), indices in zip_groups.groups.items():
    date = str(date)
    zip_code = str(zip_code)[:5]
    total_loans = len(self.date_patterns[date][zip_code])
    if total_loans >= self.ZIP_CLUSTER_THRESHOLD:
        amounts = [loan['amount'] for loan in self.date_patterns[date][zip_code]]
        business_types = [loan['business_type'] for loan in self.date_patterns[date][zip_code]]
        if max(amounts) - min(amounts) < min(amounts) * 0.1:
            flag = f'Part of cluster: {total_loans} similar loans in ZIP {zip_code} on {date}'
            risk_scores.loc[indices, 'RiskScore'] += 10
            for idx in indices:
                risk_scores.at[idx, 'RiskFlags'].append(flag)
        if len(set(business_types)) == 1:
            flag = f'Cluster of identical business types in ZIP {zip_code}'
            risk_scores.loc[indices, 'RiskScore'] += 15
            for idx in indices:
                risk_scores.at[idx, 'RiskFlags'].append(flag)
```

#### ZIP-Based Clustering: The Geographic Pulse

This code maps loans by date and ZIP code, hunting for suspicious bursts:
- **Tracking**: It logs each loan’s date, ZIP (first 5 digits), amount, and business type in `self.date_patterns`, building a daily geographic timeline.
- **Threshold**: If a ZIP hits `ZIP_CLUSTER_THRESHOLD` (5) loans in a day, it checks:
  - **Tight Amounts**: Loans within 10% of each other (e.g., all ~$20,000) add 10 points.
  - **Same Type**: All identical business types (e.g., "Sole Proprietorships") add 15 points.

#### Lender Batches: The Factory Floor

Then, in `analyze_networks`, it zooms into lender-specific batches:

```python
# In analyze_networks
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
```

Here, it keys loans by lender, location, and date:
- **Batch Check**: Five or more loans on the same day from one lender branch trigger scrutiny.
- **Similarity**: Amounts within 10% score 15 points, plus a `pattern_score` if names (e.g., "CashKing," "LoanKing") look fishy.

#### The Combo Punch: Sequential Boost

The real kicker comes in `calculate_risk_scores` when clustering meets sequential loan numbers:

```python
# In calculate_risk_scores
seq_mask = risk_scores['RiskFlags'].str.contains("Sequential loan numbers") & (risk_scores['RiskFlags'].str.count(';') > 1)
risk_scores.loc[seq_mask, 'RiskScore'] *= 1.5
```

If a loan’s flags include "Sequential loan numbers" and at least one other red flag, the risk score jumps by 50%—a multiplier that screams automation.

#### Why It Works

Fraudsters love efficiency—same lender, same branch, same day, same script. Legit businesses? They’re messier—different needs, staggered timing. A real district might see a workshop spike, but amounts vary (rent, staff, utilities differ). Fraud rings churn out cookie-cutter apps—five $20,832 loans in one ZIP or lender batch, all "consulting LLCs." Add sequential numbers, and it’s a dead-end street.

#### The Non-Programmer’s Analogy

Imagine a counterfeiter printing 10 identical $20 bills in one afternoon—no real bank churns out cash that fast and uniform. Now picture a ZIP code or lender spitting out five $20,832 loans, same type, same day. It’s not a business district; it’s a fraud press. This code’s like a detective spotting the ink still wet, especially when the serial numbers line up too neatly.

---

### The Devil in the Demographics

Fraudulent loan applications often reveal subtle clues in how they handle demographic information. Legitimate applicants tend to provide varied responses, while fraudsters may leave fields blank or use generic placeholders. Our system checks for this pattern with precision:

```python
demo_fields = ['Race', 'Gender', 'Ethnicity']
missing_all_demo = chunk[demo_fields].apply(lambda x: x.str.lower().isin(self.UNANSWERED_SET)).all(axis=1) & lender_has_flags
risk_scores.loc[missing_all_demo, 'RiskScore'] += 10
risk_scores.loc[missing_all_demo, 'RiskFlags'] = risk_scores.loc[missing_all_demo, 'RiskFlags'].apply(
    lambda x: x + ['Missing all demographics']
)
```

Here, `self.UNANSWERED_SET` includes terms like `'unanswered'` and `'unknown'`. The code doesn’t just flag missing demographics—it requires other suspicious indicators (`lender_has_flags`) to be present. This ensures we’re not penalizing incomplete data alone but spotting it as part of a broader fraud pattern.

**For non-programmers:** Imagine reviewing a stack of forms where some applicants skip a question or two, but others leave every personal detail as “N/A” *and* show other red flags—like dubious addresses or lenders. Those are the ones we scrutinize.

---

### Risk Score Aggregation: Putting It All Together

Our fraud detection hinges on a weighted scoring system that balances the importance and reliability of different indicators. Here’s how we define the weights:

```python
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
```
These weights guide the `calculate_risk_scores` method, though scores are applied dynamically based on conditions, not directly multiplied by these weights. For instance, `'jobs'` gets a high weight (0.20) because job counts are harder to fake convincingly, while `'demographics'` (0.05) is less decisive alone. The system also uses interaction rules to amplify scores when multiple red flags align, like:

```python
INTERACTION_RULES = {
    ("Missing all demographics", "High-risk business type"): 1.05,
    # ... other rules ...
}
```

**For non-programmers:** Think of this as a chef blending ingredients—some (like jobs) add a strong flavor, others (like demographics) are subtle unless paired with something suspicious. The interaction rules are like tasting a dish and realizing two odd flavors together signal trouble.

---

### The Business Type Legitimacy Scale

We categorize business types by their fraud risk, based on extensive data analysis:

```python
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
```

The scoring logic ties these to other factors:

```python
high_risk_bt_mask = chunk['BusinessType'].isin(self.HIGH_RISK_BUSINESS_TYPES) & lender_has_flags
bt_risk_scores = chunk['BusinessType'].map(lambda x: 15 * self.HIGH_RISK_BUSINESS_TYPES.get(x, 0)).astype(float).fillna(0.0)
risk_scores.loc[high_risk_bt_mask, 'RiskScore'] += bt_risk_scores[high_risk_bt_mask]
risk_scores.loc[high_risk_bt_mask, 'RiskFlags'] = risk_scores.loc[high_risk_bt_mask, 'RiskFlags'].apply(
    lambda x: x + ['High-risk business type']
)
```

A “Sole Proprietorship” only triggers a score (e.g., 15 * 0.85 = 12.75) if other flags are present, reflecting context-driven suspicion.

**For non-programmers:** It’s like judging a stranger’s story. Saying “I work alone” isn’t odd, but paired with a questionable address or loan amount, it raises eyebrows.

---

### Secondary Analysis System: Statistical Pattern Detection

The secondary analysis system, implemented in `analyze_patterns_in_suspicious_loans.py`, is designed to identify statistical patterns within Paycheck Protection Program (PPP) loan data that may indicate fraudulent activity. Unlike a primary analysis focused on individual loan assessments, this system examines aggregate trends across datasets, leveraging advanced data processing and machine learning techniques to enhance detection capabilities.

#### Data Loading and Preparation

The analysis begins with efficient data ingestion:

```python
class SuspiciousLoanAnalyzer:
    def __init__(self, suspicious_file: str, full_data_file: str):
        self.suspicious_file = suspicious_file
        self.full_data_file = full_data_file
        self.sus_data = None
        self.full_data = None
        self.naics_lookup = self._load_naics_codes()

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        print("Loading suspicious loans data...")
        sus_cols = [
            "LoanNumber", "BorrowerName", "RiskScore", "RiskFlags", "BorrowerAddress", 
            "BorrowerCity", "BorrowerState", "BorrowerZip", "Latitude", "Longitude", 
            "Census Tract Code", "OriginatingLender", "InitialApprovalAmount", 
            "BusinessType", "Race", "Gender", "Ethnicity", "NAICSCode", 
            "JobsReported", "HubzoneIndicator", "LMIIndicator", "NonProfit"
        ]
        sus_dtypes = {
            "LoanNumber": str,
            "BorrowerName": str,
            "RiskScore": "float32",
            "InitialApprovalAmount": "float32",
            "JobsReported": "Int32",
            # Additional type specifications omitted for brevity
        }
        sus = pd.read_csv(
            self.suspicious_file,
            engine='pyarrow',
            dtype=sus_dtypes,
            usecols=sus_cols
        )
        print("Loading full loan dataset with Dask...")
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
        full = full_dd.compute(scheduler='processes', num_workers=32)
```

The system employs `pyarrow` to load the suspicious loans dataset, optimizing memory usage with specific data types (e.g., `float32` for numerical columns). For the full dataset, Dask facilitates parallel processing of large files, filtering loans to the $5,000–$22,000 range—a segment identified as prone to irregularities. Selected columns, such as `BorrowerAddress` and `NAICSCode`, are prioritized based on their statistical relevance to pattern detection.

#### Address-Based Feature Engineering

Address analysis is a critical component of the system, utilizing binary indicators to classify locations:

```python
residential_indicators = {'apt', 'unit', 'suite', '#', 'po box', 'residence', 'residential', 'apartment', 'room', 'floor'}
commercial_indicators = {'plaza', 'building', 'tower', 'office', 'complex', 'center', 'mall', 'commercial', 'industrial', 'park'}
address_str = df['BorrowerAddress'].astype(str).str.lower()
df['HasResidentialIndicator'] = address_str.apply(lambda x: any(ind in x for ind in residential_indicators)).astype('uint8')
df['HasCommercialIndicator'] = address_str.apply(lambda x: any(ind in x for ind in commercial_indicators)).astype('uint8')
```

This method scans each address for predefined residential or commercial keywords, assigning a binary value (1 or 0) to `HasResidentialIndicator` and `HasCommercialIndicator`. While a residential indicator alone may not signify fraud—given many legitimate businesses operate from homes—it becomes significant when combined with other factors, such as multiple businesses registered at a single address.

#### Statistical Validation with Chi-Square Testing

To assess the significance of categorical variables (e.g., lenders or geographic locations), the system employs the chi-square test:

```python
def analyze_categorical_patterns(
    self,
    sus: pd.DataFrame,
    full: pd.DataFrame,
    column: str,
    title: str,
    min_occurrences: int = 5,
    high_threshold_occurrences: int = 25,
) -> None:
    sus[column] = sus[column].fillna('Unknown')
    full[column] = full[column].fillna('Unknown')
    s_counts = sus[column].value_counts()
    f_counts = full[column].value_counts()
    contingency = []
    for category in sorted(set(s_counts.index) | set(f_counts.index)):
        suspicious = s_counts.get(category, 0)
        total = f_counts.get(category, 0)
        non_suspicious = max(0, total - suspicious)
        if suspicious > 0 or non_suspicious > 0:
            contingency.append([suspicious, non_suspicious])
    cont_table = np.array(contingency)
    if cont_table.size > 0 and cont_table.shape[0] > 1:
        _, p_chi2, _, _ = stats.chi2_contingency(cont_table)
```        

This function constructs a contingency table comparing the frequency of categories in suspicious versus non-suspicious loans, then applies the chi-square test. A low p-value indicates a statistically significant deviation from expected distributions. The analysis runs with dual thresholds (5 and 25 occurrences) to capture both emerging and prominent patterns.

#### Representation Ratio Analysis

The representation ratio quantifies overrepresentation of features in suspicious loans:

```python
def calculate_representation_ratio(
    self,
    suspicious_counts: pd.Series,
    full_counts: pd.Series,
    min_occurrences: int = 5,
) -> pd.DataFrame:
    suspicious_counts = suspicious_counts.clip(lower=0)
    full_counts = full_counts.clip(lower=0)
    sus_total = suspicious_counts.sum()
    sus_pct = suspicious_counts / sus_total if sus_total > 0 else 0
    full_total = full_counts.sum()
    full_pct = full_counts / full_total if full_total > 0 else 0
    ratios = pd.Series(
        {idx: self.safe_divide(
            sus_pct.get(idx, 0),
            full_pct.get(idx, 0),
            default=0.0
        ) for idx in set(suspicious_counts.index) | set(full_counts.index)}
    )
```

This method computes the proportion of a category (e.g., a specific NAICS code) within suspicious loans relative to its proportion in the full dataset. A ratio significantly greater than 1 highlights potential areas of concern, with results filtered to ensure a minimum of five occurrences for reliability.

#### Numerical Stability with Safe Division

To maintain computational integrity, a `safe_divide` function handles edge cases:

```python
def safe_divide(self, a: float, b: float, default: float = 0.0) -> float:
    try:
        if b == 0:
            return default
        result = a / b
        if np.isinf(result) or np.isnan(result):
            return default
        return result
    except Exception:
        return default
```        

This ensures robustness against division by zero, infinite values, or NaN results—common issues when processing large, heterogeneous datasets.

#### Additional Statistical Tests for Numerical Features

For numerical variables such as loan amounts or name lengths, the system employs multiple statistical tests:

```python
name_length_tests = [
    ("t-test", stats.ttest_ind),
    ("KS test", stats.ks_2samp),
    ("Mann-Whitney U", lambda x, y: stats.mannwhitneyu(x, y, alternative="two-sided"))
]
results = Parallel(n_jobs=3)(
    delayed(run_stat_test)(test_name, test_func, sus["NameLength"].dropna(), full["NameLength"].dropna())
    for test_name, test_func in name_length_tests
)
```
These tests—the t-test for comparing means, the Kolmogorov-Smirnov test for distribution differences, and the Mann-Whitney U test for non-parametric comparisons—are executed in parallel to assess significance efficiently across datasets.

#### Machine Learning with XGBoost

The system integrates an XGBoost-based analysis for advanced pattern recognition:

```python
class XGBoostAnalyzer:
    def analyze_with_xgboost(self, sus: pd.DataFrame, full: pd.DataFrame, n_iter: int = 10, min_instances: int = 250):
        full_prepared = self.prepare_enhanced_features(full.copy(), min_instances=min_instances)
        full_prepared["Flagged"] = full_prepared["LoanNumber"].isin(sus["LoanNumber"].astype(str)).astype('uint8')
        X = full_prepared[[col for col in full_prepared.columns if col not in ["Flagged", "LoanNumber"]]]
        y = full_prepared["Flagged"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        xgb_clf = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='aucpr',
            random_state=42,
            tree_method='hist',
            n_jobs=12
        )
        random_search = RandomizedSearchCV(xgb_clf, param_distributions=param_grid, n_iter=n_iter, scoring='average_precision')
        random_search.fit(X_train, y_train)
        self.model = xgb.XGBClassifier(**random_search.best_params_)
        self.model.fit(X_train, y_train)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.3f}")
```

This component preprocesses features (e.g., address indicators, demographic data) and trains an XGBoost classifier to predict loan suspiciousness. Hyperparameters are optimized via `RandomizedSearchCV`, and performance is evaluated using metrics like ROC-AUC and average precision. SHAP values further elucidate feature contributions, enhancing interpretability.

#### Conclusion

The secondary analysis system combines efficient data handling, statistical testing, and machine learning to detect patterns indicative of fraud in PPP loan data. From chi-square tests identifying categorical anomalies to XGBoost uncovering complex interactions, it provides a comprehensive framework for analysis, grounded in robust computational techniques.

---

Thanks for reading this blog post! I hope you enjoyed it. If you did, I would really appreciate it if you checked out my web app, [FixMyDocuments.com](https://fixmydocuments.com/). It's a very useful service that leverages powerful AI tools to transform your documents from poorly formatted or scanned PDFs into beautiful, markdown formatted versions that can be easily edited and shared. Once you have processed a document, you can generate all sorts of derived documents from it with a single click, including:

* Real interactive multiple choice quizzes you can take and get graded on (and share with anyone using a publicly accessible custom hosted URL).
* Anki flashcards for studying, with a slick, interactive interface (and which you can also share with others).
* A slick HTML presentation slide deck based on your document, or a PDF presentation formatted using LaTeX.
* A really detailed and penetrating executive summary of your document.
* Comprehensive "mindmap" diagrams and outlines that explore your document thoroughly.
* Readability analysis and grade level versions of your original document.
* Lesson plans generated from your document, where you can choose the level of the target audience.

It's useful for teachers, tutors, business people, and more. When you sign up using a Google account, you get enough free credits that let you process several documents. Give it a try!
