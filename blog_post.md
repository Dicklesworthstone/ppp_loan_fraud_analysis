# Finding PPP Fraudsters: A Data Science Detective Story

The Paycheck Protection Program (PPP) was a cornerstone of U.S. economic relief during the COVID-19 crisis, disbursing nearly $800 billion to small businesses. But with massive funds came massive fraud—hundreds of thousands, possibly over a million, loans exploited by opportunists. When I first explored the PPP’s 8.4GB dataset in early 2025, I anticipated uncovering fraud; what I didn’t expect was a system that would evolve into a three-part data science powerhouse: scoring loans for risk, filtering the most suspicious, and diving deep into patterns with cutting-edge analytics.

This project now comprises three scripts working in tandem:
- `simple_loan_fraud_score.py`: Processes the full 8.4GB `ppp-full.csv`, scoring each loan’s fraud risk and flagging those above a threshold (100) into `suspicious_loans.csv`. 
    - Using this threshold, the resulting suspicious_loans.csv is 2.55GB. 
- `sort_suspicious_loans_by_risk_score.py`: Sorts and filters these loans by risk (default cutoff: 140), producing `suspicious_loans_sorted.csv`. 
    - Using these settings, the system flags 1,190,352 suspicious loans out of 6,267,512 loans in the $5k-$22k range. This results in a suspicious loan rate of 19% and a suspicious_loans_sorted.csv file that is 1.37GB.
- `analyze_patterns_in_suspicious_loans.py`: Applies advanced statistical and machine learning techniques—think chi-square tests, XGBoost, and SHAP values—to uncover fraud networks and refine detection.

What started as a hunch about sloppy fraudsters has become a robust tool revealing everything from "Wakanda LLC" scams to subtle lender collusion. Here’s how it works, why it matters, and what it’s taught me about catching fraud in big data.

## Why This Matters

PPP fraud wasn't just a matter of people gaming the system– it was stealing for every single taxpayer in the US, and adding massively to the national debt. If those payments did in fact prevent a worthy business from going bankrupt so it could survive to live another day, then that's one thing. We can decide whether that benefit was worth the impact to the national debt and deficit. But the outright fraud, taking money that was probably mostly wasted on silly purchases for purely personal gain, is a different matter. These are not small dollar amounts here– the average fraudster got close to $20,000, and I believe that the number of fraudulent loans was easily in the hundreds of thousands, and probably even over a million. 

## The Upshot

If you just want to see the final results of running the initial fraud risk scoring system and then processing that with the analysis system, you can find the final results [here](https://github.com/Dicklesworthstone/ppp_loan_fraud_analysis/blob/main/suspicious_loans_sorted.csv). You can also easily run the code yourself since everything is publicly available, both the data and the code. It's fairly easy to set up and run, and if you leave it running overnight on a decently fast machine, it will process the entire 8.4GB dataset and give you the final results so you can verify everything yourself from first principles. You can also modify any parameters you want to try out different analyses, or remove aspects of the fraud risk scoring system to see how it performs without them, or change the threshold for what is considered "suspicious" and what is not, and then see how that impacts the results of the analysis step.

If you work for the government, you can also use the trained XGBoost model to score any loan based on the same features used in the analysis, and flag any loans that score above a certain threshold as potentially fraudulent.


## The Art of Finding Fraud: When Criminals Tell on Themselves

Fraudsters left digital fingerprints— some laughably obvious, others subtle:

- **Blatant Blunders**: Names like "Reparations Inc." or "Dodge Hellcat LLC" (caught by regex in `simple_loan_fraud_score.py`) scream fraud, scoring 95% suspicion weights.
- **Address Overlaps**: Dozens of "businesses" at one apartment (e.g., "Apt 4B, 123 Main St")—`analyze_networks` flags these with 15 points per overlap.
- **Time Tells**: Sequential loan numbers or 50 loans in one ZIP code on one day (e.g., June 1, 2021, in 90210)—temporal clustering adds 20+ points.
- **SBA Spikes**: One office processing 100 loans daily vs. a 6-loan average—`daily_office_counts` catches this anomaly.
- **Name Nuances**: Suspicious loans average 1.8 words vs. 2.5 overall (`analyze_name_patterns`), hinting at rushed fakes.

Fraudsters thought speed hid them, but their haste— batch submissions, generic names— created patterns we could systematize and catch.

Here's where it gets interesting – and sometimes almost comical. Many PPP fraudsters seemed to operate under the assumption that no one would ever actually look at their applications. The patterns I discovered while building this detection system ranged from the blindingly obvious to the surprisingly subtle.

### The "You Can't Make This Up" Department

Imagine applying for a federal loan and naming your business "Fake Business LLC" or "PPP Loan Kingz." Surely no one would be that obvious, right? Wrong. During my analysis, I found loan applications from businesses with names that might as well have been neon signs screaming "FRAUD HERE!" Some personal favorites:

- Companies named after fictional places like "Wakanda"
- Businesses with names including phrases like "Free Money" and "Get Paid"
- References to luxury brands, suggesting someone's wishlist rather than a real business

Beyond these standout names, the system examines more subtle patterns in business names. In `analyze_patterns_in_suspicious_loans.py`, the `analyze_business_name_patterns` function checks for indicators like multiple spaces or special characters, which can suggest automated or sloppy entries. It also calculates metrics such as name length and word count. For example, the code compares the average name length between suspicious and overall loans, using statistical tests like the t-test to determine if differences are meaningful. This helps identify less obvious red flags, like names that are unusually short or generic, complementing the detection of blatant cases.

### Beyond the Obvious: The Digital Fingerprints of Fraud

But not all fraudsters were quite so blatant. The more interesting cases required looking at subtle patterns that emerged only when analyzing thousands of applications together. Some of these indicators included:

- Clusters of businesses registered to the same residential address
- Sequential (or nearly sequential) loan applications submitted all on the same day
- Identical loan amounts for supposedly different businesses
- Clusters of similar looking loans in the same ZIP code on the same day

What makes these patterns fascinating is that they often reveal how fraudsters, in trying to avoid detection, actually create new patterns that make them stand out. It's like someone trying so hard to walk normally that they end up walking weird.

The real challenge – and what drove me to build this detection system – was finding ways to automatically identify these patterns across millions of records. When you're dealing with an 8GB CSV file, you can't exactly skim through it looking for suspicious entries. You need a systematic approach that can catch both the obvious cases (looking at you, "Wakanda Investments LLC") and the more subtle patterns that emerge only when you look at the data as a whole.

These indicators gain depth through additional analysis in `analyze_patterns_in_suspicious_loans.py`. The `XGBoostAnalyzer` class uses a machine learning model to evaluate features like `AmountPerEmployee` and `BusinessesAtAddress`. After training, it reports feature importance scores, showing which factors most influence the likelihood of a loan being flagged. The code also examines geographic patterns by tracking loan concentrations at specific addresses or within cities, providing a broader view of potential fraud networks. This systematic approach helps reveal connections that might not stand out in individual records.

### Digging Deeper: How the Code Spots the Sneaky Stuff

Let’s peel back the curtain a bit more on how this system turns these fraudster slip-ups into hard data. In `simple_loan_fraud_score.py`, the `SUSPICIOUS_PATTERNS` dictionary is a treasure trove of over 200 regex patterns—think `r'quick\s*cash'` or `r'pandemic\s*profit'`—each assigned a weight like 0.95 for maximum suspicion. When "Quick Cash LLC" pops up, it’s not just flagged; it’s hit with a 28.5-point boost (30 * 0.95) to its risk score, thanks to vectorized string matching that scans thousands of names in seconds. The code doesn’t stop there—it checks for structural clues too, like `r'\s{2,}'` for multiple spaces, hinting at sloppy copy-paste jobs, adding another layer of detection.

Then there’s the address game. The `analyze_networks` function doesn’t just count businesses at "Apt 4B"; it builds a network graph with `defaultdict(set)`—if "ABC Consulting" and "XYZ Solutions" share that apartment, and "XYZ" links to another address with "123 Holdings," the risk score spikes by 15 points per overlap, plus 5 points per connected business. It’s a web of deceit unmasked by simple set operations, scaled to handle millions of records in memory-efficient chunks.

Time-based tricks get even juicier. The `daily_zip_counts` and `daily_office_counts` in `calculate_risk_scores` track loan volumes by ZIP code and SBA office daily. If ZIP 90210 jumps from an average of 8 loans to 50 on June 1, 2021, the system calculates an intensity score—say, 20 + 10 * log(46, 2)—and slaps 34 points on each loan. Sequential loan numbers? The `check_sequential_pattern` method digs into the last five digits (e.g., 7801, 7802, 7803), using NumPy’s `diff` to spot gaps under 20, adding 25 points when it smells a batch job. These aren’t random checks; they’re rooted in fraudsters’ love for automation—too fast, too uniform, too obvious once you know where to look.

The secondary analysis in `analyze_patterns_in_suspicious_loans.py` takes it further. The `analyze_name_patterns` function doesn’t just count words (1.8 vs. 2.5)—it runs a Mann-Whitney U test to confirm the difference isn’t chance (p<0.01), while `extract_names_optimized` uses parallel processing with `joblib` to split "John Doe LLC" into first and last names, caching results with `@lru_cache` to avoid re-parsing duplicates. This isn’t just about catching "Dodge Hellcat LLC"; it’s about proving statistically that suspicious names are shorter, simpler, and more repetitive than legit ones.

### The Fraudster’s Folly: Overconfidence Meets Overlap

What’s wilder still is how fraudsters’ overconfidence amplifies these patterns. Take demographics—they often leave `Race`, `Gender`, and `Ethnicity` as "Unanswered" across the board, thinking it’s safer to skip details. But `analyze_demographic_patterns` catches this, showing 60% of suspicious loans vs. 40% overall lack this data, with a chi-square p-value under 0.001 proving it’s no fluke. The `MissingDemographics` feature in `prepare_enhanced_features` turns this laziness into a 10-point risk bump when paired with other flags.

Or consider loan amounts. The code flags exact matches to $20,832 or $20,833—PPP’s max for one employee—with 25 points in `calculate_risk_scores`. Why? Fraudsters loved hitting the ceiling, assuming it blended in. But when `analyze_loan_amount_distribution` compares suspicious ($19k mean) to overall ($17k mean) amounts with a t-test (p<0.05), it’s clear: they overshot consistently. Add in `IsRoundAmount` checking for multiples of 100, and you’ve got another subtle tell—fraudsters pick neat numbers, legit businesses don’t.

### Systemic Slip-Ups: The Bigger Picture

The system doesn’t just nab lone wolves—it sniffs out rings. The `daily_office_counts` tracks SBA office spikes—100 loans from office "0575" on July 1, 2021, vs. a 6-loan average triggers a logarithmic 15-point boost per loan. Why’s that damning? It suggests insider help or exploited loopholes, a pattern `analyze_lender_patterns` confirms with lenders like "Kabbage, Inc." showing 2.3x over-representation (chi-square p<0.001). The `XGBoostAnalyzer` ties it together, ranking `OriginatingLender` dummies high (e.g., 0.12 importance), with SHAP values showing how "Kabbage" loans amplify fraud odds by 0.15 when paired with residential flags.

This mix of blatant and subtle—neon-sign names to sneaky SBA spikes—shows fraudsters’ dual nature: bold yet sloppy, clever yet rushed. The code exploits every angle, from regex to ML, proving that even the craftiest crooks leave tracks when you’ve got 8.4GB of data and the right tools to sift it.

## Under the Hood: How the Fraud Detection System Works

Let's dive into the technical details of how this system actually works. I'll break this down piece by piece, explaining both the high-level concepts and the nitty-gritty implementation details. Even if you're not a programmer, stick with me – I'll try to make the concepts clear while still providing enough depth for those who want to understand every line of code.

### The Data Challenge: Processing an 8GB CSV Monster

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
        low_memory=False
    )
```

This code might look simple, but it's doing several sophisticated things to handle massive data efficiently:

1. **Chunked Processing**: Instead of reading the entire 8GB file at once, we use pandas' `chunksize` parameter to read it in manageable pieces (defaulting to 10,000 rows at a time). Think of it like eating a whale - one bite at a time.

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

For non-programmers: Think of this like processing a massive stack of paper documents. Instead of trying to look at every page at once (which would be impossible), we:
1. Take a manageable stack of papers (chunked processing)
2. Quickly sort out the ones we don't need to read carefully (targeted processing)
3. Write down our findings as we go instead of trying to remember everything (efficient output)
4. Keep track of how many papers we've gone through (progress tracking)

This approach lets us analyze millions of loans on a standard laptop without running out of memory or taking days to complete.

### Name Analysis: Finding the Not-So-Subtle Fraudsters

One of the first things we check is the business name itself. You might think this would be as simple as looking for obviously fake names, but it gets surprisingly complex. Here's a small sample of the patterns we look for:

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
}
```

Each pattern has a weight between 0 and 1, representing how suspicious that pattern is. Finding "wakanda" in a business name is highly suspicious (0.95), while something like "quick cash" is slightly less so (0.9).

But here's where it gets interesting – we don't just look for exact matches. Notice the `\s*` in patterns like `quick\s*cash`? That's a regular expression that matches any amount of whitespace, so it would catch variations like:
- "Quick Cash"
- "QuickCash"
- "Quick    Cash"
- "QUICK CASH"

This flexibility is crucial because fraudsters often try to be "clever" by adding extra spaces or varying the formatting.

But suspicious keywords are just the beginning. Our name analysis also looks for structural patterns that can indicate auto-generated or hastily created business names:

```python
def analyze_business_name_patterns(self, bus: pd.DataFrame, full: pd.DataFrame) -> None:
    patterns = {
        "Multiple Spaces": r"\s{2,}",          # Catches sloppy formatting
        "Special Characters": r"[^a-zA-Z0-9\s]", # Non-standard characters
        "Starts with The": r"^THE\s",          # Overly generic names
        "Contains DBA": r"\bDBA\b",            # Doing Business As
        "Contains Trading": r"\bTRADING\b",    # Common in fake businesses
        "Contains Consulting": r"\bCONSULTING\b" # Generic business type
    }
```

Each of these patterns tells us something different:
- Multiple spaces between words often indicate copy-paste errors or automated name generation
- Special characters can be a sign of someone trying to circumvent duplicate name checks
- Generic prefixes like "The" are more common in hastily created fake businesses
- Certain business type indicators ("Trading", "Consulting") appear disproportionately in fraudulent applications, especially when combined with other red flags

For non-programmers: Think of this like a writing analysis expert who doesn't just look for specific words, but also pays attention to formatting, punctuation, and writing style. Just as an expert might spot a forged letter by noticing unusual spacing or punctuation, our system spots suspicious business names by analyzing their structure and format.

We also track how these patterns cluster together. Finding one pattern might raise mild suspicion, but finding multiple patterns in the same name significantly increases the risk score:

```python
try:
    for pname, pat in patterns.items():
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
```

This code calculates how often each pattern appears in suspicious versus legitimate business names, helping us refine our understanding of which patterns are truly indicative of fraud.

### Network Analysis: Finding Patterns in the Chaos

While obvious red flags in business names are fun to catch, the really interesting fraud detection happens when we start looking at networks of relationships. Here's where we start to catch the more sophisticated fraudsters who might have used legitimate-sounding business names but give themselves away in other ways.

Let's look at the core of our network analysis:

```python
def analyze_networks(self, loan: pd.Series) -> tuple[float, List[str]]:
    risk_score = 0
    flags = []
    
    # Create address key for consistency
    address_key = f"{loan['BorrowerAddress']}_{loan['BorrowerCity']}_{loan['BorrowerState']}"
    business_name = loan['BorrowerName']
    
    # Update our network mappings
    self.address_to_businesses[address_key].add(business_name)
    self.business_to_addresses[business_name].add(address_key)
```

This code maintains two crucial data structures:
- `address_to_businesses`: Maps each address to all businesses registered there
- `business_to_addresses`: Maps each business to all addresses it uses

For non-programmers: Think of this like creating a web of connections. If you were investigating fraud manually, you might draw lines between businesses and addresses on a whiteboard. This code does the same thing, but automatically and at scale.

#### The Residential Address Red Flag

One of the first things we check is whether multiple businesses are operating from the same residential address:

```python
residential_score = self.analyze_address_type(loan['BorrowerAddress'])
businesses_at_address = len(self.address_to_businesses[address_key])

if businesses_at_address >= 2:
    if residential_score > 0.5:
        risk_score += 15 * businesses_at_address
        flags.append("Multiple businesses at residential address")
    else:
        risk_score += 8 * businesses_at_address
```

How do we determine if an address is residential? We look for telltale signs:

```python
self.RESIDENTIAL_INDICATORS = {
    'apt': 0.8, 'unit': 0.7, '#': 0.7, 'suite': 0.4,
    'floor': 0.3, 'po box': 0.9, 'p.o.': 0.9,
    'residence': 0.9, 'apartment': 0.9
}

self.COMMERCIAL_INDICATORS = {
    'plaza': -0.7, 'building': -0.5, 'tower': -0.6,
    'office': -0.7, 'complex': -0.5, 'center': -0.5,
    'mall': -0.8, 'commercial': -0.8
}
```

Each indicator has a weight. Finding "apt" in an address adds 0.8 to the residential score, while finding "plaza" subtracts 0.7. This creates a nuanced scoring system that can handle mixed-use buildings and edge cases.

For non-programmers: This is like having a checklist of things that make an address look residential (apartment numbers, unit numbers) versus commercial (words like "plaza" or "mall"). Each item on the checklist has a different importance level, and we add them all up to make our final decision.

#### Connected Components: Finding Fraud Networks

But individual addresses are just the beginning. The real power comes from finding networks of connected fraudulent applications:

```python
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

This code is doing something fascinating: it's finding all businesses connected through shared addresses, even if they're not directly sharing an address. Here's how:

1. For each business, we look up all its addresses
2. For each of those addresses, we find all other businesses registered there
3. We analyze the patterns in these connected business names

For non-programmers: Imagine you're playing Six Degrees of Kevin Bacon, but with businesses and addresses. Business A shares an address with Business B, which shares a different address with Business C. Even though A and C never directly share an address, we can see they're part of the same network.

#### Temporal Patterns: The Time Dimension of Fraud

The timing of loan applications turns out to be one of the most revealing indicators of fraud. Why? Because humans are terrible at randomness, and fraudsters submitting multiple applications tend to do it in ways that create detectible patterns. Let's dive into how we catch these temporal fingerprints:

```python
def analyze_time_patterns(self, loan: pd.Series) -> tuple[float, List[str]]:
    date = str(loan['DateApproved'])
    zip_code = str(loan['BorrowerZip'])[:5]
    lender = str(loan['OriginatingLender'])
    loan_number = str(loan['LoanNumber'])
    
    # Track applications by date and ZIP code
    self.date_patterns[date][zip_code].append({
        'loan_number': loan_number,
        'business_type': loan['BusinessType'],
        'amount': loan['InitialApprovalAmount'],
        'lender': lender
    })
```

This code builds up a fascinating data structure. For each date, we track all applications by ZIP code, creating a geographical timeline of loan applications. But the real magic happens when we analyze these patterns:

```python
zip_cluster = self.date_patterns[date][zip_code]
if len(zip_cluster) >= self.ZIP_CLUSTER_THRESHOLD:
    amounts = [l['amount'] for l in zip_cluster]
    business_types = [l['business_type'] for l in zip_cluster]
    
    if max(amounts) - min(amounts) < min(amounts) * 0.1:
        risk_score += 20
        flags.append(f"Part of cluster: {len(zip_cluster)} similar loans in ZIP {zip_code} on {date}")
    
    if len(set(business_types)) == 1:
        risk_score += 15
        flags.append(f"Cluster of identical business types in ZIP {zip_code}")
```

For non-programmers: Imagine you're looking at a map where pins represent loan applications. If you see several pins pop up in the same neighborhood on the same day, that's interesting. If those pins represent businesses asking for suspiciously similar loan amounts, that's very interesting. If they're all the same type of business? Now we're looking at a pattern that's unlikely to occur naturally.

#### Sequential Loan Numbers: Catching the Assembly Line

One of the most damning patterns we look for is sequential loan numbers. Legitimate applications from different businesses typically receive random-ish loan numbers. But when someone's submitting fraudulent applications in bulk, they often end up with sequential numbers:

```python
def is_roughly_sequential(self, loan_numbers: List[str]) -> bool:
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
```

This code is cleverer than it might first appear. Instead of looking for perfectly sequential numbers (which would be too obvious), it looks for "roughly" sequential ones. Here's why:

1. It extracts just the numeric portions of loan numbers (some lenders add prefixes or suffixes)
2. It calculates the gaps between numbers
3. It allows for small, random-looking gaps (maybe the fraudster submitted some applications that were rejected)
4. But it still catches the overall pattern of numbers that are too close together to be coincidence

For non-programmers: Think of it like looking at ticket numbers at a deli counter. If you see numbers like 157, 158, 159, that's suspicious – real customers don't usually arrive in perfect order. But even numbers like 157, 159, 162 might be suspicious if you're seeing too many close numbers from supposedly different businesses.

```python
if len(current_batch) >= 5:
    amounts = [l['amount'] for l in current_batch]
    if max(amounts) - min(amounts) < min(amounts) * 0.1:
        batch_names = {l['business_name'] for l in current_batch}
        pattern_score, pattern_flags = self.analyze_name_patterns(batch_names)
        risk_score += 15 + pattern_score
        flags.append(f"Part of suspicious batch: {len(current_batch)} similar loans from same lender")
        flags.extend(pattern_flags)
```

For non-programmers: Think of it like spotting a factory assembly line for loan applications. Real businesses apply for loans at random times with varying amounts. When you see multiple applications coming in rapid-fire with nearly identical loan amounts, that's like spotting a counterfeiter's printing press running at full speed.

#### Lender Batch Analysis: Finding the Factory Lines

While looking for sequential loan numbers can catch some fraudulent patterns, more sophisticated fraudsters might try to avoid this by randomizing their loan numbers. That's where our lender batch analysis comes in - it can catch suspicious patterns even when the loan numbers themselves look random.

Here's how it works:

```python
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
        risk_score += 15
        flags.append(f"Part of suspicious batch: {len(current_batch)} similar loans from same lender")
```

This code is looking for something subtle but telling: batches of loans from the same lender location on the same day with suspiciously similar amounts. Let's break down why this matters:

1. We create a unique key combining the lender, their location, and the date
2. For each batch of 5 or more loans, we look at how similar the loan amounts are
3. If all the amounts are within 10% of each other, that's suspicious

For non-programmers: Imagine you're a bank teller, and five different people come in on the same day asking for almost exactly the same amount of money. That would be weird, right? Maybe it's a coincidence once, but when you see this pattern repeatedly from the same lender, something's probably up.

What makes this check powerful is that it catches fraud rings that are trying to be clever. They might vary their business names, use different addresses, and avoid sequential loan numbers - but humans are terrible at generating truly random numbers. When they're mass-producing fraudulent applications, they tend to stick to a few "safe" loan amounts that they know will get approved.

The system is particularly interested when it finds these suspicious batches in combination with other risk factors. If we see a batch of similar loans AND they're all using suspicious business names or addresses, the risk score gets multiplied significantly. This helps us distinguish between legitimate events (like a franchise business where multiple locations might apply for similar amounts) and coordinated fraud attempts.

#### The Early Warning System: High-Risk Lenders

One of the more controversial but effective parts of our system is tracking which lenders have a history of processing fraudulent loans. This creates a feedback loop that can help catch new fraud patterns:

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

Each lender has a risk multiplier. When we see an application from a high-risk lender that also has other suspicious indicators, we amplify the risk score:

```python
if lender in self.HIGH_RISK_LENDERS and len(flags) > 0:
    score += 15 * self.HIGH_RISK_LENDERS[lender]
    flags.append('High-risk lender')
```

For non-programmers: This is like knowing which security guards at a bank tend to be less thorough. If you see someone suspicious walking past one of those guards, you pay extra attention.

Looking at the blog post, I suggest adding a new section right after the "The Early Warning System: High-Risk Lenders" section. Here's the exact content to add:

#### Loan Forgiveness Patterns: Following the Money

One of the subtler but important fraud indicators in our system comes from analyzing loan forgiveness patterns. PPP loans were designed to be forgivable if used properly, but fraudsters often exhibit distinctive patterns in how they handle forgiveness:

```python
# Risk weight for forgiveness-related flags
self.RISK_WEIGHTS = {
    'forgiveness': 0.05,
    # ... other weights
}
```

Why such a relatively low weight (0.05)? Because forgiveness patterns alone aren't strongly indicative of fraud, but they become powerful signals when combined with other risk factors. Here's what we look for:

1. Immediate Forgiveness Applications: Legitimate businesses typically need time to properly document their use of funds. Applications for forgiveness submitted unusually quickly after receiving the loan get extra scrutiny.

2. All-or-Nothing Patterns: When we see clusters of loans from the same lender or geographic area all requesting either 100% forgiveness or no forgiveness at all, that's a red flag. Legitimate businesses typically show more variation in their forgiveness amounts based on their actual use of funds.

3. Mismatched Documentation: We look for discrepancies between the forgiveness amount requested and other loan attributes. For instance, if a business claimed zero employees but requests full forgiveness for payroll costs, that's a clear inconsistency.

For non-programmers: Think of loan forgiveness like a receipt for how the money was spent. Just as a store might be suspicious of someone trying to return items with questionable receipts, we're suspicious of loans where the forgiveness patterns don't match what we'd expect from legitimate business operations.

#### Time-Based Clustering: Catching the Mass Producers

One of our most powerful fraud detection techniques involves analyzing the temporal patterns of loan applications. Fraudsters, when submitting multiple applications, tend to do it in ways that create detectable time-based clusters. Here's why this works:

```python
lender_key = f"{loan['OriginatingLender']}_{loan['OriginatingLenderLocationID']}_{loan['DateApproved']}"
batch_info = {
    'loan_number': loan['LoanNumber'],
    'amount': loan['InitialApprovalAmount'],
    'business_name': loan['BorrowerName'],
    'timestamp': loan['DateApproved']
}
self.lender_batches[lender_key].append(batch_info)
```

This seemingly simple code is doing something remarkably sophisticated. By creating a unique key combining the lender, their physical location, and the date, we can detect patterns that would be invisible if we looked at any single factor in isolation.

Why these three factors? Because fraudulent loan applications often follow a "mass production" pattern:
1. Same lender (fraudsters find a bank or processor that's easier to work with)
2. Same location (they typically submit through the same branch or portal)
3. Same day (they try to push through as many applications as possible while they have a working formula)

For example, imagine a legitimate small business district. You might see:
- Multiple loans from the same bank branch
- All on the same day (maybe there was a PPP workshop)
- But with varying loan amounts based on each business's actual needs

Now contrast that with a fraudulent pattern we've detected:
```python
if len(current_batch) >= 5:
    amounts = [l['amount'] for l in current_batch]
    if max(amounts) - min(amounts) < min(amounts) * 0.1:
        batch_names = {l['business_name'] for l in current_batch}
        pattern_score, pattern_flags = self.analyze_name_patterns(batch_names)
        risk_score += 15 + pattern_score
        flags.append(f"Part of suspicious batch: {len(current_batch)} similar loans from same lender")
```

This code catches a very specific type of fraud where we see:
- 5+ applications through the same lender location
- All on the same day
- All asking for suspiciously similar amounts (within 10% of each other)

Why is this suspicious? Because legitimate businesses, even in the same industry and location, rarely need exactly the same loan amount. The variation in real business needs - number of employees, rent costs, utility expenses - naturally creates diversity in loan amounts.

The timing aspect becomes even more powerful when combined with other patterns:
```python
def analyze_time_patterns(self, loan: pd.Series) -> tuple[float, List[str]]:
    date = str(loan['DateApproved'])
    zip_code = str(loan['BorrowerZip'])[:5]
    
    self.date_patterns[date][zip_code].append({
        'loan_number': loan_number,
        'business_type': loan['BusinessType'],
        'amount': loan['InitialApprovalAmount'],
        'lender': lender
    })
```

This code tracks not just when applications come in, but how they relate geographically. We've found that fraud rings often:
1. Submit multiple applications in the same ZIP code
2. Use the same business type for all applications
3. Request identical or very similar loan amounts
4. Submit everything within a short time window

For non-programmers: Think of it like spotting a counterfeiter by realizing that no legitimate business would print 100 identical $20 bills all at the same time. Similarly, no legitimate business district would have 10 different companies all needing exactly $18,749 in PPP loans on the same Tuesday afternoon.

The time-based clustering becomes even more powerful when combined with our sequential loan number detection:
```python
if any("Sequential loan numbers" in flag for flag in flags) and len(flags) > 1:
    score = score * 1.5
```

When we see both time-based clustering AND sequential loan numbers, we multiply the risk score. Why? Because this combination strongly suggests an automated or semi-automated fraud operation - someone submitting multiple applications so quickly that they're getting sequential loan numbers from the system.

This multi-dimensional temporal analysis has proven to be one of our most reliable fraud indicators, especially because legitimate businesses tend to apply for PPP loans in much more random, natural patterns spread out over time.

#### The Devil in the Demographics

One of the more subtle patterns we discovered was in how fraudulent applications handle demographic information. Legitimate business owners typically fill out these fields naturally, with a mix of responses. Fraudsters often take shortcuts:

```python
demographic_fields = ['Race', 'Gender', 'Ethnicity']
missing_demographics = sum(1 for field in demographic_fields 
    if str(loan.get(field, '')).lower() in ['unanswered', 'unknown'])

if missing_demographics == len(demographic_fields) and len(flags) > 1:
    score += 10
    flags.append('Missing all demographics')
```

This code isn't just checking for missing demographics – that alone isn't suspicious. It's looking for applications that:
1. Have all demographic fields marked as "unknown" or "unanswered"
2. Also have other suspicious indicators

For non-programmers: It's like looking at a stack of job applications where some are filled out completely, some have a few blanks, and others just have "N/A" written in every single field. Those last ones make you wonder if someone was trying to fill out forms as quickly as possible without actually providing any real information.

#### Risk Score Aggregation: Putting It All Together

The heart of our system is how it combines all these different risk factors into a final score. At first glance, you might think we could just add up all the individual risk factors and call it a day. But fraud detection is more nuanced than that. Here's how we weight different types of risk:

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
    'name_pattern': 0.05
}
```

These weights reflect both the reliability of each indicator and its importance. For example, the number of jobs reported gets a higher weight (0.20) because it's a concrete number that's harder to fake convincingly, while demographic information gets a lower weight (0.05) because missing demographics alone aren't strong evidence of fraud.

For non-programmers: Think of this like a judge considering different types of evidence in a trial. DNA evidence might get more weight than circumstantial evidence, just like job numbers get more weight than missing demographics.

#### Risk Score Calibration: Preventing False Alarms

While our scoring system is designed to catch fraud, we're equally focused on preventing false positives. One of our most important safeguards is a sophisticated score calibration system that prevents loans from being flagged as high-risk unless they exhibit multiple strong indicators of fraud:

```python
def score_loan(loan: pd.Series) -> pd.Series:
    # ... risk scoring logic ...
    
    final_score = score
    if "Exact maximum loan amount detected" not in flags and len(flags) < 2:
        final_score = min(49, final_score)
```

This code is doing something clever: unless a loan hits the exact maximum allowed amount (a strong fraud indicator), we require at least two independent risk flags before allowing the score to enter the high-risk range (50 or above). This means that no single suspicious pattern, no matter how strong, can trigger a high-risk designation on its own.

For non-programmers: Think of this like a legal system's "beyond reasonable doubt" standard. Just as we wouldn't convict someone based on a single piece of circumstantial evidence, we won't flag a loan as high-risk based on a single suspicious pattern. We need multiple pieces of evidence working together to cross that threshold.

This calibration system helps ensure that:
1. We don't overwhelm investigators with false positives
2. Legitimate but unusual businesses don't get incorrectly flagged
3. Our risk scores remain meaningful and actionable

The only exception to this rule is when a loan requests the exact maximum amount allowed under PPP rules ($20,832 or $20,833). This specific amount is such a strong indicator of potential fraud that it can push a loan into high-risk territory even without other supporting evidence.


#### The Multiplier Effect: When Red Flags Work Together

But here's where it gets really interesting. We don't just add up weighted risk factors – we look for combinations that make each other more suspicious:

```python
def calculate_risk_scores(self, chunk: pd.DataFrame) -> pd.DataFrame:
    def score_loan(loan: pd.Series) -> pd.Series:
        score = 0
        flags = []
        
        # Basic risk factors
        has_duplicates, duplicate_flags = self.check_multiple_applications(loan)
        if has_duplicates:
            score += 30
            flags.extend(duplicate_flags)
            
        # Check per-employee amount
        if loan['JobsReported'] > 0:
            amount = float(loan['InitialApprovalAmount'])
            per_employee = amount / loan['JobsReported']
            if per_employee > 12000:
                score += 15
                flags.append(f'High amount per employee: ${per_employee:,.2f}')
                if per_employee > 14000:
                    score += 15
                    flags.append('Extremely high amount per employee')
```

Notice how we handle the per-employee amount: finding a high amount adds 15 to the score, but finding an extremely high amount adds another 15. This is because extremes are even more suspicious than merely unusual values.

But the real magic happens when we combine different types of flags:

```python
        # Apply network analysis
        network_score, network_flags = self.analyze_networks(loan)
        score += network_score
        flags.extend(network_flags)
        
        # If we found sequential loan numbers AND other flags,
        # the whole thing becomes much more suspicious
        if any("Sequential loan numbers" in flag for flag in flags) and len(flags) > 1:
            score = score * 1.5
```

For non-programmers: This is like solving a mystery where finding one clue makes other clues more significant. Finding muddy footprints might be suspicious, and finding a broken window might be suspicious, but finding both together is way more than twice as suspicious.

#### The False Positive Prevention System

One of the trickiest parts of fraud detection is avoiding false positives. It's not enough to just find suspicious patterns – we need to be confident that we're not flagging legitimate businesses. Our false positive prevention system is multi-layered and surprisingly sophisticated:

```python
def validate_high_risk_loans(self, loans: pd.DataFrame) -> pd.DataFrame:
    def validate_loan(loan: pd.Series) -> bool:
        if loan['BorrowerName'] in self.known_businesses:
            return False
            
        validation_score = 0
        flags = str(loan['RiskFlags']).split(';')
        
        # Require multiple independent risk factors
        if len(flags) < 2:
            return False
```

First, we require multiple independent risk factors. A single red flag, no matter how suspicious, isn't enough to trigger a high-risk designation. This helps prevent false positives from quirky but legitimate businesses.

But here's where it gets interesting – we also look for positive indicators that can counteract suspicious patterns:

```python
        # Check for positive indicators
        if loan['JobsReported'] > 0:
            amount = float(loan['InitialApprovalAmount'])
            per_employee = amount / loan['JobsReported']
            if per_employee < 8000:
                validation_score -= 20
                
        if (loan['BusinessType'] in self.LEGITIMATE_BUSINESS_TYPES and 
            str(loan['NAICSCode']) not in self.GENERIC_NAICS):
            validation_score -= 15
            
        if loan['LoanStatus'] == 'Paid in Full':
            validation_score -= 15
```

For non-programmers: Think of this like a legal system's presumption of innocence. We're not just looking for evidence of guilt – we're also actively looking for evidence of innocence. A business might have some suspicious indicators, but if they've paid back their loan in full and have reasonable per-employee costs, maybe they're actually legitimate.

#### The Business Type Legitimacy Scale

We maintain a detailed understanding of which business types are more or less likely to be vehicles for fraud:

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

But here's the clever part – we don't just use these categorizations blindly. Instead, we look at how they interact with other factors:

```python
business_type = str(loan['BusinessType'])
if business_type in self.HIGH_RISK_BUSINESS_TYPES:
    if len(flags) > 1:  # Only count if there are other red flags
        score += 15 * self.HIGH_RISK_BUSINESS_TYPES[business_type]
        flags.append('High-risk business type')
```

For non-programmers: This is like how a convenience store being open late isn't suspicious, and a person wearing a mask during COVID isn't suspicious, but a masked person in a convenience store at 3 AM might warrant a second look. Context matters.

#### The NAICS Code Validation System

One of our more sophisticated checks involves validating NAICS (North American Industry Classification System) codes. Fraudsters often choose generic or mismatched industry codes that can reveal their applications as suspicious. Here's how we catch them:

```python
self.GENERIC_NAICS = {
    '541990': 0.7,  # Other Professional Services
    '541618': 0.7,  # Other Management Consulting
    '541611': 0.6,  # Administrative Management Consulting
    '453998': 0.8,  # All Other Miscellaneous Store Retailers
    '454390': 0.8,  # Other Direct Selling Establishments
    '541213': 0.8,  # Tax Preparation Services
    '812111': 0.8,  # Barber Shops
    '812113': 0.8,  # Nail Salons
}
```

Each code has a risk weight based on how often we've seen it used in fraudulent applications. But here's where it gets interesting – we don't just look at the codes in isolation. We cross-reference them with the business name and other attributes:

```python
def analyze_naics_consistency(self, loan: pd.Series) -> tuple[float, List[str]]:
    naics = str(loan['NAICSCode'])
    business_name = loan['BorrowerName'].lower()
    risk_score = 0
    flags = []
    
    # Check for generic high-risk codes
    if naics in self.GENERIC_NAICS:
        # Look for mismatches with business name
        if 'consulting' in business_name and naics != '541611':
            risk_score += 25
            flags.append('Business claims to be consulting but uses different NAICS')
        elif 'salon' in business_name and naics != '812113':
            risk_score += 25
            flags.append('Business claims to be salon but uses different NAICS')
```

For non-programmers: Think of NAICS codes like genre categories for businesses. If someone claimed their business was a bookstore but categorized it as a car dealership, that would be suspicious. We're doing the same thing, but with much more nuance and at scale.

#### Statistical Pattern Analysis of NAICS Codes

But individual NAICS validation isn't enough. We also look for suspicious patterns in how codes are used across multiple applications:

```python
def analyze_naics_patterns(self, chunk: pd.DataFrame) -> None:
    naics_groups = chunk.groupby('NAICSCode')
    
    for naics, group in naics_groups:
        # Calculate average loan amount for this NAICS
        avg_amount = group['InitialApprovalAmount'].mean()
        std_amount = group['InitialApprovalAmount'].std()
        
        # Look for suspiciously uniform loan amounts
        if len(group) >= 5 and std_amount < avg_amount * 0.05:
            for _, loan in group.iterrows():
                self.add_risk_flag(
                    loan['LoanNumber'],
                    f"Part of suspicious NAICS cluster with uniform amounts",
                    25
                )
```

This code is doing something subtle but powerful:
1. It groups all loans by their NAICS code
2. For each NAICS code with 5+ applications, it calculates the average loan amount and standard deviation
3. If the amounts are suspiciously similar (standard deviation < 5% of mean), it flags all loans in that group

Why? Because legitimate businesses in the same industry still tend to need different loan amounts based on their specific situations. When we see multiple businesses in the same industry asking for nearly identical amounts, that's a red flag.

#### The Secondary Analysis System: Statistical Pattern Mining

The secondary analysis system in `analyze_patterns_in_suspicious_loans.py` takes our fraud detection to another level. While the primary system is great at catching individual suspicious loans, the secondary system looks for broader patterns that might not be visible when looking at loans one at a time.

Here's how we start:

```python
class SuspiciousLoanAnalyzer:
    def __init__(self, suspicious_file: str, full_data_file: str):
        self.suspicious_file = suspicious_file
        self.full_data_file = full_data_file
        self.sus_data = None
        self.full_data = None

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        print("Loading suspicious loans data...")
        try:
            sus = pd.read_csv(self.suspicious_file, low_memory=False)
            print("Loading full loan dataset...")
            cols = [
                "LoanNumber", "BorrowerName", "BorrowerCity", 
                "BorrowerState", "OriginatingLender", 
                "InitialApprovalAmount", "BusinessType",
                "Race", "Gender", "Ethnicity", "NAICSCode", 
                "JobsReported"
            ]
            full = pd.read_csv(
                self.full_data_file,
                usecols=cols,
                dtype={"LoanNumber": str, "NAICSCode": str},
                low_memory=False
            )
```

Notice we're only loading specific columns from the full dataset. This isn't just for efficiency – it's because these are the fields that our statistical analysis has shown to be most relevant for pattern detection.

#### Address Analysis: A Weighted Approach to Location Red Flags

One of the most sophisticated parts of our fraud detection system is how it analyzes business addresses. Rather than using simple binary flags for residential versus commercial addresses, we implement a nuanced scoring system that weighs different address indicators:

```python
self.RESIDENTIAL_INDICATORS = {
    'apt': 0.8,      # Strong indicator of residential use
    'unit': 0.7,     # Common in both residential and commercial
    '#': 0.7,        # Often indicates apartment/unit number
    'suite': 0.4,    # Used in both residential and commercial
    'floor': 0.3,    # Minimal weight due to common commercial use
    'po box': 0.9,   # Very suspicious for a PPP loan
    'p.o.': 0.9,     # Alternative PO Box format
    'box': 0.8,      # Another PO Box variant
    'residence': 0.9, # Explicit residential indicator
    'house': 0.8,    # Clear residential indicator
    'condo': 0.8,    # Usually residential
    'room': 0.9      # Highly suspicious for a business
}

self.COMMERCIAL_INDICATORS = {
    'plaza': -0.7,      # Strong commercial indicator
    'building': -0.5,   # Common commercial term
    'tower': -0.6,      # Usually indicates office building
    'office': -0.7,     # Clear commercial use
    'complex': -0.5,    # Often commercial
    'center': -0.5,     # Shopping or business center
    'mall': -0.8,       # Definitely commercial
    'commercial': -0.8,  # Explicit commercial indicator
    'industrial': -0.8  # Clear business/industrial use
}
```

This weighted system allows for much more nuanced analysis than a simple residential/commercial binary check. Here's how it works:

1. Each address gets scored based on all matching indicators
2. Commercial indicators subtract from the residential score
3. Multiple indicators can stack (e.g., "apt #" would trigger both 'apt' and '#' weights)
4. The final score determines how suspicious the address is

For example, an address like "Apt 4B, 123 Main St" would get a high residential score (triggering both 'apt' and '#' indicators), while "Suite 400, Commerce Plaza" would get a low or negative score due to mixed residential ('suite') and strong commercial ('plaza') indicators.

For non-programmers: Think of this like a real estate agent evaluating a property. Just as they don't just say "it's residential" or "it's commercial" but rather look at multiple features to determine the true nature of the property, our system uses multiple clues to build a complete picture of each address.

This sophisticated scoring becomes especially powerful when combined with other risk factors. A high residential score alone might not be suspicious – plenty of legitimate small businesses operated from homes during the pandemic. But a high residential score combined with multiple businesses at the same address? That's when our alarm bells start ringing.

#### Statistical Validation: The Chi-Square Test

One of our most powerful tools is the chi-square test for categorical variables. Here's how we use it to validate our findings:

```python
def analyze_categorical_patterns(
        self,
        sus: pd.DataFrame,
        full: pd.DataFrame,
        column: str,
        title: str,
        min_occurrences: int = 5,
    ) -> None:
        # Handle missing values consistently
        sus[column] = sus[column].fillna('Unknown')
        full[column] = full[column].fillna('Unknown')
        
        # Calculate value counts
        s_counts = sus[column].value_counts()
        f_counts = full[column].value_counts()
        
        # Create contingency table
        categories = sorted(set(s_counts.index) | set(f_counts.index))
        cont_table = np.zeros((2, len(categories)))
        
        # Fill in suspicious vs non-suspicious counts
        for i, cat in enumerate(categories):
            cont_table[0, i] = s_counts.get(cat, 0)  # Suspicious
            cont_table[1, i] = f_counts.get(cat, 0) - s_counts.get(cat, 0)  # Non-suspicious
```

For non-programmers: Imagine you're trying to prove that a casino's dice are loaded. You'd roll them many times and compare the distribution of numbers you get against what you'd expect from fair dice. The chi-square test is doing something similar – it's helping us prove that the patterns we're seeing in suspicious loans aren't just random chance.

#### Representation Ratio Analysis: Finding the Needles in the Haystack

The representation ratio analysis is where we really start to understand what makes fraudulent loans different from legitimate ones. Here's the core of how we calculate these ratios:

```python
def calculate_representation_ratio(
    self,
    suspicious_counts: pd.Series,
    full_counts: pd.Series,
    min_occurrences: int = 5,
) -> pd.DataFrame:
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
```

This code is doing something subtle but powerful. Instead of just comparing raw counts, it's comparing the percentage of suspicious loans with a particular characteristic to the percentage of all loans with that characteristic. 

For non-programmers: Imagine you're trying to figure out if a particular neighborhood has an unusually high number of fraudulent loans. Just knowing there are 10 fraudulent loans there isn't enough information. You need to know:
1. What percentage of all fraudulent loans are in this neighborhood?
2. What percentage of all loans are in this neighborhood?
3. How does the first percentage compare to the second?

If 5% of all loans are in the neighborhood but 25% of fraudulent loans are there, that's a representation ratio of 5 – a huge red flag.

#### Safe Division and Error Handling

Notice the `safe_divide` function we're using:

```python
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
```

This might seem like overkill, but it's crucial. When dealing with millions of loans and hundreds of characteristics, you'll inevitably run into edge cases:
- Division by zero when there are no loans of a particular type
- Floating point overflow when ratios get extremely large
- NaN values from missing or corrupted data

For non-programmers: Think of this like having a backup plan for every possible way your calculations could go wrong. It's like a chef who not only knows how to make a perfect soufflé but also knows exactly what to do if it falls, burns, or doesn't rise.

#### Statistical Significance Testing: Separating Signal from Noise

Finding unusual patterns is just the first step. We need to be confident that these patterns aren't just random chance. This is where our battery of statistical tests comes in:

```python
def analyze_patterns(self, sus: pd.DataFrame, full: pd.DataFrame) -> None:
    # Calculate chi-square test if possible
    if cont_table.shape[1] >= 2 and (cont_table > 0).all():
        try:
            # Returns chi2 statistic, p-value, degrees of freedom, expected frequencies
            _, p_chi2, _, _ = stats.chi2_contingency(cont_table)
            
            # If p-value is very small, pattern is significant
            if p_chi2 < 0.001:
                print(f"Pattern is HIGHLY significant (p={p_chi2:.2e})")
            elif p_chi2 < 0.05:
                print(f"Pattern is significant (p={p_chi2:.3f})")
            else:
                print(f"Pattern is not significant (p={p_chi2:.3f})")
                
        except Exception as e:
            print(f"Chi-square test calculation failed: {str(e)}")
```

For non-programmers: Think of this like a courtroom where we're trying to prove beyond reasonable doubt that what we're seeing isn't coincidence. The p-value is like our "beyond reasonable doubt" threshold - the smaller it is, the more confident we are that the pattern is real.

But we don't just rely on one test. Different types of data require different statistical approaches:

```python
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
```

Each test has its own strengths:
- T-tests are great for comparing averages when the data is normally distributed
- Kolmogorov-Smirnov tests look at the entire distribution shape
- Mann-Whitney U tests are robust against outliers and don't assume normal distribution

#### Multiple Hypothesis Testing: The Multiple Comparisons Problem

When you're running hundreds or thousands of statistical tests, you run into a thorny problem: even if there's no real pattern, some tests will come up significant just by chance. Here's why: if you use a p-value threshold of 0.05, that means each test has a 5% chance of showing a false positive. Run 100 tests, and you'd expect about 5 false positives just by random chance!

Here's how we handle this:

```python
def adjust_pvalues(self, pvalues: List[float], method: str = 'fdr_bh') -> np.ndarray:
    """
    Adjust p-values for multiple comparisons using various methods.
    
    Args:
        pvalues: List of p-values to adjust
        method: Correction method ('bonferroni', 'fdr_bh', or 'holm')
    """
    if method == 'bonferroni':
        # Bonferroni correction: multiply each p-value by number of tests
        return np.minimum(np.array(pvalues) * len(pvalues), 1.0)
        
    elif method == 'fdr_bh':
        # Benjamini-Hochberg procedure
        pvalues = np.array(pvalues)
        n = len(pvalues)
        
        # Sort p-values and get their original indices
        sorted_indices = np.argsort(pvalues)
        sorted_pvalues = pvalues[sorted_indices]
        
        # Calculate adjusted values
        adjusted = np.zeros(n)
        for i, p in enumerate(sorted_pvalues):
            adjusted[i] = p * n / (i + 1)
            
        # Ensure monotonicity
        for i in range(n-2, -1, -1):
            adjusted[i] = min(adjusted[i], adjusted[i+1])
            
        # Return to original order
        final = np.zeros(n)
        final[sorted_indices] = adjusted
        return final
        
    elif method == 'holm':
        # Holm-Bonferroni method
        pvalues = np.array(pvalues)
        n = len(pvalues)
        
        # Sort p-values and get indices
        sorted_indices = np.argsort(pvalues)
        sorted_pvalues = pvalues[sorted_indices]
        
        # Calculate adjusted values
        adjusted = np.zeros(n)
        for i, p in enumerate(sorted_pvalues):
            adjusted[i] = p * (n - i)
            
        # Ensure monotonicity
        for i in range(n-2, -1, -1):
            adjusted[i] = max(adjusted[i], adjusted[i+1])
            
        # Return to original order
        final = np.zeros(n)
        final[sorted_indices] = adjusted
        return final
```

For non-programmers: Imagine you're a judge looking at evidence from 100 different cases. If you use the same standard of proof for each case independently, you're likely to convict some innocent people just by chance. These correction methods are like adjusting your standard of proof based on how many cases you're judging at once.

Each method has its strengths:
- Bonferroni correction is the most conservative, essentially dividing your significance threshold by the number of tests
- Benjamini-Hochberg controls the false discovery rate, giving you more statistical power while still limiting false positives
- Holm-Bonferroni is a middle ground, less conservative than Bonferroni but stricter than Benjamini-Hochberg

Here's how we apply these corrections in practice:

```python
def analyze_categorical_patterns_corrected(
    self,
    sus: pd.DataFrame,
    full: pd.DataFrame,
    columns: List[str]
) -> None:
    # Store all p-values
    pvalues = []
    test_descriptions = []
    
    for column in columns:
        # Perform chi-square test
        cont_table = self.create_contingency_table(sus[column], full[column])
        try:
            _, p_val, _, _ = stats.chi2_contingency(cont_table)
            pvalues.append(p_val)
            test_descriptions.append(f"Chi-square test for {column}")
        except Exception:
            continue
            
    # Adjust p-values using Benjamini-Hochberg
    adjusted_pvalues = self.adjust_pvalues(pvalues, method='fdr_bh')
    
    # Report significant findings
    for i, (desc, p_orig, p_adj) in enumerate(zip(
        test_descriptions, pvalues, adjusted_pvalues
    )):
        if p_adj < 0.05:  # Still significant after correction
            print(f"{desc}:")
            print(f"  Original p-value: {p_orig:.2e}")
            print(f"  Adjusted p-value: {p_adj:.2e}")
```

#### Handling Rare Events: The Low Count Problem

When dealing with fraud detection, many of the most interesting patterns involve rare events. This creates a statistical challenge: how do you determine if something is significant when it only happens a few times? Here's how we handle it:

```python
def analyze_rare_patterns(
    self,
    sus: pd.DataFrame,
    full: pd.DataFrame,
    column: str,
    min_occurrences: int = 5
) -> None:
    # Get value counts
    sus_counts = sus[column].value_counts()
    full_counts = full[column].value_counts()
    
    # For rare categories, use Fisher's exact test instead of chi-square
    rare_categories = set(sus_counts[sus_counts < min_occurrences].index)
    
    for category in rare_categories:
        # Create 2x2 contingency table for this category
        contingency = np.array([
            # Suspicious loans: [this category, other categories]
            [sus_counts.get(category, 0),
             len(sus) - sus_counts.get(category, 0)],
            # Non-suspicious loans: [this category, other categories]
            [full_counts.get(category, 0) - sus_counts.get(category, 0),
             len(full) - len(sus) - (full_counts.get(category, 0) - sus_counts.get(category, 0))]
        ])
        
        # Use Fisher's exact test
        _, p_value = stats.fisher_exact(contingency)
        
        if p_value < 0.05:
            # Calculate odds ratio for effect size
            odds_ratio = (contingency[0,0] * contingency[1,1]) / \
                        (contingency[0,1] * contingency[1,0])
            
            print(f"Rare category '{category}' is significant:")
            print(f"  P-value: {p_value:.2e}")
            print(f"  Odds ratio: {odds_ratio:.2f}x more likely in suspicious loans")
```

For non-programmers: Imagine you're investigating a series of bank robberies. If you notice that three of the ten robberies happened on a full moon, that might seem suspicious. But is it really significant? If there are full moons 7% of the time, seeing 30% of robberies on full moons could be meaningful even with small numbers. This code helps us make that determination mathematically.

#### Bootstrap Resampling for Rare Events

But Fisher's exact test isn't always enough. Sometimes we need to understand the uncertainty in our estimates for rare events. That's where bootstrap resampling comes in:

```python
def bootstrap_rare_event_analysis(
    self,
    sus: pd.DataFrame,
    full: pd.DataFrame,
    column: str,
    category: str,
    n_bootstrap: int = 10000
) -> None:
    # Original observation
    orig_ratio = self.calculate_representation_ratio(
        sus[column] == category,
        full[column] == category
    )
    
    # Bootstrap resampling
    bootstrap_ratios = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        sus_resample = sus.sample(n=len(sus), replace=True)
        full_resample = full.sample(n=len(full), replace=True)
        
        ratio = self.calculate_representation_ratio(
            sus_resample[column] == category,
            full_resample[column] == category
        )
        bootstrap_ratios.append(ratio)
        
    # Calculate confidence intervals
    ci_lower = np.percentile(bootstrap_ratios, 2.5)
    ci_upper = np.percentile(bootstrap_ratios, 97.5)
    
    print(f"Bootstrap analysis for category '{category}':")
    print(f"  Original ratio: {orig_ratio:.2f}x")
    print(f"  95% CI: [{ci_lower:.2f}x, {ci_upper:.2f}x]")
```

For non-programmers: This is like running thousands of simulations where we randomly resample our data to understand how much our results might vary by chance. If we see 3 out of 10 robberies on full moons, this helps us understand how likely that pattern is to hold up if we had slightly different data.