# Phase 5: Naive Bayes Intrusion Sensor

## 1. Why Naive Bayes Multiplies Likelihoods
Bayes' rule says the probability of a class (`normal` or `attack`) given a traffic record is proportional to the class prior times the joint probability of seeing all six features together. Estimating that joint probability directly would require counting every possible combination of protocol, port, bytes, failed logins, hour, and connection rate ($3 \times 6 \times 5 \times 4 \times 4 \times 4 = 5{,}760$ cells per class). Naive Bayes avoids that by assuming each feature is conditionally independent given the class, turning one giant table into six tiny per-feature likelihood tables multiplied together (or added in log space to prevent underflow).

## 2. The Zero-Count Problem and Laplace Smoothing
If a feature value never appears in a class during training (common when data is scarce), its raw likelihood is $0$, and $\log(0) = -\infty$. Because Naive Bayes multiplies every feature likelihood together, a single unseen value vetoes all other evidence and forces that class probability to zero—or rules out every class completely. Laplace smoothing adds $\alpha = 1$ to every count (and $\alpha \cdot K$ to the denominator for a $K$-valued feature), turning unseen values into rare events instead of impossible ones.

## 3. Why We Judge by Recall (and F1 / False Alarms), Not Accuracy Alone
In our dataset 80% of traffic records are normal and only 20% are attacks. A trivial classifier that always predicts `normal` achieves 80.0% accuracy while catching 0% of attacks (recall = 0.0). For CyberStrike's Defender, missing an intrusion is far more costly than investigating an occasional false alarm, so we measure recall (detection rate), false alarm rate, precision, and F1—and tune the alert threshold to lift recall cleanly.

## 4. Why the Test Pile Is Used Only Once
We split the 10,000 records stratified 60/20/20 into train, validation, and test sets. Training learns feature counts on the train set, and model decisions (threshold selection, smoothing studies, forward feature selection) use only the development/validation data. If we tuned choices on the test set, its score would be optimistically biased by leaking test information; touching the test set once at the end gives an honest measurement of `detection_rate` and `false_alarm_rate` for Phase 6.

