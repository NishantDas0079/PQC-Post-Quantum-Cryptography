# Post-Quantum Cryptography & Side-Channel Analysis
# ML-KEM (Kyber) — Theory, Implementation, and Real-World Security Risks

This repository presents a comprehensive study of ML-KEM (Kyber) — the NIST-standardized post-quantum Key Encapsulation Mechanism (FIPS-203) — with a strong focus on practical security.

While ML-KEM is mathematically secure against quantum attacks, real-world implementations remain vulnerable to side-channel attacks (SCA).
This repository bridges that critical gap by covering:

# Foundations of post-quantum cryptography

Internal working of ML-KEM

Polynomial arithmetic & NTT

Template attacks and Correlation Power Attacks (CPA)

Real experimental results on power-trace datasets

This repo is designed for students, researchers, and security engineers who want to understand not only why ML-KEM is secure, but how it can still fail in practice.

# Motivation: Why Post-Quantum Cryptography Matters

Modern cryptography (RSA, ECC) relies on problems that are:

Hard for classical computers

Easy for quantum computers using Shor’s algorithm

With quantum computers advancing, today’s encrypted data may be harvested now and decrypted later.

To address this, NIST launched a global standardization effort (2016–2024), resulting in the adoption of ML-KEM as the next-generation public-key primitive.

# What Is ML-KEM (Kyber)?

ML-KEM (Module-Lattice-Based Key Encapsulation Mechanism) is a quantum-resistant cryptographic scheme used to securely exchange secret keys over public channels.

Core Properties

Based on Learning With Errors (LWE)

Resistant to both classical and quantum attacks

Efficient enough for real-world deployment

Standardized as FIPS-203 (2024)

ML-KEM does not encrypt messages directly — it securely exchanges a shared secret key used for symmetric encryption.

# Learning With Errors (LWE): The Mathematical Backbone

At the heart of ML-KEM lies intentional noise.

# Instead of clean equations:

𝐴
⋅
𝑠
=
𝑡
A⋅s=t

# ML-KEM uses:

𝐴
⋅
𝑠
+
𝑒
=
𝑡
A⋅s+e=t

Where:

s → secret key

e → small random error

t → public key

The noise makes recovering s computationally infeasible — even for quantum computers.

Errors are not a flaw — they are the defense mechanism.

# ML-KEM Algorithm Overview
1️⃣ Key Generation (KeyGen)

Generates:

Public key (A, t)

Secret key s

Computation:

𝑡
=
𝐴
⋅
𝑠
+
𝑒
(
m
o
d
𝑞
)
t=A⋅s+e(modq)
# Encapsulation (Encaps)

Sender:

Uses recipient’s public key

Generates a ciphertext (u, v)

Result:

Shared secret key embedded in ciphertext

# Decapsulation (Decaps)

Receiver:

Uses secret key s

Computes:

𝑣
′
=
𝑣
−
𝑢
𝑇
⋅
𝑠
v
′
=v−u
T
⋅s

Recovers the shared secret

# This decapsulation phase is the primary target of side-channel attacks.

# Polynomial Arithmetic & NTT in ML-KEM

ML-KEM relies heavily on polynomial multiplications, which are computationally expensive.

To optimize this:

Number Theoretic Transform (NTT) is used

NTT is the modular analogue of FFT

Enables fast convolution via component-wise multiplication

# Polynomial multiplication:

𝑓
×
𝑔
=
INTT
(
NTT
(
𝑓
)
∘
NTT
(
𝑔
)
)
f×g=INTT(NTT(f)∘NTT(g))

The pair-pointwise multiplication (basemul) inside NTT becomes a leakage hotspot.

# Side-Channel Attacks: The Practical Threat

Even when cryptography is mathematically secure, physical implementations leak information through:

Power consumption

Timing

Electromagnetic radiation

This repository focuses on power analysis attacks targeting ML-KEM decapsulation.

# Side-Channel Attack Models Studied
🔴 Profiled Template Attack

Attack model:

Attacker has access to:

Device

Training traces

Builds statistical templates based on leakage classes

Key ideas:

Classify traces by Hamming weight

Identify Points of Interest (PoIs) using SOSD

Match unknown traces to templates

Results:

Full recovery of ML-KEM secret sub-keys

Requires a few hundred traces

# Unprofiled Correlation Power Attack (CPA)

Attack model:

No profiling phase

Uses statistical correlation

Method:

Guess key values

Compute hypothetical leakage (Hamming weight)

Measure Pearson correlation with power traces

Results:

More efficient than template attacks

Some sub-keys recovered with ~30 traces

# CPA outperforms template attacks on the given dataset.

# Power Analysis Attacks

Power analysis attacks exploit variations in a device’s power consumption during cryptographic operations.

🔹 Simple Power Analysis (SPA)

Uses single or few power traces

Relies on visible patterns in power consumption

Effective against naive implementations

Example:
Different operations (addition vs multiplication) consume distinguishable power.

🔹 Differential Power Analysis (DPA)

Uses statistical analysis across many traces

Targets intermediate secret-dependent values

More powerful than SPA

🔹 Correlation Power Analysis (CPA)

A refined form of DPA

Uses correlation coefficients between:

hypothetical leakage (HW/HD models)

measured power traces

Extremely effective against ML-KEM implementations

✔ Covered extensively in this repository

🔹 Template Attacks

Profiled attacks

Attacker builds statistical templates using a similar device

Highly accurate with fewer attack traces

✔ Demonstrated in this repository

# Timing Attacks

Timing attacks exploit variations in execution time.

Conditional branches

Cache hits/misses

Early termination conditions

Example:

Variable-time polynomial reductions

Conditional rejection sampling

Even nanosecond-level timing differences can leak secrets.

# Electromagnetic (EM) Attacks

EM attacks measure electromagnetic radiation emitted by a device.

Advantages:

Non-invasive

High spatial resolution

Can isolate individual components

Often more powerful than power analysis.

# Cache-Based Side-Channel Attacks

Exploit shared cache behavior in CPUs.

Examples:

Prime+Probe

Flush+Reload

Leakage source:

Memory access patterns

Data-dependent cache usage

Common in:

Cloud environments

Multi-tenant systems

# Acoustic Attacks

Use sound emitted by hardware components:

Voltage regulators

Capacitors

Coils

Surprisingly effective against:

RSA

ECC

Embedded devices

# Fault Injection Attacks

Instead of observing leakage, attackers induce faults:

Voltage glitching

Clock glitching

Laser fault injection

Electromagnetic fault injection

Used to:

Bypass security checks

Extract secrets via faulty outputs

# Optical & Photonic Attacks

Use light emission from transistors

Requires sophisticated lab equipment

Highly invasive but extremely precise

# Dataset & Experimental Setup

Platform: STM32F3 microcontroller

Frequency: 7.372 MHz

Implementation: C reference ML-KEM

Dataset includes:

Power traces

Ciphertext coefficients

Known intermediate values

This makes the study reproducible and practical, not theoretical.
