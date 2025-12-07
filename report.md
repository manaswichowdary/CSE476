# CSE476 Final Project Report: Inference-Time Agent

## Agent Overview
This agent solves reasoning tasks (Math and Coding) using a robust "Composite Strategy" that combines three advanced inference techniques: Chain of Thought (CoT), Self-Consistency (Majority Voting), and Multi-step Reflection.

## Implementation Details

### Architecture
- **Client**: `src/client.py` handles API comms.
- **Agent**: `src/agent.py` routes `math` -> Composite Strategy and `coding` -> Reflection.
- **Composite Strategy**: `src/strategies.py:composite_math_strategy`.

### Inference Strategies

#### 1. Composite Strategy (Max Accuracy)
Designed for Math/Logic. It chains two powerful techniques:
1.  **Exploration (Self-Consistency)**:
    -   Generates $N=5$ parallel solutions with high temperature.
    -   Majority Vote selects the most robust candidate.
2.  **Verification (Reflection)**:
    -   Takes the SC winner.
    -   Critiques it using strict prompts ("Check for extraneous roots", "Verify constraints").
    -   If verified correct, returns immediately. Else, attempts to fix.

#### 2. Chain of Thought (CoT)
Baseline strategy with domain-specific prompts (Algebra, Coding best practices).

#### 3. Reflection / Self-Correction
Used for Coding and as the second stage of the Composite Strategy.
-   **Loop**: Generate -> Critique -> Fix (Max 3 steps).
-   **Early Stopping**: Stops if critique confirms correctness.

## Performance & efficiency
-   **Accuracy**:
    -   Verified on dev subset: 3/5 Correct on hard math problems (Problems 2, 3, 5).
    -   Solved complex problems involving inequality constraints and extraneous algebraic roots.
-   **Efficiency**:
    -   Composite Strategy uses ~6-8 calls (5 for SC, 1-3 for Reflection).
    -   Well within the **20 calls/question** limit.

## How to Run
1.  **Generate Test Set Predictions** (Final Submission):
    ```bash
    python3 generate_submission.py
    ```
    (Supports resume and incremental saving)

2.  **Verify Format**:
    ```bash
    python3 verify_format.py
    ```
