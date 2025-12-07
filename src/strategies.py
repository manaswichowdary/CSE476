"""
Inference strategies for the agent.
"""
import re
from collections import Counter
from src.client import call_model_chat_completions

def clean_response(text: str) -> str:
    """Helper to clean response text."""
    return (text or "").strip()

def normalize_text(s: str) -> str:
    s = (str(s) or "").strip().lower()
    s = re.sub(r"[^\w\s\-']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def extract_answer_candidate(text: str) -> str:
    """
    Attempts to extract the final answer from text.
    Prioritizes boxed answers.
    """
    # Pattern 1: Latex boxed
    boxed = re.search(r"\\boxed\{(.*?)\}", text)
    if boxed:
        return normalize_text(boxed.group(1))
    
    # Pattern 2: "Answer: X" at the end
    ans_match = re.search(r"Answer:\s*(.*)$", text, re.IGNORECASE | re.DOTALL)
    if ans_match:
        return normalize_text(ans_match.group(1))
        
    return normalize_text(text)

def chain_of_thought(prompt: str, model: str = "bens_model", system_prompt: str = None, temperature: float = 0.0, domain: str = "general") -> str:
    """Standard Chain of Thought strategy."""
    if system_prompt is None:
        if domain == "math":
            system_prompt = (
                "You are an intelligent math solver. "
                "1. Use substitution to simplify complex equations. "
                "2. Solve algebraically. "
                "3. CHECK FOR EXTRANEOUS SOLUTIONS (especially with square roots). "
                "4. Verify all constraints. "
                "End your answer with \\boxed{answer}."
            )
        elif domain == "coding":
            system_prompt = (
                "You are an expert software engineer. "
                "Write clean, efficient, and correct code. "
                "Think about edge cases. "
                "Return the final answer as code or explanation as requested."
            )
        else:
            system_prompt = (
                "You are a helpful assistant. Think step by step. "
                "End your answer with \\boxed{answer} if applicable."
            )
    
    cot_prompt = f"{prompt}\n\nLet's think step by step."
    
    result = call_model_chat_completions(cot_prompt, system=system_prompt, model=model, temperature=temperature)
    if result['ok']:
        return clean_response(result['text'])
    else:
        return f"Error: {result['error']}"

def reflection_strategy(prompt: str, model: str = "bens_model", max_steps: int = 3, domain: str = "general", initial_ans: str = None) -> str:
    """
    Multistep Reflection strategy.
    
    Args:
        initial_ans: If provided, skips the initial generation step and starts critiquing this answer.
    """
    print(f"  Running Reflection (max_steps={max_steps}) for {domain}...")
    
    # Initial Attempt (if not provided)
    if initial_ans:
        current_ans = initial_ans
        print("    Using provided initial answer from previous stage.")
    else:
        current_ans = chain_of_thought(prompt, model=model, domain=domain)
    
    for step in range(max_steps):
        # Critique
        critique_prompt = f"Original Question: {prompt}\n\nCurrent Answer: {current_ans}\n\nCritique the above answer. Is it strictly correct? Verify the reasoning. If it is correct, start your response with 'CORRECT'. If incorrect, explain why."
        critique_res = call_model_chat_completions(critique_prompt, system="You are a strict critical reviewer.", model=model)
        critique_text = clean_response(critique_res.get('text', ''))
        
        print(f"    Step {step+1} Critique: {critique_text[:100]}...")
        
        # Improved stop condition
        normalized_critique = normalize_text(critique_text)
        if normalized_critique.startswith("correct") or "is correct" in normalized_critique[:50] or "**correct**" in critique_text.lower()[:50]:
             print("    Critique passed. Stopping reflection.")
             break
        
        # Improvement Step
        improvement_prompt = f"Original Question: {prompt}\n\nPrevious Answer: {current_ans}\n\nCritique: {critique_text}\n\nBased on the critique, provide a corrected explanation and answer. Verify your algebra. End with \\boxed{{answer}}."
        imp_res = call_model_chat_completions(improvement_prompt, system="You are a helpful solver. Fix the answer.", model=model)
        
        if imp_res['ok']:
            current_ans = clean_response(imp_res['text'])
        else:
            break # API error, keep previous
            
    return current_ans

def self_consistency(prompt: str, model: str = "bens_model", n: int = 3, domain: str = "general") -> str:
    """
    Self-consistency with robust voting.
    """
    candidates = []
    full_responses = []
    
    print(f"  Running Self-Consistency (n={n})...")
    
    for i in range(n):
        # High temperature for diversity
        resp = chain_of_thought(prompt, model=model, temperature=0.7, domain=domain)
        full_responses.append(resp)
        
        # Extract candidate
        cand = extract_answer_candidate(resp)
        if cand:
            candidates.append(cand)
    
    if not candidates:
        return full_responses[0] if full_responses else "Error"
        
    # Vote on candidates
    c = Counter(candidates)
    most_common_cand, count = c.most_common(1)[0]
    
    print(f"    Voting results: {c.most_common()}")
    print(f"    Winner: {most_common_cand} with {count}/{n} votes")
    
    # Return the full response corresponding to the winner to preserve context/reasoning
    for resp in full_responses:
        if extract_answer_candidate(resp) == most_common_cand:
            return resp
            
    return full_responses[0] # Fallback

def composite_math_strategy(prompt: str, model: str = "bens_model") -> str:
    """
    Composite strategy:
    1.  Run Self-Consistency (n=5) to get a robust initial candidate.
    2.  Run Reflection on that candidate to verify and fix any remaining errors.
    """
    print("Running Composite Strategy: Self-Consistency -> Reflection")
    
    # Stage 1: Exploration via SC
    sc_ans = self_consistency(prompt, model=model, n=5, domain="math")
    
    # Stage 2: Verification via Reflection
    # We pass the SC winner as the 'initial_ans' to reflection
    final_ans = reflection_strategy(prompt, model=model, max_steps=3, domain="math", initial_ans=sc_ans)
    
    return final_ans
