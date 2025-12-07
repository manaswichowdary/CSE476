
import json
from src.strategies import chain_of_thought, reflection_strategy, self_consistency, composite_math_strategy

class Agent:
    def __init__(self, model_name: str = "bens_model"):
        self.model_name = model_name

    def _infer_domain(self, prompt: str) -> str:
        """
        Infers the domain of the problem based on keywords.
        """
        prompt_lower = prompt.lower()
        
        # Coding keywords
        coding_keywords = ['python', 'code', 'function', 'algorithm', 'class', 'programming', 'def ', 'return', 'import', 'java', 'c++']
        if any(k in prompt_lower for k in coding_keywords):
            return 'coding'
            
        # Math keywords
        math_keywords = ['calculate', 'compute', 'solve', 'equation', 'how many', 'probability', 'value of', 'remainder', 'sum', 'integral', 'derivative', '$', '=', 'find the']
        # Check for digits/math symbols + keywords
        if any(k in prompt_lower for k in math_keywords) and any(c.isdigit() for c in prompt):
            return 'math'
            
        return 'general'

    def solve(self, problem: dict) -> str:
        """
        Solves a single problem instance.
        problem: dict with keys 'input', 'domain', etc.
        """
        prompt = problem.get('input', '')
        # Infer domain if validation/test set doesn't have it
        domain = problem.get('domain')
        if not domain or domain == 'general':
            # Try to infer a more specific domain even if labeled "general" (if strictly generic)
            # But primarily for missing labels
            inferred = self._infer_domain(prompt)
            if inferred != 'general':
                domain = inferred
            elif not domain:
                 domain = 'general'
        
        print(f"Solving problem in domain: {domain}")
        
        try:
            if domain == 'math':
                # Use composite strategy for maximum accuracy
                return composite_math_strategy(prompt, model=self.model_name)
            elif domain == 'coding':
                # Coding benefits from standard CoT with code-specific prompts, or reflection.
                return reflection_strategy(prompt, model=self.model_name, domain=domain)
            else:
                return chain_of_thought(prompt, model=self.model_name, domain=domain)
        except Exception as e:
            print(f"Error during solving: {e}")
            return "Error"
