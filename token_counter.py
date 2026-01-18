"""
Token Counter Module

Handles token counting and optimization for LLM prompts.
"""

import tiktoken


class TokenCounter:
    """Handles token counting and optimization for LLM prompts."""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize token counter.
        
        Args:
            model: LLM model name to get encoding for
        """
        self.model = model
        try:
            # Map model names to their encodings
            if "gpt-4" in model or "gpt-4o" in model:
                encoding_name = "cl100k_base"
            elif "gpt-3.5" in model:
                encoding_name = "cl100k_base"
            else:
                encoding_name = "cl100k_base"  # Default
            
            self.encoding = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            print(f"Warning: Could not load encoding for {model}, using cl100k_base: {e}")
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
    
    def get_model_max_tokens(self, model: str) -> int:
        """Get maximum context tokens for a model."""
        max_tokens_map = {
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
            "gpt-4-turbo": 128000,
            "gpt-4": 8192,
            "gpt-3.5-turbo": 16385,
            "gpt-3.5-turbo-16k": 16385,
        }
        return max_tokens_map.get(model, 4096)  # Default conservative limit
    
    def optimize_content(self, content: str, max_tokens: int, reserve_tokens: int = 2000) -> str:
        """
        Optimize content to fit within token limit.
        
        Args:
            content: Content to optimize
            max_tokens: Maximum tokens allowed
            reserve_tokens: Tokens to reserve for prompt and response
            
        Returns:
            Optimized content that fits within token limit
        """
        available_tokens = max_tokens - reserve_tokens
        
        if self.count_tokens(content) <= available_tokens:
            return content
        
        # Truncate content intelligently
        lines = content.split('\n')
        optimized_lines = []
        current_tokens = 0
        
        for line in lines:
            line_tokens = self.count_tokens(line + '\n')
            if current_tokens + line_tokens > available_tokens:
                break
            optimized_lines.append(line)
            current_tokens += line_tokens
        
        result = '\n'.join(optimized_lines)
        if len(result) < len(content):
            result += f"\n\n... (truncated to fit {available_tokens} tokens)"
        
        return result
