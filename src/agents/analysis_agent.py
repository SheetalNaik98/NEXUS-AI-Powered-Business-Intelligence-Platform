from src.llms.openai_llm import OpenAILLM
from src.llms.gemini_llm import GeminiLLM
from src.llms.anthropic_llm import AnthropicLLM

class AnalysisAgent:
    """
    Agent responsible for analyzing business data and providing insights.
    """
    
    def __init__(self, llm_type: str = "openai", model: Optional[str] = None):
        """
        Initialize the Analysis Agent.
        
        Args:
            llm_type: Type of LLM to use (openai, anthropic, gemini, huggingface)
            model: Specific model name (if None, uses default)
        """
        self.llm_type = llm_type.lower()
        
        # Initialize the appropriate LLM
        if self.llm_type == "openai":
            self.llm = OpenAILLM(model=model if model else "gpt-3.5-turbo")
        elif self.llm_type == "anthropic":
            self.llm = AnthropicLLM(model=model if model else "claude-2")
        elif self.llm_type == "gemini":
            self.llm = GeminiLLM(model=model if model else "gemini-pro")
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")
    
    # Rest of the class remains the same
