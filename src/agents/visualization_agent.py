import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from typing import Dict, List, Optional, Union, Any, Tuple
from src.llms.openai_llm import OpenAILLM
from src.llms.gemini_llm import GeminiLLM

class VisualizationAgent:
    """
    Agent responsible for generating data visualizations.
    """
    
    def __init__(self, llm_type: str = "openai", model: Optional[str] = None):
        """
        Initialize the Visualization Agent.
        
        Args:
            llm_type: Type of LLM to use (openai, gemini, huggingface)
            model: Specific model name (if None, uses default)
        """
        self.llm_type = llm_type.lower()
        
        # Initialize the appropriate LLM
        if self.llm_type == "openai":
            self.llm = OpenAILLM(model=model if model else "gpt-3.5-turbo")
        elif self.llm_type == "gemini":
            self.llm = GeminiLLM(model=model if model else "gemini-pro")
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")
    
    def get_data_description(self, df: pd.DataFrame) -> str:
        """
        Generate a description of the dataframe for the LLM.
        
        Args:
            df: The DataFrame to describe
            
        Returns:
            Text description of the DataFrame
        """
        # Get basic dataframe info
        num_rows, num_cols = df.shape
        columns = df.columns.tolist()
        dtypes = df.dtypes.to_dict()
        
        # Get sample data
        sample = df.head(5).to_string()
        
        # Prepare the description
        description = f"""
        DataFrame Summary:
        - Number of rows: {num_rows}
        - Number of columns: {num_cols}
        - Columns: {columns}
        - Data types: {dtypes}
        
        Sample data (first 5 rows):
        {sample}
        """
        
        return description
    
    def suggest_visualization(self, df: pd.DataFrame, query: str) -> Dict[str, Any]:
        """
        Suggest an appropriate visualization for the query.
        
        Args:
            df: The DataFrame to visualize
            query: The user's business query
            
        Returns:
            Dictionary with visualization suggestions
        """
        # Generate a description of the data
        data_description = self.get_data_description(df)
        
        # Get visualization suggestion from LLM
        viz_suggestion = self.llm.suggest_visualization(data_description, query)
        
        return viz_suggestion
    
    def create_visualization(self, df: pd.DataFrame, visualization_request: str) -> Tuple[plt.Figure, str]:
        """
        Create a visualization based on the request.
        
        Args:
            df: The DataFrame to visualize
            visualization_request: Description of the requested visualization
            
        Returns:
            Tuple of (matplotlib figure, generated code)
        """
        # Generate a description of the data
        data_description = self.get_data_description(df)
        
        # Get code for visualization from LLM
        code = self.llm.generate_code(data_description, visualization_request)
        
        # Create a new figure
        fig = plt.figure(figsize=(10, 6))
        
        try:
            # Execute the generated code
            # For safety, we're using a restricted namespace
            namespace = {
                'pd': pd, 
                'plt': plt, 
                'sns': sns, 
                'np': pd.np, 
                'df': df,
                'fig': fig
            }
            
            exec(code, namespace)
            
            # Apply some styling
            plt.tight_layout()
            
            return fig, code
        except Exception as e:
            # If there's an error, create a simple error visualization
            plt.close(fig)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"Error creating visualization:\n{str(e)}", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12, color='red')
            plt.tight_layout()
            
            return fig, f"# Error in generated code:\n# {str(e)}\n\n{code}"
    
    def save_visualization(self, fig: plt.Figure, filename: str = "visualization.png") -> str:
        """
        Save the visualization to a file.
        
        Args:
            fig: The matplotlib figure
            filename: The filename to save to
            
        Returns:
            Path to the saved file
        """
        # Create directory if it doesn't exist
        import os
        os.makedirs("output", exist_ok=True)
        
        # Save the figure
        filepath = os.path.join("output", filename)
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        
        return filepath
    
    def get_image_base64(self, fig: plt.Figure) -> str:
        """
        Convert matplotlib figure to base64 encoded string.
        
        Args:
            fig: The matplotlib figure
            
        Returns:
            Base64 encoded string of the image
        """
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        
        return img_str
