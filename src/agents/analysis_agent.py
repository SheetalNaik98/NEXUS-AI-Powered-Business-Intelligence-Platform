import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from src.llms.openai_llm import OpenAILLM
from src.llms.gemini_llm import GeminiLLM

class AnalysisAgent:
    """
    Agent responsible for analyzing business data and providing insights.
    """
    
    def __init__(self, llm_type: str = "openai", model: Optional[str] = None):
        """
        Initialize the Analysis Agent.
        
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
        
        # Get summary statistics for numeric columns
        numeric_summary = df.describe().to_string()
        
        # Prepare the description
        description = f"""
        DataFrame Summary:
        - Number of rows: {num_rows}
        - Number of columns: {num_cols}
        - Columns: {columns}
        - Data types: {dtypes}
        
        Sample data (first 5 rows):
        {sample}
        
        Summary statistics for numeric columns:
        {numeric_summary}
        """
        
        return description
    
    def analyze(self, df: pd.DataFrame, query: str) -> str:
        """
        Analyze the data based on the user's query.
        
        Args:
            df: The DataFrame to analyze
            query: The user's business query
            
        Returns:
            Analysis result as text
        """
        # Generate a description of the data
        data_description = self.get_data_description(df)
        
        # Get analysis from LLM
        analysis = self.llm.analyze_data(data_description, query)
        
        return analysis
    
    def predict(self, df: pd.DataFrame, prediction_query: str) -> str:
        """
        Generate predictions based on the data.
        
        Args:
            df: The DataFrame with historical data
            prediction_query: The prediction request
            
        Returns:
            Prediction result as text
        """
        # Generate a description of the data
        data_description = self.get_data_description(df)
        
        # Generate a summary of historical data patterns
        # This could be enhanced with actual time series analysis
        time_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower() or 'year' in col.lower() or 'month' in col.lower()]
        
        if time_columns:
            time_col = time_columns[0]
            # Basic trend analysis
            if pd.api.types.is_numeric_dtype(df[time_col]):
                df = df.sort_values(by=time_col)
            elif pd.api.types.is_datetime64_dtype(df[time_col]):
                df = df.sort_values(by=time_col)
            
            # Find numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col != time_col]
            
            if numeric_cols:
                # Generate summary of patterns
                historical_summary = f"The dataset contains time series data with time indicator '{time_col}' and numeric metrics {numeric_cols}."
                
                # Check for seasonality and trends
                for col in numeric_cols[:3]:  # Limit to first 3 for brevity
                    if len(df) > 3:
                        # Simple trend detection
                        first_half_mean = df.iloc[:len(df)//2][col].mean()
                        second_half_mean = df.iloc[len(df)//2:][col].mean()
                        
                        if second_half_mean > first_half_mean * 1.1:
                            historical_summary += f"\n- {col} shows an upward trend (increased by {((second_half_mean/first_half_mean)-1)*100:.1f}%)."
                        elif first_half_mean > second_half_mean * 1.1:
                            historical_summary += f"\n- {col} shows a downward trend (decreased by {((first_half_mean/second_half_mean)-1)*100:.1f}%)."
                        else:
                            historical_summary += f"\n- {col} shows a relatively stable pattern."
            else:
                historical_summary = "The dataset contains time-based data, but no clear numeric metrics for prediction."
        else:
            historical_summary = "The dataset does not contain clear time indicators for time series analysis."
        
        # Get prediction from LLM
        prediction = self.llm.predict_trends(data_description, historical_summary, prediction_query)
        
        return prediction
    
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
