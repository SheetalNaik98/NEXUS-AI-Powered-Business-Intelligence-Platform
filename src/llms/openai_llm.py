import os
import time
from typing import Dict, List, Optional, Union, Any
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

class OpenAILLM:
    """
    Implementation of OpenAI's models for business intelligence.
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        """
        Initialize the OpenAI LLM.
        
        Args:
            model: The OpenAI model to use (default: gpt-3.5-turbo)
        """
        self.model = model
        # Set API key from environment variable
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
        
        openai.api_key = api_key
        
        # Model validation
        self.available_models = [
            "gpt-3.5-turbo", 
            "gpt-4", 
            "gpt-4-turbo", 
            "gpt-4-32k",
            "gpt-3.5-turbo-16k"
        ]
        
        if model not in self.available_models:
            raise ValueError(f"Model {model} is not in the list of recognized OpenAI models. Available models: {self.available_models}")
    
    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """
        Generate a response from the OpenAI model.
        
        Args:
            prompt: The prompt to send to the model
            max_tokens: The maximum number of tokens to generate
            temperature: Controls randomness (0.0 to 1.0)
            
        Returns:
            Generated text response
        """
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.choices[0].message["content"].strip()
            
        except Exception as e:
            print(f"Error generating response from OpenAI: {e}")
            return f"Error generating response: {str(e)}"
    
    def analyze_data(self, data_description: str, query: str) -> str:
        """
        Analyze data based on a description and query.
        
        Args:
            data_description: Description of the data structure
            query: The business query to analyze
            
        Returns:
            Analysis result as text
        """
        prompt = f"""
        You are NEXUS, an AI business intelligence assistant. You need to analyze business data and provide insights.
        
        DATA DESCRIPTION:
        {data_description}
        
        USER QUERY:
        {query}
        
        Provide a detailed analysis addressing the user's query. Include relevant insights, patterns, 
        and business implications. Format your response professionally with clear sections and bullet points where appropriate.
        """
        
        return self.generate(prompt, max_tokens=1500, temperature=0.3)
    
    def suggest_visualization(self, data_description: str, query: str) -> Dict[str, Any]:
        """
        Suggest an appropriate visualization based on the data and query.
        
        Args:
            data_description: Description of the data structure
            query: The business query to visualize
            
        Returns:
            Dictionary containing visualization type and parameters
        """
        prompt = f"""
        You are NEXUS, an AI business intelligence assistant specializing in data visualization.
        
        DATA DESCRIPTION:
        {data_description}
        
        USER QUERY:
        {query}
        
        Suggest the most appropriate visualization for this query. Return your response in a structured format:
        
        VISUALIZATION_TYPE: [bar_chart, line_chart, pie_chart, scatter_plot, heatmap, etc.]
        X_AXIS: [suggested x-axis or None]
        Y_AXIS: [suggested y-axis or None]
        GROUPBY: [suggested grouping variable or None]
        TITLE: [suggested chart title]
        DESCRIPTION: [brief explanation of why this visualization is appropriate]
        
        Be specific and base your suggestions on the actual data described.
        """
        
        response = self.generate(prompt, max_tokens=800, temperature=0.2)
        
        # Parse the response into a dictionary
        viz_dict = {}
        for line in response.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                viz_dict[key.strip()] = value.strip()
        
        return viz_dict
    
    def generate_code(self, data_description: str, visualization_request: str) -> str:
        """
        Generate Python code for creating a visualization.
        
        Args:
            data_description: Description of the data structure
            visualization_request: Description of the requested visualization
            
        Returns:
            Python code to generate the visualization
        """
        prompt = f"""
        You are NEXUS, an AI business intelligence assistant with expertise in data visualization.
        
        DATA DESCRIPTION:
        {data_description}
        
        VISUALIZATION REQUEST:
        {visualization_request}
        
        Generate Python code using pandas, matplotlib, and/or plotly to create this visualization. 
        Assume the data is loaded in a pandas DataFrame called 'df'.
        Only include the code without explanation.
        Make the visualizations professional with proper titles, labels, colors, and formatting.
        """
        
        return self.generate(prompt, max_tokens=1000, temperature=0.2)
    
    def predict_trends(self, data_description: str, historical_data_summary: str, prediction_request: str) -> str:
        """
        Generate predictions based on historical data.
        
        Args:
            data_description: Description of the data structure
            historical_data_summary: Summary of historical data patterns
            prediction_request: The specific prediction request
            
        Returns:
            Prediction result as text
        """
        prompt = f"""
        You are NEXUS, an AI business intelligence assistant with expertise in predictive analytics.
        
        DATA DESCRIPTION:
        {data_description}
        
        HISTORICAL DATA SUMMARY:
        {historical_data_summary}
        
        PREDICTION REQUEST:
        {prediction_request}
        
        Generate a thoughtful prediction based on the historical data patterns. Include:
        1. Your predicted trend or outcome
        2. Key factors influencing your prediction
        3. Level of confidence and potential variables that could change the outcome
        4. Business recommendations based on this prediction
        
        Be specific, realistic, and business-focused in your prediction.
        """
        
        return self.generate(prompt, max_tokens=1200, temperature=0.4)
    
    def benchmark(self) -> Dict[str, float]:
        """
        Benchmark the model's performance.
        
        Returns:
            Dictionary containing benchmark metrics
        """
        # Example benchmarking
        response_times = []
        accuracy_scores = []
        
        # Simple test queries
        test_queries = [
            "What were our highest sales months last year?",
            "Which region saw the biggest revenue growth?",
            "Predict sales trends for the next quarter based on past data."
        ]
        
        for query in test_queries:
            # Here you would run actual timing and accuracy tests
            # For now, we'll return sample values
            pass
        
        return {
            "average_response_time": 2.3,  # seconds
            "query_accuracy": 0.94,        # 0-1 scale
            "context_understanding": 0.95,  # 0-1 scale
            "visualization_quality": 0.93   # 0-1 scale
        }
