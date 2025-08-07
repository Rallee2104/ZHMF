import requests  

class LlamaLLM:  
    """  
    LLM class using the locally deployed Llama model service via HTTP API  
    """  
    
    def __init__(self, model_args: dict, api_key: str, base_url: str, model: str):  
        """  
        Initialize a local Llama model  
        Args:  
            model_args: A dictionary of model configuration arguments  
            api_key: API key (not used for local service)  
            base_url: The base URL for the deployed model's API  
            model: The name of the deployed model  
        """  
        print("Llama init marker")  
        self.model_args = model_args  
        self.temperature = 0.7  
        self.max_input_length = 4096 * 2  
        self.max_output_length = 256
        self.model = model  
        self.base_url = base_url  

        print("Local Llama model initialized.")  

    def __call__(self, prompt: str, max_output_length: int = None) -> str:  
        """  
        Generate response for a given prompt using the local Llama service  
        Args:  
            prompt: Input prompt text  
            max_output_length: Optional max length for the output, overrides default  

        Returns:  
            Generated string response  
        """  
        max_length = max_output_length if max_output_length is not None else self.max_output_length  

        payload = {  
            "model": self.model,  
            "messages": [  
                {"role": "user", "content": prompt}
            ],  
            "max_tokens": max_length,  
            "temperature": 0.7,  
            "do_sample": True,
            "top_p": 0.95    
        }  

        try:  
            response = requests.post(f"{self.base_url}/v1/chat/completions", json=payload)  
            
            if response.status_code != 200:  
                print(f"Error response: {response.status_code} - {response.text}")  
                return ""  
                
            # Extract generated content  
            result = response.json()  
            response_text = result["choices"][0]["message"]["content"].strip()  

            return response_text  

        except Exception as e:  
            print(f"Error in generating response: {e}")  
            return ""  

    def count_tokens(self, text: str) -> int:  
        """  
        Placeholder method to count tokens in text  
        Note: Using a simple heuristic for token counting since we don't have direct tokenizer access.  
        Args:  
            text: Text for which to count tokens  

        Returns:  
            Token count (approximate)  
        """  
        return len(text) // 4  

    def truncate_text(self, text: str, max_tokens: int) -> str:  
        """  
        Truncate text to fit within the max_tokens limit  
        Args:  
            text: Input text to truncate  
            max_tokens: Maximum number of tokens  

        Returns:  
            Truncated text  
        """  
        tokens = self.count_tokens(text)  
        if tokens <= max_tokens:  
            return text  
        truncated_text = text[: max_tokens * 4]  
        return truncated_text  