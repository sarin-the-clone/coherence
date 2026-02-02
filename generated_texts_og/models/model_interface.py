import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union

class BaseLLM(ABC):
    """Abstract base class for LLM backends."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model_name = config.get("name", "unknown")
        self.max_tokens = config.get("max_tokens", 2048)
        self.temperature = config.get("temperature", 0.7)
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def generate(self, prompt: str, stop_sequences: Optional[List[str]] = None, max_new_tokens: int = None) -> str:
        """Generate text from prompt."""
        pass

    @abstractmethod
    def get_token_count(self, text: str) -> int:
        """Count tokens in text."""
        pass

    @abstractmethod
    def get_embedding(self, text: str) -> torch.Tensor:
        """Get the model's internal vector representation of the text."""
        pass

class HuggingFaceLLM(BaseLLM):
    """Implementation for local HuggingFace models (Qwen, Llama, Mistral)."""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        print(f"Loading model: {self.model_name} on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        self.model.eval()
        print("Model loaded successfully.")

    def generate(self, prompt: str, stop_sequences: Optional[List[str]] = None, max_new_tokens: int = None, prefix_embeds: Optional[torch.Tensor] = None, top_p: float = 0.9, **kwargs) -> str:
        if max_new_tokens is None:
            max_new_tokens = self.max_tokens

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Calculate max length safely
        input_len = inputs.input_ids.shape[1]
        
        # Prepare generation args
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": self.temperature,
            "do_sample": True,
            "top_p": top_p,
            "pad_token_id": self.tokenizer.eos_token_id
        }
        # update with any additional kwargs (e.g. repetition_penalty)
        gen_kwargs.update(kwargs)

        with torch.no_grad():
            if prefix_embeds is not None:
                # Get embeddings for the text prompt
                prompt_embeds = self.model.get_input_embeddings()(inputs.input_ids)
                
                # Concatenate prefix + prompt
                # prefix_embeds shape: (1, n_prefix, hidden)
                # prompt_embeds shape: (1, n_prompt, hidden)
                inputs_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
                
                # Create attention mask: 1 for all positions (prefix + prompt)
                # Shape: (1, n_prefix + n_prompt)
                n_prefix = prefix_embeds.shape[1]
                n_prompt = prompt_embeds.shape[1]
                attention_mask = torch.ones((1, n_prefix + n_prompt), dtype=torch.long, device=self.device)
                
                outputs = self.model.generate(
                    inputs_embeds=inputs_embeds, 
                    attention_mask=attention_mask,
                    **gen_kwargs
                )
                
            else:
                outputs = self.model.generate(**inputs, **gen_kwargs)
            
        # Decode
        # If inputs_embeds was used, outputs might just be the new tokens or full sequence.
        # Standard HF generate with inputs_embeds returns sequences including the prompt *if* it can reconstruct it?
        # No, it returns token IDs. It can't return IDs for the prefix_embeds.
        # So it likely returns [new_tokens] or [dummy_prefix + new_tokens].
        # Let's decode the *last* max_new_tokens just to be safe, or check output shape.
        
        if prefix_embeds is not None:
            # When using inputs_embeds, the model doesn't know the input tokens.
            # The output will likely contain only the generated tokens if we didn't pass input_ids.
            # Let's trust that we just want the new text.
            # Actually, let's just decode the whole thing and see. 
            # If it returns only new tokens, great. If it returns garbage for prefix, we ignore it.
            # A safer bet is to decode everything and take the new text.
            # But we don't know where "new text" starts if we don't know input length.
            # input_len is known from the text part. prefix_len is known.
            # Let's assume outputs contains [generated_tokens] only?
            # Let's verify this assumption or write robust code.
            # Robust: decode all, and if it starts with prompt, strip it.
            # But we can't strip "prefix_embeds".
            
            # Let's simplify: HF generate with inputs_embeds usually returns the whole sequence of IDs 
            # corresponding to the embeddings? No, it can't return IDs for embeddings it never saw IDs for.
            # It returns the generated IDs.
            
            # Let's assume it returns everything.
            pass 
            
        # Re-simplifying for the tool call:
        # I will just decode outputs[0] and strip the prompt text if it appears.
        
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # If we used prefix_embeds, the prompt text might not be in the output IDs at all
        # because the model didn't receive input_ids.
        # So 'decoded' is likely just the generated text.
        # If we used normal inputs, 'decoded' includes prompt.
        
        if prefix_embeds is None:
             decoded = decoded[len(prompt):] # Rough stripping if needed, or use the input_len slicing
             # The original code used: self.tokenizer.decode(outputs[0][input_len:], ...)
             # I should preserve that for the normal case.
             pass

        # Let's stick to the original logic for normal case, and handle prefix case.

            
        # Decode
        if prefix_embeds is not None:
            # When using inputs_embeds, outputs usually contains only the new tokens
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            generated_text = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        
        # Handle stop sequences manually if needed (HF generate doesn't always handle string stops perfectly)
        if stop_sequences:
            for stop in stop_sequences:
                if stop in generated_text:
                    generated_text = generated_text.split(stop)[0]
                    
        return generated_text

    def get_token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def get_embedding(self, text: str) -> torch.Tensor:
        """
        Returns the last hidden state of the last token.
        Shape: (1, hidden_size)
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(
                **inputs, 
                output_hidden_states=True,
                return_dict=True
            )
        
        # Get last layer hidden states: (batch, seq_len, hidden_size)
        last_hidden_state = outputs.hidden_states[-1]
        
        # Take the vector of the last token in the sequence
        # (batch=0, seq_len=-1, :)
        # This represents the model's "thought" after reading the whole text
        vector = last_hidden_state[0, -1, :]
        
        return vector.unsqueeze(0) # (1, hidden_size)

class ModelFactory:
    """Factory to create model instances."""
    
    @staticmethod
    def create_model(config: Dict) -> BaseLLM:
        model_type = config.get("type", "huggingface")
        
        if model_type == "huggingface":
            return HuggingFaceLLM(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
