import abc
import time
import torch
import openai
from decouple import config
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


openai.api_key = config('API_KEY')
openai.api_base = config('API_BASE')
openai.api_type = config('API_TYPE')
openai.api_version = config('API_VERSION')


class GPT3:
    """Class for interacting with the OpenAI API."""
    def __init__(self):
        self.deployment_id = 'text-davinci-003'
        self.deployment_id = 'd365-sales-davinci003'

    def submit_request(self, prompt, temperature=0.7, max_tokens=1024, n=1, split_by=None):
        """Submit a request to the OpenAI API."""
        error_counter = 0
        
        while(True):
            try:
                response = openai.Completion.create(engine=self.deployment_id,
                                                    prompt=prompt,
                                                    max_tokens=max_tokens,
                                                    temperature=temperature,
                                                    n=n
                                                    )
                break

            except Exception as anything:
                if anything.args[0] == 'string indices must be integers' or 'The response was filtered' in anything.args[0]:
                    return {'choices': [{'text': ''}]}

                time.sleep(1)
                error_counter += 1
                if error_counter > 10:
                    raise anything

        response = [res['text'].strip() for res in response['choices']]
        return response


class LLamaV2:
    def __init__(self, model_size=7, model_type='chat', device_map="auto"):
        """
        @param model_size: choose from 7b, 13b and 70b
        @param model_type: regular or chat
        """
        assert model_size == 7 or model_size == 13 or model_size == 70
        assert model_type == 'base' or model_type == 'chat'
    
        print(f'llama model_size: {model_size}, model_type: {model_type}')
        

        if model_type == 'chat':
            MODEL = f'meta-llama/Llama-2-{model_size}b-chat-hf'
        else:
            MODEL = f'meta-llama/Llama-2-{model_size}b-hf'
        
        self.model = AutoModelForCausalLM.from_pretrained(MODEL, 
                                                          low_cpu_mem_usage=True, 
                                                          device_map=device_map,  
                                                          torch_dtype=torch.float16)
        
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.model_config = GenerationConfig.from_model_config(self.model.config)


    def submit_request(self, prompt, temperature=0.6, max_length=300, top_p=0.9, split_by='Question:'):

        self.model_config.temperature = temperature
        self.model_config.top_p = top_p
        self.model_config.max_new_tokens = max_length
        self.model_config.do_sample = True


        encoded = self.tokenizer(prompt, return_tensors="pt")
        generated = self.model.generate(encoded["input_ids"].to(self.model.device), 
                                        generation_config=self.model_config)[0]

        decoded = self.tokenizer.decode(generated, 
                                        skip_special_tokens=True, 
                                        clean_up_tokenization_spaces=False)
      
        response = decoded.strip().split(prompt)
        response = [res.strip() for res in response if res != ''][0]
        response = response.split(split_by)[0].strip()
        response = response.split('\n')[0].strip()
        return [response]


class Falcon:
    def __init__(self):
        MODEL = "tiiuae/falcon-40b-instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, device_map='auto',
                                                     trust_remote_code=True, torch_dtype=torch.bfloat16)
        self.model_config = GenerationConfig.from_model_config(self.model.config)

        self.model_config.do_sample = True

    def submit_request(self, prompt, max_tokens=512, top_k=10):

        self.model_config.max_new_tokens = max_tokens
        self.model_config.top_k = top_k

        encoded = self.tokenizer(prompt, return_tensors="pt")
        generated = self.model.generate(encoded["input_ids"].cuda(), generation_config=self.model_config, pad_token_id=self.tokenizer.eos_token_id)[0]

        decoded = self.tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        split_by = prompt.split('\n')[-1].strip()
        response = [decoded.split(split_by)[-1].strip()]

        print(f'decoded:\n{decoded}\n')
        print(f'response:\n{response}\n')
        return response


class EmbeddingModel:
    @abc.abstractmethod
    def submit_embedding_request(self, text):
        """
        submit embedding request by the given model
        @param text: text to be embedded
        @return: embedding vector of the text
        """
        pass


class GPTEmbedding(EmbeddingModel):
    """Class for interacting with the OpenAI API Embedding model."""
    def __init__(self):
        self.deployment_id = 'text-embedding-ada-002'

    def submit_embedding_request(self, text):
        """Submit a request to the OpenAI API."""
        error_counter = 0

        while (True):
            try:
                response = openai.Embedding.create(engine=self.deployment_id, input=text)
                break

            except Exception as anything:
                time.sleep(1)
                error_counter += 1
                
                if error_counter > 10:
                    raise anything

        response = response['data'][0]['embedding']
        return response


class SBert(EmbeddingModel):
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L12-v2')

    def submit_embedding_request(self, text):
        response = self.model.encode([text], convert_to_tensor=True)

        return response


class E5(EmbeddingModel):
    def __init__(self):
        self.model = SentenceTransformer('intfloat/e5-large-v2')

    def submit_embedding_request(self, text):
        response = self.model.encode([text], convert_to_tensor=True)

        return response


class BERTembedding(EmbeddingModel):
    def __init__(self):
        self.model = SentenceTransformer("bert-base-uncased")

    def submit_embedding_request(self, text):
        response = self.model.encode([text], convert_to_tensor=True)

        return response
