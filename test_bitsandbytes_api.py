"""Check BitsAndBytesConfig from transformers."""
try:
    from transformers import BitsAndBytesConfig
    config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype='float16')
    print("BitsAndBytesConfig from transformers:", config)
except Exception as e:
    print(f"Error: {e}")
