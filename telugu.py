from transformers import AutoProcessor, AutoModelForCTC

def telugu_model():
    processor = AutoProcessor.from_pretrained("Harveenchadha/vakyansh-wav2vec2-telugu-tem-100")
    model = AutoModelForCTC.from_pretrained("Harveenchadha/vakyansh-wav2vec2-telugu-tem-100")
    return model,processor 