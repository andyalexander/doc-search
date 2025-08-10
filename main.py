import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoTokenizer, AutoModel
from pdf2image import convert_from_path
from PIL import Image
import numpy as np
from huggingface_hub import hf_hub_download
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "mps"


# -------- Step 1: Model loading with progress --------
# print("Downloading & loading SmolDocling...")
# doc_model_name = "ds4sd/SmolDocling-256M-preview"
# doc_processor = AutoProcessor.from_pretrained(doc_model_name, use_fast=True, truncation=True)
# doc_model = AutoModelForImageTextToText.from_pretrained(doc_model_name).to(device)
# doc_model.eval()  # Set model to evaluation mode

# -------- 1. Load RolmOCR for OCR --------
print("Loading RolmOCR model...")
ocr_model_name = "reducto/RolmOCR"
ocr_processor = AutoProcessor.from_pretrained(ocr_model_name, use_fast=True, truncation=True)
ocr_model = AutoModelForImageTextToText.from_pretrained(ocr_model_name).to(device)


# -------- 2. Load text embedding model --------
print("Downloading & loading text embedding model...")
text_emb_model_name = "BAAI/bge-large-en-v1.5"
text_tokenizer = AutoTokenizer.from_pretrained(text_emb_model_name, use_fast=True)
text_model = AutoModel.from_pretrained(text_emb_model_name).to(device)
text_model.eval()

# -------- 3. Load image embedding model --------
print("Downloading & loading image embedding model...")
img_emb_model_name = "openai/clip-vit-large-patch14"
clip_processor = AutoProcessor.from_pretrained(img_emb_model_name, use_fast=True)
clip_model = AutoModel.from_pretrained(img_emb_model_name).to(device)
clip_model.eval()

# -------- Helper: Extract text from page image --------
def extract_text_from_image(image: Image.Image) -> str:
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "Extract all the content from this page as plain text."}
        ]}
    ]
    prompt = ocr_processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = ocr_processor(text=prompt, images=[image], return_tensors="pt").to(device)
    with torch.no_grad():
        generated_ids = ocr_model.generate(**inputs, max_new_tokens=8192)
    prompt_len = inputs.input_ids.shape[1]
    trimmed = generated_ids[:, prompt_len:]
    text = ocr_processor.batch_decode(trimmed, skip_special_tokens=False)[0].lstrip()
    # Clean the output
    # text = text.replace("<end_of_utterance>", "").strip()

    return text

# -------- Helper: Get text embedding --------
def get_text_embedding(text: str):
    inputs = text_tokenizer(text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        outputs = text_model(**inputs)
    emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
    print(f"Text embedding shape: {emb.shape}")
    return emb

# -------- Helper: Get image embedding --------
def get_image_embedding(image: Image.Image):
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = clip_model.get_image_features(**inputs)
    emb = outputs.cpu().numpy()[0]
    print(f"Image embedding shape: {emb.shape}")
    return emb

# -------- Combine embeddings --------
def combine_embeddings(text_emb, image_emb, method="concat"):
    if method == "concat":
        return np.concatenate([text_emb, image_emb]).tolist()
    elif method == "average":
        return ((text_emb + image_emb) / 2).tolist()
    else:
        raise ValueError("Unknown method")

# -------- Main: Process PDF --------
def pdf_to_multimodal_embeddings(pdf_path, method="concat"):
    images = convert_from_path(pdf_path, dpi=150)
    all_page_embeddings = []

    for page_num, image in enumerate(tqdm(images, desc="Processing pages")):
        # Extract text
        text = extract_text_from_image(image)

        # Generate embeddings
        text_emb = get_text_embedding(text)
        image_emb = get_image_embedding(image)

        # Combine
        final_emb = combine_embeddings(text_emb, image_emb, method=method)

        all_page_embeddings.append({
            "page": page_num + 1,
            "text": text,
            "embedding": final_emb
        })

    return all_page_embeddings

# -------- Example usage --------
if __name__ == "__main__":
    pdf_path = "your_document.pdf"
    embeddings = pdf_to_multimodal_embeddings(pdf_path)

    print(f"Generated {len(embeddings)} embeddings.")
    print("First page vector length:", len(embeddings[0]['embedding']))
