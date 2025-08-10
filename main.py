import torch
import faiss
import duckdb
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoTokenizer, AutoModel

# ---------- Config ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ocr_model_name = "reducto/RolmOCR"
text_emb_name = "BAAI/bge-large-en-v1.5"
img_emb_name = "openai/clip-vit-large-patch14"
faiss_path = "document_index.faiss"
db_path = "metadata.duckdb"

# ---------- Load Models ----------
print("Loading RolmOCR...")
ocr_processor = AutoProcessor.from_pretrained(ocr_model_name, use_fast=True)
ocr_model = AutoModelForImageTextToText.from_pretrained(ocr_model_name).to(device)

print("Loading text embedding model...")
text_tokenizer = AutoTokenizer.from_pretrained(text_emb_name)
text_emb_model = AutoModel.from_pretrained(text_emb_name).to(device)

print("Loading image embedding model...")
clip_processor = AutoProcessor.from_pretrained(img_emb_name)
clip_model = AutoModel.from_pretrained(img_emb_name).to(device)


# ---------- Helper Functions ----------
def extract_text_from_image(image: Image.Image) -> str:
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "Extract the plain text from this page."}
    ]}]
    prompt = ocr_processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = ocr_processor(text=prompt, images=[image], return_tensors="pt").to(device)
    with torch.no_grad():
        gen = ocr_model.generate(**inputs, max_new_tokens=4096)
    prompt_len = inputs.input_ids.shape[1]
    trimmed = gen[:, prompt_len:]
    return ocr_processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

def get_text_embedding(text: str) -> np.ndarray:
    tokens = text_tokenizer(text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        out = text_emb_model(**tokens)
    return out.last_hidden_state.mean(dim=1).cpu().numpy()[0]

def get_image_embedding(image: Image.Image) -> np.ndarray:
    proc = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        feat = clip_model.get_image_features(**proc)
    return feat.cpu().numpy()[0]

def combine_embeddings(te, ie, method="concat"):
    te_arr, ie_arr = np.array(te), np.array(ie)
    return np.concatenate((te_arr, ie_arr)).astype("float32") if method == "concat" else ((te_arr + ie_arr) / 2).astype("float32")


# ---------- Database Setup ----------
def init_databases(embedding_size):
    # FAISS
    try:
        index = faiss.read_index(faiss_path)
    except:
        index = faiss.IndexFlatL2(embedding_size)

    # DuckDB
    con = duckdb.connect(db_path)
    con.execute("""
    CREATE TABLE IF NOT EXISTS metadata (
        id INTEGER PRIMARY KEY,
        document_class TEXT,
        filename TEXT,
        page_number INTEGER
    )
    """)
    con.close()

    return index


# ---------- Store PDF Embeddings ----------
def store_pdf(pdf_path, document_class, method="concat"):
    pages = convert_from_path(pdf_path, dpi=150)

    # Example embedding to determine dimensionality
    sample_text = get_text_embedding("test")
    sample_img = get_image_embedding(pages[0])
    sample_emb = combine_embeddings(sample_text, sample_img, method)
    index = init_databases(len(sample_emb))

    con = duckdb.connect(db_path)

    current_id = index.ntotal
    for page_num, img in enumerate(tqdm(pages, desc="Processing pages")):
        text = extract_text_from_image(img)
        te = get_text_embedding(text)
        ie = get_image_embedding(img)
        emb = combine_embeddings(te, ie, method)

        index.add(np.array([emb], dtype="float32"))

        con.execute("""
        INSERT INTO metadata (id, document_class, filename, page_number)
        VALUES (?, ?, ?, ?)
        """, (current_id, document_class, pdf_path, page_num + 1))

        current_id += 1

    faiss.write_index(index, faiss_path)
    con.close()
    print(f"✅ Stored embeddings for {pdf_path} as class '{document_class}'")


# ---------- Classification ----------
def classify_document(pdf_path, k=3, method="concat"):
    index = faiss.read_index(faiss_path)
    con = duckdb.connect(db_path, read_only=True)

    class_votes = {}
    pages = convert_from_path(pdf_path, dpi=150)

    for img in tqdm(pages, desc="Classifying pages"):
        text = extract_text_from_image(img)
        te = get_text_embedding(text)
        ie = get_image_embedding(img)
        emb = combine_embeddings(te, ie, method).reshape(1, -1)

        distances, indices = index.search(emb, k)

        for idx in indices[0]:
            doc_class = con.execute("SELECT document_class FROM metadata WHERE id = ?", (int(idx),)).fetchone()[0]
            class_votes[doc_class] = class_votes.get(doc_class, 0) + 1

    con.close()
    predicted_class = max(class_votes, key=class_votes.get)
    return predicted_class, class_votes


# ---------- Example ----------
if __name__ == "__main__":
    # Ingest a document
    store_pdf("document.pdf", document_class="recipe")

    # Classify a new document
    pred_class, votes = classify_document("unknown.pdf")
    print("\n📄 Predicted document class:", pred_class)
    print("Votes by class:", votes)
