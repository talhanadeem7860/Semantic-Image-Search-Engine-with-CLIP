import streamlit as st
import os
import numpy as np
import faiss
import glob
from PIL import Image
from sentence_transformers import SentenceTransformer
import torch

# --- Configuration ---
IMAGE_DIR = "image_collection"
INDEX_PATH = "faiss_image_index.idx"
IMAGE_PATHS_FILE = "image_paths.txt"


@st.cache_resource
def load_model():
    """Load the CLIP model only once."""
    print("[INFO] Loading CLIP model...")
    model = SentenceTransformer('clip-ViT-B-32')
    print("[INFO] CLIP model loaded.")
    return model

@st.cache_data
def create_or_load_index(_model, image_dir):
    if os.path.exists(INDEX_PATH) and os.path.exists(IMAGE_PATHS_FILE):
        print(f"[INFO] Loading existing FAISS index from {INDEX_PATH}...")
        index = faiss.read_index(INDEX_PATH)
        with open(IMAGE_PATHS_FILE, 'r') as f:
            image_paths = [line.strip() for line in f.readlines()]
        print(f"[INFO] Loaded {len(image_paths)} image paths.")
        return index, image_paths

    print(f"[INFO] Creating new FAISS index for images in {image_dir}...")
    image_paths = glob.glob(os.path.join(image_dir, "*.jpg")) + \
                  glob.glob(os.path.join(image_dir, "*.png"))
    
    if not image_paths:
        st.error(f"No images found in the '{image_dir}' directory. Please add some images.")
        return None, None

    # Create embeddings in batches
    batch_size = 32
    all_embeddings = []
    
    progress_bar = st.progress(0, text="Creating image embeddings...")
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = [Image.open(path) for path in batch_paths]
        embeddings = _model.encode(batch_images, convert_to_tensor=True, show_progress_bar=False)
        all_embeddings.append(embeddings.cpu().numpy())
        progress_bar.progress((i + len(batch_paths)) / len(image_paths), text=f"Processed {i + len(batch_paths)} / {len(image_paths)} images")

    # Concatenate all embeddings and create the FAISS index
    image_embeddings = np.vstack(all_embeddings)
    dimension = image_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(image_embeddings)

    # Save the index and the image paths for future use
    faiss.write_index(index, INDEX_PATH)
    with open(IMAGE_PATHS_FILE, 'w') as f:
        for path in image_paths:
            f.write(f"{path}\n")
    
    progress_bar.empty()
    print(f"[INFO] Index created and saved with {len(image_paths)} images.")
    return index, image_paths


def search_images(text_query, _model, _index, _image_paths, k=5):
    """
    Search for images using a text query.
    """
    # Create the text embedding
    text_embedding = _model.encode([text_query], convert_to_tensor=True).cpu().numpy()
    
    # Perform the search
    distances, indices = _index.search(text_embedding, k)
    
    # Get the paths of the matched images
    matched_paths = [_image_paths[i] for i in indices[0]]
    return matched_paths


st.set_page_config(layout="wide")
st.title("🖼️ Semantic Image Search Engine")
st.write("Search for images using natural language descriptions. This app uses OpenAI's CLIP model to understand the content of both text and images.")

# Load the model and create/load the index
model = load_model()
index, image_paths = create_or_load_index(model, IMAGE_DIR)

if index is not None:
    # --- Search Interface ---
    st.header("Search for an Image")
    query = st.text_input("Describe the image you want to find:", "a happy dog playing in a field")
    num_results = st.slider("Number of results to show:", 1, 20, 5)

    if st.button("Search"):
        if query:
            with st.spinner("Searching for matching images..."):
                results = search_images(query, model, index, image_paths, k=num_results)

            st.header("Search Results")
            if results:
                # Display results in columns
                cols = st.columns(num_results)
                for i, path in enumerate(results):
                    with cols[i]:
                        st.image(path, use_column_width=True, caption=os.path.basename(path))
            else:
                st.warning("No matching images found.")
        else:
            st.warning("Please enter a search query.")
else:
    st.info("Please create an 'image_collection' directory and add images to it, then refresh the page.")