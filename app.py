# from fastapi import FastAPI, Request
# from fastapi.responses import HTMLResponse, JSONResponse
# from fastapi.templating import Jinja2Templates
# import pandas as pd
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from fastapi.staticfiles import StaticFiles
# # Ensure the correct absolute path to the directory containing your static files
# import os


# # Initialize FastAPI app
# app = FastAPI()

# # Initialize Jinja2 templates
# templates = Jinja2Templates(directory="templates")

# # Mount the static files directory
# current_dir = os.path.dirname(os.path.abspath(__file__))
# app.mount("/static", StaticFiles(directory=current_dir), name="static")
# # Download required NLTK data
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('stopwords')

# # Load the dataset
# df = pd.read_csv('cleaned_disease_dataset.csv')

# def extract_keywords(description):
#     """Extract and clean keywords using NLTK."""
#     stop_words = set(stopwords.words('english'))
#     tokens = word_tokenize(description.lower())
#     keywords = {word for word in tokens if word.isalnum() and word not in stop_words}
#     return keywords

# # Store disease keywords in a dictionary
# disease_keywords = {}
# for _, row in df.iterrows():
#     disease = row['Disease']
#     keywords = extract_keywords(row['Description'])
#     disease_keywords[disease] = {
#         "keywords": keywords,
#         "precautions": row['Precautions']
#     }

# def jaccard_similarity(set1, set2):
#     """Calculate Jaccard similarity between two sets."""
#     intersection = len(set1.intersection(set2))
#     union = len(set1.union(set2))
#     return intersection / union if union != 0 else 0

# def predict_disease(user_input):
#     """Predict disease by matching user symptoms with stored keywords."""
#     user_tokens = extract_keywords(user_input)
    
#     best_match = None
#     best_score = 0  # Higher Jaccard similarity = better match

#     for disease, data in disease_keywords.items():
#         disease_tokens = data['keywords']
        
#         if disease_tokens:
#             similarity_score = jaccard_similarity(user_tokens, disease_tokens)
            
#             if similarity_score > best_score:
#                 best_score = similarity_score
#                 best_match = {
#                     "disease": disease,
#                     "precautions": data['precautions']
#                 }

#     return best_match

# @app.get("/", response_class=HTMLResponse)
# async def read_root(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})

# @app.post("/predict")
# async def predict(request: Request):
#     data = await request.json()
#     user_input = data['symptoms']
#     result = predict_disease(user_input)
    
#     if result:
#         return JSONResponse({
#             'success': True,
#             'disease': result['disease'],
#             'precautions': result['precautions']
#         })
#     else:
#         return JSONResponse({
#             'success': False,
#             'message': 'No matching disease found.'
#         }) 





from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os

# Download NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Initialize FastAPI app
app = FastAPI()

# Setup static and template directories
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load your dataset
df = pd.read_csv('cleaned_disease_dataset.csv')

def extract_keywords(description):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(description.lower())
    return {word for word in tokens if word.isalnum() and word not in stop_words}

# Preprocess and store keyword data
disease_keywords = {
    row['Disease']: {
        "keywords": extract_keywords(row['Description']),
        "precautions": row['Precautions']
    }
    for _, row in df.iterrows()
}

def jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union else 0

def predict_disease(user_input):
    user_tokens = extract_keywords(user_input)
    best_match = None
    best_score = 0

    for disease, data in disease_keywords.items():
        score = jaccard_similarity(user_tokens, data['keywords'])
        if score > best_score:
            best_score = score
            best_match = {"disease": disease, "precautions": data['precautions']}

    return best_match

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    user_input = data.get('symptoms', '')
    result = predict_disease(user_input)

    if result:
        return JSONResponse({'success': True, **result})
    else:
        return JSONResponse({'success': False, 'message': 'No matching disease found.'})
