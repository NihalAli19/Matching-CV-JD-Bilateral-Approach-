import streamlit as st
import os
import re
import math
import nltk
import numpy as np
import pandas as pd
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
import pickle

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Initialize session state
if 'user_type' not in st.session_state:
    st.session_state.user_type = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'default_cvs' not in st.session_state:
    st.session_state.default_cvs = {}
if 'default_jds' not in st.session_state:
    st.session_state.default_jds = {}

# Load model once
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Text cleaning function
def clean_text(text):
    text = re.sub(r'-{5,}.*?-{5,}', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Bigram model functions
def build_bigram_model(text):
    tokenize = nltk.word_tokenize
    tokens = ['<s>'] + tokenize(text.lower()) + ['</s>']
    bigrams = list(nltk.bigrams(tokens))
    
    unigram_counts = Counter(tokens)
    bigram_counts = Counter(bigrams)
    
    vocab = set(tokens)
    V = len(vocab)
    
    def prob(w1, w2):
        return (bigram_counts[(w1, w2)] + 1) / (unigram_counts[w1] + V)
    
    return prob

def score_text_likelihood(text, model):
    tokenize = nltk.word_tokenize
    tokens = ['<s>'] + tokenize(text.lower()) + ['</s>']
    bigrams = list(nltk.bigrams(tokens))
    log_prob = 0
    for w1, w2 in bigrams:
        log_prob += math.log(model(w1, w2))
    return log_prob

# Combined scoring function
def get_combined_scores(target_text, documents, doc_names, alpha=0.7):
    model = load_model()
    
    # Generate embeddings
    all_texts = [target_text] + documents
    embeddings = model.encode(all_texts)
    
    target_embedding = embeddings[0].reshape(1, -1)
    doc_embeddings = embeddings[1:]
    
    # Calculate VSM similarities
    vsm_scores = []
    for doc_embedding in doc_embeddings:
        sim = cosine_similarity(target_embedding, doc_embedding.reshape(1, -1))[0][0]
        vsm_scores.append(sim)
    
    # Build bigram model for target text
    target_model = build_bigram_model(target_text)
    
    # Calculate bigram scores
    bigram_scores = []
    for doc in documents:
        score = score_text_likelihood(doc, target_model)
        bigram_scores.append(score)
    
    # Normalize bigram scores
    min_ll = min(bigram_scores)
    max_ll = max(bigram_scores)
    
    normalized_bigram_scores = []
    for score in bigram_scores:
        if max_ll == min_ll:
            normalized_bigram_scores.append(0.5)
        else:
            normalized_bigram_scores.append((score - min_ll) / (max_ll - min_ll))
    
    # Combine scores
    combined_scores = []
    for i in range(len(documents)):
        combined_score = alpha * vsm_scores[i] + (1 - alpha) * normalized_bigram_scores[i]
        combined_scores.append({
            'name': doc_names[i],
            'vsm_score': vsm_scores[i],
            'bigram_score': bigram_scores[i],
            'normalized_bigram': normalized_bigram_scores[i],
            'combined_score': combined_score
        })
    
    # Sort by combined score
    combined_scores.sort(key=lambda x: x['combined_score'], reverse=True)
    return combined_scores




def load_documents_from_folder(folder_path):
    documents = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
            except UnicodeDecodeError:
                # Fallback to latin-1 if UTF-8 fails
                with open(file_path, "r", encoding="latin-1") as f:
                    content = f.read().strip()
            key_name = os.path.splitext(filename)[0].replace('_', ' ').title()
            documents[key_name] = content
    return documents
# Define the folders
CV_FOLDER = "./cvs"
JD_FOLDER = "./jds"


# Load CVs and Job Descriptions
DEFAULT_CVS = load_documents_from_folder(CV_FOLDER)
DEFAULT_JDS = load_documents_from_folder(JD_FOLDER)


# Main app
def main():
    st.set_page_config(
        page_title="Job-CV Matching System",
        page_icon="🤝",
        layout="wide"
    )
    
    st.title("🤝 Job-CV Matching System")
    st.markdown("---")
    
    # User type selection
    if st.session_state.user_type is None:
        st.markdown("### Welcome! Please select your role:")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("👔 I'm a Recruiter", use_container_width=True, type="primary"):
                st.session_state.user_type = "recruiter"
                st.rerun()
        
        with col3:
            if st.button("👤 I'm a Candidate", use_container_width=True, type="secondary"):
                st.session_state.user_type = "candidate"
                st.rerun()
        
        st.markdown("---")
        st.info("**Recruiters**: Get recommended CVs based on your job descriptions\n\n**Candidates**: Get recommended jobs based on your CV")
        
    else:
        # Show current role and option to switch
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.session_state.user_type == "recruiter":
                st.markdown("### 👔 Recruiter Dashboard")
            else:
                st.markdown("### 👤 Candidate Dashboard")
        
        with col2:
            if st.button("🔄 Switch Role", type="secondary"):
                st.session_state.user_type = None
                st.rerun()
        
        st.markdown("---")
        
        # Show appropriate interface
        if st.session_state.user_type == "recruiter":
            show_recruiter_interface()
        else:
            show_candidate_interface()

def show_recruiter_interface():
    st.markdown("## Find the Best CVs for Your Job Description")
    
    tab1, tab2 = st.tabs(["📝 Custom Job Description", "📋 Default Job Descriptions"])
    
    with tab1:
        st.markdown("### Enter Your Job Description")
        
        job_title = st.text_input("Job Title", placeholder="e.g., Senior Software Engineer")
        job_description = st.text_area(
            "Job Description",
            height=300,
            placeholder="Paste your complete job description here..."
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            alpha = st.slider("VSM Weight", 0.0, 1.0, 0.7, 0.1, help="Higher values prioritize semantic similarity")
        
        if st.button("🔍 Find Matching CVs", type="primary", disabled=not job_description.strip()):
            if job_description.strip():
                with st.spinner("Analyzing CVs and finding matches..."):
                    # Use default CVs for demo
                    cv_texts = list(DEFAULT_CVS.values())
                    cv_names = list(DEFAULT_CVS.keys())
                    
                    cleaned_jd = clean_text(job_description)
                    cleaned_cvs = [clean_text(cv) for cv in cv_texts]
                    
                    results = get_combined_scores(cleaned_jd, cleaned_cvs, cv_names, alpha)
                    
                    st.success(f"Found {len(results)} matching CVs!")
                    display_cv_results(results, job_title if job_title else "Your Job Description")
    
    with tab2:
        st.markdown("### Default Job Descriptions Analysis")
        st.info("These results show CV matches for our default job descriptions.")
        
        # Select job description
        selected_jd = st.selectbox("Select a Job Description to Analyze:", list(DEFAULT_JDS.keys()))
        
        if selected_jd:
            st.markdown(f"#### 📄 {selected_jd}")
            st.text_area(f"{selected_jd} Description", value=DEFAULT_JDS[selected_jd], height=200, disabled=True)
            
            if st.button(f"Analyze {selected_jd}", key=f"analyze_{selected_jd}"):
                with st.spinner(f"Analyzing CVs for {selected_jd}..."):
                    cv_texts = list(DEFAULT_CVS.values())
                    cv_names = list(DEFAULT_CVS.keys())
                    
                    cleaned_jd = clean_text(DEFAULT_JDS[selected_jd])
                    cleaned_cvs = [clean_text(cv) for cv in cv_texts]
                    
                    results = get_combined_scores(cleaned_jd, cleaned_cvs, cv_names, 0.7)
                    display_cv_results(results, selected_jd)

def show_candidate_interface():
    st.markdown("## Find the Best Jobs for Your CV")
    
    tab1, tab2 = st.tabs(["📄 Upload Your CV", "👥 Default CV Examples"])
    
    with tab1:
        st.markdown("### Upload Your CV")
        
        # File upload option
        uploaded_file = st.file_uploader(
            "Upload CV (PDF/TXT)", 
            type=['pdf', 'txt'],
            help="Upload your CV in PDF or text format"
        )
        
        # Text input option
        st.markdown("**Or paste your CV text below:**")
        cv_text = st.text_area(
            "CV Content",
            height=300,
            placeholder="Paste your complete CV content here..."
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            alpha = st.slider("VSM Weight", 0.0, 1.0, 0.7, 0.1, help="Higher values prioritize semantic similarity", key="candidate_alpha")
        
        # Process uploaded file
        final_cv_text = ""
        if uploaded_file is not None:
            if uploaded_file.type == "text/plain":
                final_cv_text = str(uploaded_file.read(), "utf-8")
            elif uploaded_file.type == "application/pdf":
                st.warning("PDF processing requires additional libraries. Please copy-paste your CV text instead.")
        elif cv_text.strip():
            final_cv_text = cv_text
        
        if st.button("🔍 Find Matching Jobs", type="primary", disabled=not final_cv_text.strip()):
            if final_cv_text.strip():
                with st.spinner("Analyzing your CV and finding job matches..."):
                    jd_texts = list(DEFAULT_JDS.values())
                    jd_names = list(DEFAULT_JDS.keys())
                    
                    cleaned_cv = clean_text(final_cv_text)
                    cleaned_jds = [clean_text(jd) for jd in jd_texts]
                    
                    results = get_combined_scores(cleaned_cv, cleaned_jds, jd_names, alpha)
                    
                    st.success(f"Found {len(results)} matching job opportunities!")
                    display_job_results(results)
    
    with tab2:
        st.markdown("### Default CV Examples Analysis")
        st.info("These results show job matches for our example CVs.")
        
        # Select CV
        selected_cv = st.selectbox("Select a CV to Analyze:", list(DEFAULT_CVS.keys()))
        
        if selected_cv:
            st.markdown(f"#### 👤 {selected_cv}")
            st.text_area(f"{selected_cv} Content", value=DEFAULT_CVS[selected_cv], height=200, disabled=True)
            
            if st.button(f"Find Jobs for {selected_cv}", key=f"jobs_{selected_cv}"):
                with st.spinner(f"Finding jobs for {selected_cv}..."):
                    jd_texts = list(DEFAULT_JDS.values())
                    jd_names = list(DEFAULT_JDS.keys())
                    
                    cleaned_cv = clean_text(DEFAULT_CVS[selected_cv])
                    cleaned_jds = [clean_text(jd) for jd in jd_texts]
                    
                    results = get_combined_scores(cleaned_cv, cleaned_jds, jd_names, 0.7)
                    display_job_results(results, selected_cv)

def display_cv_results(results, job_title):
    st.markdown(f"### 📊 CV Recommendations for: {job_title}")
    
    # Create results DataFrame
    df_data = []
    for i, result in enumerate(results, 1):
        df_data.append({
            'Rank': i,
            'CV Name': result['name'],
            'Combined Score': f"{result['combined_score']:.4f}",
            'VSM Score': f"{result['vsm_score']:.4f}",
            'Bigram Score': f"{result['bigram_score']:.2f}",
            'Match %': f"{result['combined_score']*100:.1f}%"
        })
    
    df = pd.DataFrame(df_data)
    
    # Display summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Best Match", f"{results[0]['combined_score']*100:.1f}%")
    with col2:
        st.metric("Top CV", results[0]['name'].split(' - ')[0])
    with col3:
        st.metric("Total CVs", len(results))
    
    # Display results table
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Detailed breakdown with selectbox instead of nested expanders
    st.markdown("### 📈 Detailed Analysis")
    selected_cv_for_details = st.selectbox(
        "Select CV for detailed analysis:",
        options=[f"#{i+1} - {result['name']}" for i, result in enumerate(results)],
        key="cv_details_select"
    )
    
    if selected_cv_for_details:
        # Extract the index from the selection
        selected_index = int(selected_cv_for_details.split('#')[1].split(' - ')[0]) - 1
        result = results[selected_index]
        
        st.markdown(f"#### Analysis for {result['name']}")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Combined Score", f"{result['combined_score']:.4f}")
        with col2:
            st.metric("Semantic Similarity", f"{result['vsm_score']:.4f}")
        with col3:
            st.metric("Language Model Score", f"{result['bigram_score']:.2f}")
        with col4:
            st.metric("Match Percentage", f"{result['combined_score']*100:.1f}%")
        
        # Show CV preview
        if result['name'] in DEFAULT_CVS:
            st.text_area("CV Preview", DEFAULT_CVS[result['name']][:500] + "...", height=150, disabled=True)

def display_job_results(results, cv_name=None):
    title = f"📊 Job Recommendations for: {cv_name}" if cv_name else "📊 Job Recommendations for Your CV"
    st.markdown(f"### {title}")
    
    # Create results DataFrame
    df_data = []
    for i, result in enumerate(results, 1):
        df_data.append({
            'Rank': i,
            'Job Title': result['name'],
            'Combined Score': f"{result['combined_score']:.4f}",
            'VSM Score': f"{result['vsm_score']:.4f}",
            'Bigram Score': f"{result['bigram_score']:.2f}",
            'Match %': f"{result['combined_score']*100:.1f}%"
        })
    
    df = pd.DataFrame(df_data)
    
    # Display summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Best Match", f"{results[0]['combined_score']*100:.1f}%")
    with col2:
        st.metric("Top Job", results[0]['name'])
    with col3:
        st.metric("Total Jobs", len(results))
    
    # Display results table
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Detailed breakdown with selectbox instead of nested expanders
    st.markdown("### 📈 Detailed Analysis")
    selected_job_for_details = st.selectbox(
        "Select job for detailed analysis:",
        options=[f"#{i+1} - {result['name']}" for i, result in enumerate(results)],
        key="job_details_select"
    )
    
    if selected_job_for_details:
        # Extract the index from the selection
        selected_index = int(selected_job_for_details.split('#')[1].split(' - ')[0]) - 1
        result = results[selected_index]
        
        st.markdown(f"#### Analysis for {result['name']}")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Combined Score", f"{result['combined_score']:.4f}")
        with col2:
            st.metric("Semantic Similarity", f"{result['vsm_score']:.4f}")
        with col3:
            st.metric("Language Model Score", f"{result['bigram_score']:.2f}")
        with col4:
            st.metric("Match Percentage", f"{result['combined_score']*100:.1f}%")
        
        # Show job description preview
        if result['name'] in DEFAULT_JDS:
            st.text_area("Job Description Preview", DEFAULT_JDS[result['name']][:500] + "...", height=150, disabled=True)

if __name__ == "__main__":
    main()