import os
import sys
import numpy as np
import pandas as pd
import streamlit as st

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai


# =========================
# CONFIG
# =========================

st.set_page_config(
    page_title="AISCA - Recommandation Métier IA",
    page_icon="🤖",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COMPETENCIES_PATH = os.path.join(BASE_DIR, "data", "gold", "competencies.csv")
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "data", "gold", "competency_embeddings.npy")
JOBS_PATH = os.path.join(BASE_DIR, "data", "gold", "jobs.csv")
MAPPING_PATH = os.path.join(BASE_DIR, "data", "gold", "mapping.csv")
ENV_PATH = os.path.join(BASE_DIR, ".env")


# =========================
# LOAD DATA
# =========================

@st.cache_data
def load_data():
    competencies_df = pd.read_csv(COMPETENCIES_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)
    jobs_df = pd.read_csv(JOBS_PATH)
    mapping_df = pd.read_csv(MAPPING_PATH)
    return competencies_df, embeddings, jobs_df, mapping_df


@st.cache_resource
def load_model():
    return SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2",
        device="cpu"
    )


competencies_df, competency_embeddings, jobs_df, mapping_df = load_data()
model = load_model()


# =========================
# FUNCTIONS
# =========================

def compute_similarity(user_text, threshold=0.45):
    user_embedding = model.encode([user_text])

    similarities = cosine_similarity(
        user_embedding,
        competency_embeddings
    )[0]

    results_df = competencies_df.copy()
    results_df["similarity_score"] = similarities

    results_df = results_df.sort_values(
        by="similarity_score",
        ascending=False
    )

    relevant_skills = results_df[
        results_df["similarity_score"] >= threshold
    ]

    return results_df, relevant_skills


def compute_block_scores(relevant_skills):
    if relevant_skills.empty:
        return pd.DataFrame(columns=["block", "similarity_score"])

    block_scores = relevant_skills.groupby("block")["similarity_score"].mean().reset_index()

    block_scores = block_scores.sort_values(
        by="similarity_score",
        ascending=False
    )

    return block_scores


def recommend_jobs(relevant_skills):
    if relevant_skills.empty:
        return pd.DataFrame()

    detected_skills = relevant_skills[
        ["competency_id", "competency", "similarity_score"]
    ]

    job_skill_scores = mapping_df.merge(
        detected_skills,
        on="competency_id",
        how="inner"
    )

    if job_skill_scores.empty:
        return pd.DataFrame()

    job_scores = job_skill_scores.groupby("job_id").agg(
        matched_skills=("competency_id", "count"),
        recommendation_score=("similarity_score", "mean")
    ).reset_index()

    job_scores = job_scores.merge(
        jobs_df,
        on="job_id",
        how="left"
    )

    target_jobs = [
        "Senior Data Scientist",
        "Data Scientist",
        "Machine Learning Engineer",
        "DevOps Engineer",
        "Software Engineer",
        "Backend Developer",
        "Full Stack Developer",
        "Data Analyst",
        "Solutions Architect"
    ]

    job_scores = job_scores[
        job_scores["job_title"].isin(target_jobs)
    ]

    top_jobs = job_scores.sort_values(
        by=["matched_skills", "recommendation_score"],
        ascending=False
    ).head(3)

    def get_matched_skills(job_id):
        skills = job_skill_scores[
            job_skill_scores["job_id"] == job_id
        ]["competency"].unique()

        return ", ".join(skills)

    top_jobs["matched_skill_names"] = top_jobs["job_id"].apply(get_matched_skills)

    return top_jobs


def generate_genai_summary(skills, jobs):
    load_dotenv(ENV_PATH)
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return "Clé API Gemini introuvable. Ajoutez GEMINI_API_KEY dans le fichier .env."

    genai.configure(api_key=api_key)

    model_genai = genai.GenerativeModel("gemini-3-flash-preview")

    prompt = f"""
Tu es un conseiller carrière spécialisé en intelligence artificielle.

Voici les compétences détectées chez l'utilisateur :
{skills}

Voici les métiers recommandés par le moteur de scoring :
{jobs}

Ta mission :
1. Résumer le profil professionnel de l'utilisateur.
2. Expliquer pourquoi les métiers recommandés sont cohérents.
3. Identifier les compétences à renforcer.
4. Proposer une roadmap d'apprentissage sur 6 mois.

Réponds en français avec un style professionnel, clair et synthétique.
"""

    response = model_genai.generate_content(prompt)
    return response.text


# =========================
# UI
# =========================

st.title("🤖 AISCA — Agent Intelligent Sémantique pour la Cartographie des Compétences")

st.markdown(
    """
Cette application analyse un profil utilisateur en langage naturel, détecte les compétences associées,
calcule une similarité sémantique avec un référentiel de compétences, puis recommande des métiers adaptés.
"""
)

st.sidebar.header("Paramètres")

threshold = st.sidebar.slider(
    "Seuil de similarité",
    min_value=0.30,
    max_value=0.80,
    value=0.45,
    step=0.05
)

st.sidebar.markdown("---")
st.sidebar.markdown("Modèle NLP : SBERT multilingue")
st.sidebar.markdown("GenAI : Gemini")

user_text = st.text_area(
    "Décrivez votre expérience, vos projets ou vos compétences :",
    height=180,
    placeholder="Exemple : J’ai développé des applications Python, travaillé avec Docker, déployé des services sur AWS et entraîné des modèles de machine learning."
)

analyze_button = st.button("Analyser mon profil")

if analyze_button:

    if not user_text.strip():
        st.warning("Veuillez saisir une description de profil.")
    else:
        with st.spinner("Analyse sémantique en cours..."):
            results_df, relevant_skills = compute_similarity(
                user_text,
                threshold=threshold
            )

            block_scores = compute_block_scores(relevant_skills)
            top_jobs = recommend_jobs(relevant_skills)

        st.subheader("Compétences détectées")

        st.dataframe(
            relevant_skills[
                ["competency", "block", "similarity_score"]
            ],
            use_container_width=True
        )

        st.subheader("Top 10 similarités")

        st.dataframe(
            results_df[
                ["competency", "block", "similarity_score"]
            ].head(10),
            use_container_width=True
        )

        st.subheader("Score par bloc de compétences")

        if not block_scores.empty:
            st.bar_chart(
                block_scores.set_index("block")["similarity_score"]
            )
        else:
            st.info("Aucun bloc de compétences détecté avec ce seuil.")

        st.subheader("Métiers recommandés")

        if not top_jobs.empty:
            st.dataframe(
                top_jobs[
                    [
                        "job_title",
                        "category",
                        "matched_skills",
                        "recommendation_score",
                        "matched_skill_names"
                    ]
                ],
                use_container_width=True
            )

            st.subheader("Synthèse générée par IA")

            with st.spinner("Génération de la synthèse Gemini..."):
                skills = relevant_skills["competency"].tolist()
                jobs = top_jobs["job_title"].tolist()

                genai_text = generate_genai_summary(
                    skills,
                    jobs
                )

            st.markdown(genai_text)

        else:
            st.warning("Aucun métier correspondant n’a été trouvé.")