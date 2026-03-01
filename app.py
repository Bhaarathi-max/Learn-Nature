import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import os
import cv2
class_names = [
    'INDIAN BEAN BUG',
    'COMMON CROW BUTTERFLY',
    'INDIAN RED BUG',
    'ORIENTAL BEETLE',
    'PLAIN TIGER BUTTERFLY',
    'INDIAN POTTER WASP',
    'SLENDER MEADOW KATYDID',
    'SUNDOWNER MOTH',
    'TROPICAL TIGER MOTH',
    'WANDERING GLIDER']
# --- SESSION STATE INITIALIZATION ---

if "qa_answers" not in st.session_state:
    st.session_state.qa_answers = {}

if "show_questions" not in st.session_state:
    st.session_state.show_questions = False

if "initial_confidence" not in st.session_state:
    st.session_state.initial_confidence = 0

if "initial_pred_class" not in st.session_state:
    st.session_state.initial_pred_class = ""
    
# --- Global Variables ---
CONFIDENCE_THRESHOLD = 0.95
IMG_HEIGHT = 224
IMG_WIDTH = 224

# --- Paths ---
model_path = "INSECT_CNN_FINAL.keras"

# --- Load Model ---
@st.cache_resource
def load_keras_model():
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_keras_model()

# --- Taxonomy Dictionary ---
taxonomy = {
    'BEAN BUG': {
        'common_name': 'Bean Bug',
        'species': 'Riptortus pedestris',
        'genus': 'Riptortus',
        'family': 'Alydidae',
        'order': 'Hemiptera',
        'class': 'Insecta',
        'phylum': 'Arthropoda',
        'kingdom': 'Animalia'
    },
    'COMMON CROW BUTTERFLY': {
        'common_name': 'Common Crow Butterfly',
        'species': 'Euploea core',
        'genus': 'Euploea',
        'family': 'Nymphalidae',
        'order': 'Lepidoptera',
        'class': 'Insecta',
        'phylum': 'Arthropoda',
        'kingdom': 'Animalia'
    },
    'INDIAN RED BUG': {
        'common_name': 'Indian Red Bug',
        'species': 'Dysdercus cingulatus',
        'genus': 'Dysdercus',
        'family': 'Pyrrhocoridae',
        'order': 'Hemiptera',
        'class': 'Insecta',
        'phylum': 'Arthropoda',
        'kingdom': 'Animalia'
    },
    'ORIENTAL BEETLE': {
        'common_name': 'Oriental Beetle',
        'species': 'Anomala orientalis',
        'genus': 'Anomala',
        'family': 'Scarabaeidae',
        'order': 'Coleoptera',
        'class': 'Insecta',
        'phylum': 'Arthropoda',
        'kingdom': 'Animalia'
    },
    'PLAIN TIGER BUTTERFLY': {
        'common_name': 'Plain Tiger Butterfly',
        'species': 'Danaus chrysippus',
        'genus': 'Danaus',
        'family': 'Nymphalidae',
        'order': 'Lepidoptera',
        'class': 'Insecta',
        'phylum': 'Arthropoda',
        'kingdom': 'Animalia'
    },
    'INDIAN POTTER WASP': {
        'common_name': 'Potter Wasp',
        'species': 'Delta pyriforme',
        'genus': 'Delta',
        'family': 'Vespidae',
        'order': 'Hymenoptera',
        'class': 'Insecta',
        'phylum': 'Arthropoda',
        'kingdom': 'Animalia'
    },
    'SLENDER MEADOW KATYDID': {
        'common_name': 'Slender Meadow Katydid',
        'species': 'Conocephalus fasciatus',
        'genus': 'Conocephalus',
        'family': 'Tettigoniidae',
        'order': 'Orthoptera',
        'class': 'Insecta',
        'phylum': 'Arthropoda',
        'kingdom': 'Animalia'
    },
    'SUNDOWNER MOTH': {
        'common_name': 'Sundowner Moth',
        'species': 'Spingomorpha chlorea',
        'genus': 'Spingomorpha',
        'family': 'Erebidae',
        'order': 'Lepidoptera',
        'class': 'Insecta',
        'phylum': 'Arthropoda',
        'kingdom': 'Animalia'
    },
    'TROPICAL TIGER MOTH': {
        'common_name': 'Tropical Tiger Moth',
        'species': 'Asota caricae',
        'genus': 'Asota',
        'family': 'Erebidae',
        'order': 'Lepidoptera',
        'class': 'Insecta',
        'phylum': 'Arthropoda',
        'kingdom': 'Animalia'
    },
    'WANDERING GLIDER': {
        'common_name': 'Wandering Glider',
        'species': 'Pantala flavescens',
        'genus': 'Pantala',
        'family': 'Libellulidae',
        'order': 'Odonata',
        'class': 'Insecta',
        'phylum': 'Arthropoda',
        'kingdom': 'Animalia'
    }
}

# --- Streamlit-compatible ask_questions function ---
def ask_questions_streamlit():
    if st.session_state.show_questions:
       st.subheader("Please answer the following questions to help identify the insect:")

    # Initialize answers in session state if not present
    if 'qa_answers' not in st.session_state:
        st.session_state.qa_answers = {}

    # Use a form to group questions and submit at once

        # Q1
        st.session_state.qa_answers["wings_visible"] = st.radio(
            "Q1: Does the insect have *visible* wings or wing structures?",
            options=["Select...", "yes", "no", "unknown"],
            key="wings_visible_q"
        )

        if st.session_state.qa_answers["wings_visible"] == "yes":

            # Q2
            st.session_state.qa_answers["num_wings"] = st.radio(
                "Q2: How many distinct wings are clearly visible? (Consider if forewings cover hindwings)",
                options=["Select...", "2", "4", "more", "unknown"],
                key="num_wings_q"
            )

            # Q3
            st.session_state.qa_answers["transparent_wings"] = st.radio(
                "Q3: Are the wings largely transparent/clear or mostly opaque/colored?",
                options=["Select...", "transparent", "opaque", "unknown"],
                key="transparent_wings_q"
            )

            # Q4 (was text input → now radio with same options)
            st.session_state.qa_answers["wing_color_pattern"] = st.radio(
                "Q4: Describe the dominant wing color or pattern:",
                options=[
                    "black with white spots",
                    "orange with black border and white spots",
                    "clear",
                    "golden tint",
                    "other/unknown"
                ],
                key="wing_color_pattern_q"
            )

            # Q5
            st.session_state.qa_answers["resting_position"] = st.radio(
                "Q5: How does it typically hold its wings at rest?",
                options=["Select...", 'flat over body', 'tent-like', 'vertically upright', 'outstretched', 'other/unknown'],
                key="resting_position_q"
            )

        else:
            st.session_state.qa_answers["num_wings"] = "n/a"
            st.session_state.qa_answers["transparent_wings"] = "n/a"
            st.session_state.qa_answers["wing_color_pattern"] = "n/a"
            st.session_state.qa_answers["resting_position"] = "n/a"

        # Q6 (body color) → replaced text input with radio for common colors
        st.session_state.qa_answers["body_color"] = st.radio(
            "Q6: What is the insect's dominant body color?",
            options=["Select...", "red", "brown", "black", "green", "yellow", "orange", "other/unknown"],
            key="body_color_q"
        )

        # Q7
        st.session_state.qa_answers["body_texture_appearance"] = st.radio(
            "Q7: Is the body predominantly...",
            options=["Select...", 'hard and shiny', 'soft', 'hairy/furry', 'elongated with narrow middle part', 'elongated and slender', 'other/unknown'],
            key="body_texture_appearance_q"
        )

        # Q8
        st.session_state.qa_answers["num_legs"] = st.radio(
            "Q8: Number of visible legs",
            options=["Select...", "6", "8", "more", "n/a", "unknown"],
            key="num_legs_q"
        )

        # Q9
        st.session_state.qa_answers["antennae_present"] = st.radio(
            "Q9: Are antennae clearly visible?",
            options=["Select...", "yes", "no", "unknown"],
            key="antennae_present_q"
        )

        if st.session_state.qa_answers["antennae_present"] == "yes":
            # Q10
            st.session_state.qa_answers["antennae_shape"] = st.radio(
                "Q10: What is the shape of the antennae?",
                options=["Select...", "clubbed", "thread-like", "bent", "very long", "small", "3 spikes", "other", "unknown"],
                key="antennae_shape_q"
            )
            # Q11 (antennae color)
            st.session_state.qa_answers["antennae_color"] = st.radio(
                "Q11: What is the main color of the antennae?",
                options=["Select...", "black", "brown", "orange", "other/unknown"],
                key="antennae_color_q"
            )
        else:
            st.session_state.qa_answers["antennae_shape"] = "n/a"
            st.session_state.qa_answers["antennae_color"] = "n/a"

        # Q12 (eye color)
        st.session_state.qa_answers["eye_color"] = st.radio(
            "Q12: What is the predominant eye color?",
            options=["Select...", "dark", "red", "yellow", "brown", "green", "other", "n/a", "unknown"],
            key="eye_color_q"
        )

        if st.button("Submit Clarification"):
           return st.session_state.qa_answers
        else:
           return None
    
# --- rule_based_identification function (adapted to new question structure) ---
def rule_based_identification(ans):
    def contains_any(user_answer, keywords):
        if not isinstance(user_answer, str):
            user_answer = str(user_answer)
        return any(keyword.lower() in user_answer.lower() for keyword in keywords)

    # WANDERING GLIDER
    if (
        ans["wings_visible"] == "yes" and
        (ans["num_wings"] in ["4", "more", "unknown"]) and
        (ans["transparent_wings"] == "transparent" or contains_any(ans["wing_color_pattern"], ["clear", "golden tint"])) and
        ans["resting_position"] == "outstretched" and
        (contains_any(ans["body_color"], ["brown", "reddish-brown", "yellow", "orange", "other"]) or ans["body_color"] == "unknown") and
        (ans["body_texture_appearance"] == "elongated and slender" or ans["body_texture_appearance"] == "other" or ans["body_texture_appearance"] == "unknown") and
        (ans["num_legs"] in ["6", "unknown"]) and
        (ans["antennae_present"] == "no" or ans["antennae_present"] == "unknown" or (ans["antennae_present"] == "yes" and ans["antennae_shape"] == "small")) and
        (contains_any(ans["eye_color"], ["dark", "red", "brown", "yellow", "green", "other"]) or ans["eye_color"] == "unknown")
    ):
        return "WANDERING GLIDER"

    # COMMON CROW BUTTERFLY
    if (
        ans["wings_visible"] == "yes" and
        (ans["num_wings"] in ["4", "2", "unknown"]) and
        ans["transparent_wings"] == "opaque" and
        contains_any(ans["wing_color_pattern"], ["black with white spots", "black", "white spots"]) and
        ans["resting_position"] == "vertically upright" and
        contains_any(ans["body_color"], ["black", "dark"]) and
        (ans["body_texture_appearance"] in ["soft", "hairy/furry", "unknown"]) and
        (ans["num_legs"] in ["6", "unknown"]) and
        ans["antennae_present"] == "yes" and
        (ans["antennae_shape"] == "clubbed" or ans["antennae_shape"] == "unknown") and ans["antennae_color"] == "black"
    ):
        return "COMMON CROW BUTTERFLY"

    # PLAIN TIGER BUTTERFLY
    if (
        ans["wings_visible"] == "yes" and
        (ans["num_wings"] in ["4", "2", "unknown"]) and
        ans["transparent_wings"] == "opaque" and
        contains_any(ans["wing_color_pattern"], ["orange with black border and white spots", "orange", "black border", "white spots"]) and
        ans["resting_position"] == "vertically upright" and
        contains_any(ans["body_color"], ["orange", "brownish-orange"]) and
        (ans["body_texture_appearance"] in ["soft", "hairy/furry", "unknown"]) and
        (ans["num_legs"] in ["6", "unknown"]) and
        ans["antennae_present"] == "yes" and (ans["antennae_shape"] in ["clubbed", "other", "thread-like"]) and ans["antennae_color"] == "black"
    ):
        return "PLAIN TIGER BUTTERFLY"

    # SUNDOWNER MOTH
    if (
        ans["wings_visible"] == "yes" and
        (ans["num_wings"] in ["2", "4", "unknown"]) and
        ans["transparent_wings"] == "opaque" and
        contains_any(ans["wing_color_pattern"], ["brownish with dark patches", "brown", "grey", "dark patches", "subtle", "uniform", "other", "unknown"]) and
        (ans["resting_position"] in ["flat over body", "tent-like", "unknown"]) and
        (contains_any(ans["body_color"], ["brown", "grey", "black"]) or ans["body_color"] == "unknown") and
        ans["body_texture_appearance"] == "hairy/furry" and
        (ans["num_legs"] in ["6", "unknown"]) and
        ans["antennae_present"] == "yes" and (ans["antennae_shape"] in ["thread-like", "other", "unknown"]) and
        (contains_any(ans["antennae_color"], ["brown", "black"]) or ans["antennae_color"] == "unknown")
    ):
        return "SUNDOWNER MOTH"

    # TROPICAL TIGER MOTH
    if (
        ans["wings_visible"] == "yes" and
        (ans["num_wings"] in ["2", "4", "unknown"]) and
        ans["transparent_wings"] == "opaque" and
        contains_any(ans["wing_color_pattern"], ["orange and yellow", "striped", "spots", "yellow", "orange", "black"]) and
        (ans["resting_position"] in ["tent-like", "flat over body", "unknown"]) and
        contains_any(ans["body_color"], ["yellow", "orange"]) and # Adjusted to match provided images better
        ans["body_texture_appearance"] == "hairy/furry" and
        (ans["num_legs"] in ["6", "unknown"]) and
        ans["antennae_present"] == "yes" and (ans["antennae_shape"] in ["thread-like", "other", "unknown"]) and
        (contains_any(ans["antennae_color"], ["black", "brown"]) or ans["antennae_color"] == "unknown")
    ):
        return "TROPICAL TIGER MOTH"

    # ORIENTAL BEETLE
    if (
        (ans["wings_visible"] == "yes" or ans["wings_visible"] == "no") and # Wings may not be prominent
        (ans["num_wings"] in ["2", "unknown", "n/a"]) and # Hardened forewings cover hindwings, appearing as 2
        (ans["transparent_wings"] == "opaque" or ans["transparent_wings"] == "transparent") and # Elytra opaque, hindwings transparent
        contains_any(ans["wing_color_pattern"], ["brown", "metallic", "darker brown", "other", "unknown"]) and
        ans["resting_position"] == "flat over body" and
        contains_any(ans["body_color"], ["brown", "green", "black", "metallic", "other", "unknown"]) and
        ans["body_texture_appearance"] == "hard and shiny" and
        (ans["num_legs"] in ["6", "unknown"]) and
        ans["antennae_present"] == "yes" and (contains_any(ans["antennae_shape"], ["clubbed", "lamellate", "other"]) or ans["antennae_shape"] == "unknown") and
        (contains_any(ans["antennae_color"], ["brown", "black"]) or ans["antennae_color"] == "unknown")
    ):
        return "ORIENTAL BEETLE"

    # INDIAN RED BUG
    if (
        (ans["wings_visible"] == "yes" or ans["wings_visible"] == "no") and # Some are apterous, others winged
        (ans["num_wings"] in ["2", "unknown", "n/a"]) and
        ans["transparent_wings"] == "opaque" and
        contains_any(ans["wing_color_pattern"], ["red with black spots", "red", "black spots", "uniform"]) and
        ans["resting_position"] == "flat over body" and
        contains_any(ans["body_color"], ["red", "orange"]) and
        ans["body_texture_appearance"] == "soft" and
        (ans["num_legs"] in ["6", "unknown"]) and
        ans["antennae_present"] == "yes" and ans["antennae_shape"] == "thread-like" and ans["antennae_color"] == "black"
    ):
        return "INDIAN RED BUG"

    # INDIAN BEAN BUG
    if (
        (ans["wings_visible"] == "yes" or ans["wings_visible"] == "no") and # Usually winged, but not always visible
        (ans["num_wings"] in ["2", "unknown", "n/a"]) and
        ans["transparent_wings"] == "opaque" and
        contains_any(ans["wing_color_pattern"], ["brown", "uniform", "subtle", "other", "unknown"]) and
        ans["resting_position"] == "flat over body" and
        contains_any(ans["body_color"], ["brown", "dark brown", "other", "unknown"]) and
        ans["body_texture_appearance"] == "elongated and slender" and
        (ans["num_legs"] in ["6", "unknown"]) and
        ans["antennae_present"] == "yes" and ans["antennae_shape"] == "thread-like" and ans["antennae_color"] == "brown"
    ):
        return "INDIAN BEAN BUG"

    # INDIAN POTTER WASP
    if (
        ans["wings_visible"] == "yes" and
        (ans["num_wings"] in ["2", "4", "unknown"]) and # Appears as 2, technically 4
        ans["transparent_wings"] == "transparent" and
        (contains_any(ans["wing_color_pattern"], ["clear", "smoky"]) or ans["wing_color_pattern"] == "unknown") and
        (ans["resting_position"] in ["flat over body", "other", "unknown"]) and
        contains_any(ans["body_color"], ["black", "yellow", "orange", "other"]) and
        ans["body_texture_appearance"] == "elongated with narrow middle part" and
        (ans["num_legs"] in ["6", "unknown"]) and
        ans["antennae_present"] == "yes" and (contains_any(ans["antennae_shape"], ["bent", "elbowed", "other", "3 spikes"]) or ans["antennae_shape"] == "unknown") and # Added 3 spikes for robustness
        (contains_any(ans["antennae_color"], ["yellow", "black"]) or ans["antennae_color"] == "yellow" or ans["antennae_color"] == "unknown")
    ):
        return "INDIAN POTTER WASP"

    # SLENDER MEADOW KATYDID
    if (
        ans["wings_visible"] == "yes" and
        (ans["num_wings"] in ["2", "unknown"]) and
        (ans["transparent_wings"] == "opaque" or ans["transparent_wings"] == "transparent") and
        (contains_any(ans["wing_color_pattern"], ["green", "brown", "greenish"]) or ans["wing_color_pattern"] == "unknown") and
        ans["resting_position"] == "flat over body" and # Often held flat or tent-like
        (contains_any(ans["body_color"], ["green", "brown", "other"]) or ans["body_color"] == "unknown") and
        (ans["body_texture_appearance"] in ["soft", "elongated and slender", "other", "unknown"]) and
        (ans["num_legs"] in ["6", "unknown"]) and
        ans["antennae_present"] == "yes" and (ans["antennae_shape"] in ["very long", "thread-like", "other", "unknown"]) and
        (contains_any(ans["antennae_color"], ["black", "brown"]) or ans["antennae_color"] == "unknown")
    ):
        return "SLENDER MEADOW KATDID"
    else:
       return "UNCERTAIN_SPECIES"
    
# --- Streamlit App Structure ---
st.title("Insect Identification with AI and Human Clarification")
st.write("Upload an image of an insect. The AI will predict the species. If confidence is low, human clarification will be requested.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:

    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")

    # Convert uploaded image to array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    img_array = np.expand_dims(img_resized, axis=0) / 255.0

    # Predict using model
    predictions = model.predict(img_array)
    pred_index = np.argmax(predictions)

    st.session_state.initial_confidence = float(np.max(predictions))
    st.session_state.initial_confidence = initial_confidence
    st.session_state.initial_pred_class = class_names[pred_index]

    st.write(f"Confidence: {st.session_state.initial_confidence*100:.2f}%")

   # ---------- HITL TRIGGER ----------

# Initialize state variable once
    if "show_questions" not in st.session_state:
        st.session_state.show_questions = False


# LOW CONFIDENCE → trigger questions
    if st.session_state.initial_confidence < 0.95:

        st.warning("Low confidence — Human clarification required")

    # turn ON questions permanently
        st.session_state.show_questions = True

# ---------- HIGH CONFIDENCE DIRECT DISPLAY ----------
if st.session_state.initial_confidence >= 0.95:

   st.subheader("AI Prediction")
   st.write(f"Species: {st.session_state.initial_pred_class}")
   st.write(f"Confidence: {st.session_state.initial_confidence*100:.2f}%")

taxonomy_key = st.session_state.initial_pred_class.upper()
   if taxonomy_key in taxonomy:
        st.subheader("Taxonomic Classification")
        for rank, value in taxonomy[taxonomy_key].items():
            st.write(f"**{rank}:** {value}")

# ---------- QUESTION DISPLAY ----------
        if st.session_state.show_questions:
        
            user_answers = ask_questions_streamlit()
        
            if user_answers is not None:
                final_species = rule_based_identification(user_answers)
        
                st.subheader("Refined Identification")
                st.success(final_species.title())
        
                taxonomy_key = final_species.upper()
                if taxonomy_key in taxonomy:
                    st.subheader("Taxonomic Classification")
                    for rank, value in taxonomy[taxonomy_key].items():
                        st.write(f"**{rank}:** {value}")
        
            else:
                st.warning(f"Taxonomy information not found for '{st.session_state.initial_pred_class}'")
