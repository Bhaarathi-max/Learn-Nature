import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import os
import cv2
class_names = [
    'BEAN_BUG',
    'COMMON_CROW_BUTTERFLY',
    'INDIAN_RED_BUG',
    'ORIENTAL_BEETLE',
    'PLAIN_TIGER_BUTTERFLY',
    'POTTER_WASP',
    'SLENDER_MEADOW_KATYKID',
    'SUNDOWNER_MOTH',
    'TROPICAL_TIGER_MOTH',
    'WANDERING_GLIDER']

# --- Global Variables ---
CONFIDENCE_THRESHOLD = 0.95
IMG_HEIGHT = 224
IMG_WIDTH = 224

# --- Paths ---
data
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

# --- Dynamic Class Name Loading ---
@st.cache_data
def get_class_names():
    if 'dataset_path' not in globals() or not os.path.exists(dataset_path):
        st.error("Dataset path not found. Please ensure it's correctly set.")
        return []

    dummy_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    try:
        dummy_generator = dummy_datagen.flow_from_directory(
            dataset_path,
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=1,
            class_mode='categorical',
            shuffle=False,
            subset=None
        )
        class_indices = dummy_generator.class_indices
        idx_to_class = {v: k for k, v in class_indices.items()}
        class_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
        return class_names
    except Exception as e:
        st.error(f"Error generating class names from directory: {e}")
        return []

class_names = get_class_names()

# --- Taxonomy Dictionary ---
taxonomy = {
    'BEAN_BUG': {
        'common_name': 'Bean Bug',
        'species': 'Riptortus pedestris',
        'genus': 'Riptortus',
        'order': 'Hemiptera',
        'class': 'Insecta',
        'phylum': 'Arthropoda',
        'kingdom': 'Animalia'
    },
    'COMMON_CROW_BUTTERFLY': {
        'common_name': 'Common Crow Butterfly',
        'species': 'Euploea core',
        'genus': 'Euploea',
        'order': 'Lepidoptera',
        'class': 'Insecta',
        'phylum': 'Arthropoda',
        'kingdom': 'Animalia'
    },
    'INDIAN_RED_BUG': {
        'common_name': 'Indian Red Bug',
        'species': 'Dysdercus cingulatus',
        'genus': 'Dysdercus',
        'order': 'Hemiptera',
        'class': 'Insecta',
        'phylum': 'Arthropoda',
        'kingdom': 'Animalia'
    },
    'ORIENTAL_BEETLE': {
        'common_name': 'Oriental Beetle',
        'species': 'Anomala orientalis',
        'genus': 'Anomala',
        'order': 'Coleoptera',
        'class': 'Insecta',
        'phylum': 'Arthropoda',
        'kingdom': 'Animalia'
    },
    'PLAIN_TIGER_BUTTERFLY': {
        'common_name': 'Plain Tiger Butterfly',
        'species': 'Danaus chrysippus',
        'genus': 'Danaus',
        'order': 'Lepidoptera',
        'class': 'Insecta',
        'phylum': 'Arthropoda',
        'kingdom': 'Animalia'
    },
    'POTTER_WASP': {
        'common_name': 'Potter Wasp',
        'species': 'Delta dimidiatipenne',
        'genus': 'Delta',
        'order': 'Hymenoptera',
        'class': 'Insecta',
        'phylum': 'Arthropoda',
        'kingdom': 'Animalia'
    },
    'SLENDER_MEADOW_KATYKID': {
        'common_name': 'Slender Meadow Katydid',
        'species': 'Conocephalus gracilis',
        'genus': 'Conocephalus',
        'order': 'Orthoptera',
        'class': 'Insecta',
        'phylum': 'Arthropoda',
        'kingdom': 'Animalia'
    },
    'SUNDOWNER_MOTH': {
        'common_name': 'Sundowner Moth',
        'species': 'Macroglossum stellatarum',
        'genus': 'Macroglossum',
        'order': 'Lepidoptera',
        'class': 'Insecta',
        'phylum': 'Arthropoda',
        'kingdom': 'Animalia'
    },
    'TROPICAL_TIGER_MOTH': {
        'common_name': 'Tropical Tiger Moth',
        'species': 'Asota caricae',
        'genus': 'Asota',
        'order': 'Lepidoptera',
        'class': 'Insecta',
        'phylum': 'Arthropoda',
        'kingdom': 'Animalia'
    },
    'WANDERING_GLIDER': {
        'common_name': 'Wandering Glider',
        'species': 'Pantala flavescens',
        'genus': 'Pantala',
        'order': 'Odonata',
        'class': 'Insecta',
        'phylum': 'Arthropoda',
        'kingdom': 'Animalia'
    }
}

# --- Streamlit-compatible ask_questions function ---
def ask_questions_streamlit():
    st.subheader("Please answer the following questions to help identify the insect:")

    # Initialize answers in session state if not present
    if 'qa_answers' not in st.session_state:
        st.session_state.qa_answers = {}

    # Use a form to group questions and submit at once
    with st.form("clarification_form"):
        st.session_state.qa_answers["wings_visible"] = st.radio(
            "Q1: Does the insect have *visible* wings or wing structures?",
            options=["yes", "no", "unknown"],
            key="wings_visible_q"
        )

        if st.session_state.qa_answers["wings_visible"] == "yes":
            st.session_state.qa_answers["num_wings"] = st.radio(
                "Q2: How many distinct wings are clearly visible? (Consider if forewings cover hindwings)",
                options=["2", "4", "more", "unknown"],
                key="num_wings_q"
            )
            st.session_state.qa_answers["transparent_wings"] = st.radio(
                "Q3: Are the wings largely transparent/clear or mostly opaque/colored?",
                options=["transparent", "opaque", "unknown"],
                key="transparent_wings_q"
            )
            st.session_state.qa_answers["wing_color_pattern"] = st.text_input(
                "Q4: Describe the dominant wing color or pattern (e.g., 'black with white spots', 'orange with black border and white spots', 'clear', 'golden tint', 'other/unknown'):",
                key="wing_color_pattern_q"
            ).lower()
            st.session_state.qa_answers["resting_position"] = st.radio(
                "Q5: How does it typically hold its wings at rest (e.g., 'flat over body', 'tent-like', 'vertically upright', 'outstretched', 'other/unknown')?",
                options=['flat over body', 'tent-like', 'vertically upright', 'outstretched', 'other/unknown'],
                key="resting_position_q"
            )
        else:
            st.session_state.qa_answers["num_wings"] = "n/a"
            st.session_state.qa_answers["transparent_wings"] = "n/a"
            st.session_state.qa_answers["wing_color_pattern"] = "n/a"
            st.session_state.qa_answers["resting_position"] = "n/a"

        st.session_state.qa_answers["body_color"] = st.text_input(
            "Q6: What is the insect's dominant body color (e.g., 'red', 'brown', 'black', 'green', 'yellow', 'orange', 'other/unknown')?",
            key="body_color_q"
        ).lower()
        st.session_state.qa_answers["body_texture_appearance"] = st.radio(
            "Q7: Is the body predominantly 'hard and shiny', 'soft', 'hairy/furry', 'elongated with narrow middle part' (wasp-waisted), or 'elongated and slender' (dragonfly-like)?",
            options=['hard and shiny', 'soft', 'hairy/furry', 'elongated with narrow middle part', 'elongated and slender', 'other/unknown'],
            key="body_texture_appearance_q"
        )
        st.session_state.qa_answers["num_legs"] = st.radio(
            "Q8: Number of visible legs",
            options=["6", "8", "more", "n/a", "unknown"],
            key="num_legs_q"
        )

        st.session_state.qa_answers["antennae_present"] = st.radio(
            "Q9: Are antennae clearly visible?",
            options=["yes", "no", "unknown"],
            key="antennae_present_q"
        )

        if st.session_state.qa_answers["antennae_present"] == "yes":
            st.session_state.qa_answers["antennae_shape"] = st.radio(
                "Q10: What is the shape of the antennae?",
                options=["clubbed", "thread-like", "bent", "very long", "small", "3 spikes", "other", "unknown"],
                key="antennae_shape_q"
            )
            st.session_state.qa_answers["antennae_color"] = st.text_input(
                "Q11: What is the main color of the antennae (black/brown/orange/other/unknown):",
                key="antennae_color_q"
            ).lower()
        else:
            st.session_state.qa_answers["antennae_shape"] = "n/a"
            st.session_state.qa_answers["antennae_color"] = "n/a"

        st.session_state.qa_answers["eye_color"] = st.text_input(
            "Q12: What is the predominant eye color (dark/red/yellow/brown/green/other/n/a/unknown):",
            key="eye_color_q"
        ).lower()

        submitted = st.form_submit_button("Submit Clarification")

    return st.session_state.qa_answers if submitted else None

# --- rule_based_identification function (adapted to new question structure) ---
def rule_based_identification(ans):
    def contains_any(user_answer, keywords):
        if not isinstance(user_answer, str):
            user_answer = str(user_answer)
        return any(keyword.lower() in user_answer.lower() for keyword in keywords)

    # WANDERING_GLIDER
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
        return "WANDERING_GLIDER"

    # COMMON_CROW_BUTTERFLY
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
        return "COMMON_CROW_BUTTERFLY"

    # PLAIN_TIGER_BUTTERFLY
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
        return "PLAIN_TIGER_BUTTERFLY"

    # SUNDOWNER_MOTH
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
        return "SUNDOWNER_MOTH"

    # TROPICAL_TIGER_MOTH
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
        return "TROPICAL_TIGER_MOTH"

    # ORIENTAL_BEETLE
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
        return "ORIENTAL_BEETLE"

    # INDIAN_RED_BUG
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
        return "INDIAN_RED_BUG"

    # BEAN_BUG
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
        return "BEAN_BUG"

    # POTTER_WASP
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
        return "POTTER_WASP"

    # SLENDER_MEADOW_KATYKID
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
        return "SLENDER_MEADOW_KATYKID"
    else:
       return "UNCERTAIN_SPECIES"

# --- Streamlit App Structure ---
st.title("Insect Identification with AI and Human Clarification")
st.write("Upload an image of an insect, and our AI model will try to identify it. If the confidence is low, we'll ask for your help!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_resized = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    img_array = np.expand_dims(img_resized, axis=0) / 255.0

    predictions = model.predict(img_array)
    pred_index = np.argmax(predictions)
    initial_confidence = np.max(predictions)

    st.write("Predicted index:", pred_index)
    st.write("Total class names:", len(class_names))

if pred_index < len(class_names):
    initial_pred_class = class_names[pred_index]

    st.subheader("Initial AI Prediction:")
    st.write(f"**Predicted Species:** {initial_pred_class}")
    st.write(f"**Confidence:** {initial_confidence * 100:.2f}%")

else:
    st.error("Index mismatch! Model predicted an index outside class list.")
    st.stop()

    # Logic for human clarification
    if initial_confidence < CONFIDENCE_THRESHOLD:
        st.warning("Low confidence prediction – Human clarification required.")
        st.write("The AI needs your help to make a more accurate identification.")

        user_answers = ask_questions_streamlit()

        if user_answers:
            final_identified_species = rule_based_identification(user_answers)
            st.subheader("Refined Identification (based on your input):")
            if final_identified_species != "UNCERTAIN_SPECIES":
                st.success(f"**{final_identified_species.replace('_', ' ').title()}**")
                taxonomy_key = final_identified_species.upper()
                if taxonomy_key in taxonomy:
                    st.subheader("Taxonomic Hierarchy:")
                    for level, value in taxonomy[taxonomy_key].items():
                        st.write(f"  - **{level.replace('_', ' ').title()}:** {value}")
                else:
                    st.warning(f"Taxonomy information not found for '{final_identified_species.replace('_', ' ').title()}'")
            else:
                st.info("We couldn't confidently identify the species even with your input. More specific details might be needed.")
    else:
        st.success("High confidence prediction – Result accepted.")
        st.subheader("Final Identified Species:")
        st.success(f"**{initial_pred_class.replace('_', ' ').title()}**")

        taxonomy_key = initial_pred_class.upper()
        if taxonomy_key in taxonomy:
            st.subheader("Taxonomic Hierarchy:")
            for level, value in taxonomy[taxonomy_key].items():
                st.write(f"  - **{level.replace('_', ' ').title()}:** {value}")
        else:
            st.warning(f"Taxonomy information not found for '{initial_pred_class.replace('_', ' ').title()}'")
