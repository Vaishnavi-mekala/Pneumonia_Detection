"""
streamlit_app.py - Pneumonia Detection Web App

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
from pathlib import Path
import sys
import io

try:
    from fastai.vision.all import load_learner, PILImage
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "fastai"])
    from fastai.vision.all import load_learner, PILImage

# Page configuration
st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🫁",
    layout="centered",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_model(model_path: str = 'models/pneumonia_model.pkl'):
    """Load the trained model (cached for performance)."""
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f'Model not found at {p}. Run train.py first.')
    return load_learner(p)


def main():
    st.title('🫁 Pneumonia Detection System')
    st.markdown('---')
    st.write('Upload a chest X-ray image to classify it as **Normal** or **Pneumonia**.')
    
    # Sidebar
    with st.sidebar:
        st.header('About')
        st.info('This model uses ResNet50 and FastAI to classify chest X-rays.')
        st.markdown('---')
        st.subheader('Instructions:')
        st.write('1. Upload a chest X-ray image (JPG, PNG, JPEG)')
        st.write('2. Wait for the model to process')
        st.write('3. View the prediction and confidence score')
    
    # Model loading
    try:
        learner = load_model()
    except FileNotFoundError as e:
        st.error(f'❌ Model Error: {e}')
        st.info('To train the model, run: `python3 train.py`')
        return
    except Exception as e:
        st.error(f'❌ Unexpected Error: {e}')
        return
    
    # File upload
    uploaded_file = st.file_uploader('Choose a chest X-ray image', type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader('Uploaded Image')
            image_data = uploaded_file.read()
            st.image(image_data, use_column_width=True)
        
        # Make prediction
        with col2:
            st.subheader('Analysis')
            
            try:
                # Create PIL image from bytes
                import io
                from PIL import Image
                img = Image.open(io.BytesIO(image_data))
                
                # Convert to FastAI PILImage
                fastai_img = PILImage.create(img)
                
                # Get prediction
                pred, idx, probs = learner.predict(fastai_img)
                
                confidence = float(probs[idx])
                
                # Display prediction
                st.success(f'**Prediction:** {pred}')
                st.metric(
                    label='Confidence Score',
                    value=f'{confidence:.4f}',
                    delta=f'{confidence*100:.2f}%'
                )
                
                # Display all probabilities
                st.markdown('**All Probabilities:**')
                prob_data = {
                    str(class_name): float(probs[i])
                    for i, class_name in enumerate(learner.dls.vocab)
                }
                
                for class_name, prob in prob_data.items():
                    st.write(f'{class_name}: {prob:.4f} ({prob*100:.2f}%)')
                    st.progress(prob)
                
                # Risk assessment
                st.markdown('---')
                st.subheader('⚠️ Clinical Note')
                if pred == 'PNEUMONIA':
                    st.warning('🔴 Model predicts PNEUMONIA. Please consult a radiologist.')
                else:
                    st.success('🟢 Model predicts NORMAL. But always seek professional medical advice.')
                
            except Exception as e:
                st.error(f'❌ Prediction Error: {e}')
                st.error('Make sure the image is a valid chest X-ray.')
    
    # Footer
    st.markdown('---')
    st.markdown(
        '⚠️ **Disclaimer:** This model is for demonstration purposes only. '
        'Always consult qualified medical professionals for diagnosis.'
    )


if __name__ == '__main__':
    main()
