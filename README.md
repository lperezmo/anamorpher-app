# Anamorpher Streamlit App

This Streamlit app lets you upload a square image and plant hidden text so that when the image is downsampled the text appears.

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the app:
   ```bash
   streamlit run app.py
   ```
3. Upload an image, enter the hidden text and choose the embedding method. The generated adversarial image can be downloaded.

The heavy lifting is done by the scripts in `adversarial_generators` derived from the [anamorpher](https://github.com/lperezmo/anamorpher) project.
