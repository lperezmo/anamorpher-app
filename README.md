# Anamorpher Streamlit App

This Streamlit app lets you upload an image and plant hidden text so that when the image is downsampled the text appears. Uploaded images are automatically center-cropped to a square with dimensions divisible by 4.


## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the app:
   ```bash
   streamlit run app.py
   ```
3. Upload an image, enter the hidden text and choose the embedding method. After generation you'll see side-by-side previews of the original and edited image, a downscaled version revealing the hidden text, and a download button for the adversarial image.


The heavy lifting is done by the scripts in `adversarial_generators` derived from the [anamorpher](https://github.com/lperezmo/anamorpher) project.
