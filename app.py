import os
import glob
import subprocess
import tempfile
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st


# ---------- Text image helpers (adapted from backend/app.py) ----------

def wrap_text_to_fit(text: str, font, draw, max_width: int) -> list:
    """Wrap text to fit within specified width"""
    words = text.split()
    lines = []
    current_line = []

    for word in words:
        test_line = ' '.join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        text_width = bbox[2] - bbox[0]

        if text_width <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
            else:
                lines.append(word)

    if current_line:
        lines.append(' '.join(current_line))

    return lines


def create_text_image(text: str, size: int = 1092, font_size: int = 32,
                       alignment: str = 'center') -> tuple[np.ndarray, bool]:
    """Create a square text image with specified text, font size, and alignment."""
    image = Image.new('RGB', (size, size), color='#333333')
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype('Arial.ttf', font_size)
    except OSError:
        try:
            font_paths = [
                '/System/Library/Fonts/Arial.ttf',
                '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                'C\\\Windows\\Fonts\\arial.ttf'
            ]
            font = None
            for font_path in font_paths:
                if os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, font_size)
                    break
            if font is None:
                font = ImageFont.load_default()
        except OSError:
            font = ImageFont.load_default()

    margin = 10
    text_area_width = size - 2 * margin
    text_area_height = size - 2 * margin

    wrapped_lines = wrap_text_to_fit(text, font, draw, text_area_width)

    line_height = draw.textbbox((0, 0), 'Ay', font=font)[3] - draw.textbbox((0, 0), 'Ay', font=font)[1]
    total_height = len(wrapped_lines) * line_height

    text_overflowed = total_height > text_area_height

    if alignment in ['center', 'top', 'bottom']:
        if alignment == 'center':
            start_y = max(margin, (size - total_height) // 2)
        elif alignment == 'top':
            start_y = margin
        else:
            start_y = max(margin, size - margin - total_height)

        for i, line in enumerate(wrapped_lines):
            if start_y + i * line_height + line_height > size - margin:
                break
            y = start_y + i * line_height
            bbox = draw.textbbox((0, 0), line, font=font)
            line_width = bbox[2] - bbox[0]
            x = (size - line_width) // 2
            draw.text((x, y), line, font=font, fill='#00b002')

    elif alignment in ['left', 'right']:
        start_y = max(margin, (size - total_height) // 2)
        for i, line in enumerate(wrapped_lines):
            if start_y + i * line_height + line_height > size - margin:
                break
            y = start_y + i * line_height
            if alignment == 'left':
                x = margin
            else:
                bbox = draw.textbbox((0, 0), line, font=font)
                line_width = bbox[2] - bbox[0]
                x = size - margin - line_width
            draw.text((x, y), line, font=font, fill='#00b002')

    else:  # corner alignments
        if alignment.startswith('top'):
            start_y = margin
        else:
            start_y = max(margin, size - margin - total_height)
        for i, line in enumerate(wrapped_lines):
            if start_y + i * line_height + line_height > size - margin:
                break
            y = start_y + i * line_height
            if alignment.endswith('left'):
                x = margin
            else:
                bbox = draw.textbbox((0, 0), line, font=font)
                line_width = bbox[2] - bbox[0]
                x = size - margin - line_width
            draw.text((x, y), line, font=font, fill='#00b002')

    return np.array(image), text_overflowed


# ---------- Image preprocessing ----------

def prepare_decoy_image(img: Image.Image) -> Image.Image:
    """Center-crop to square and ensure dimensions divisible by 4."""
    w, h = img.size
    if w != h:
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        img = img.crop((left, top, left + min_dim, top + min_dim))

    size = (img.width // 4) * 4
    if img.width != size:
        left = (img.width - size) // 2
        top = (img.height - size) // 2
        img = img.crop((left, top, left + size, top + size))

    return img


# ---------- Adversarial generation ----------

def generate_adversarial(decoy_img: Image.Image, text: str, method: str,
                          lam: float, eps: float, gamma: float,
                          dark_frac: float | None = None,
                          offset: int | None = None) -> tuple[Image.Image, Image.Image]:
    """Generate adversarial image and target image."""
    if decoy_img.width != decoy_img.height:
        raise ValueError('Image must be square.')
    if decoy_img.width % 4 != 0:
        raise ValueError('Image dimensions must be divisible by 4.')

    target_size = decoy_img.width // 4
    target_arr, _ = create_text_image(text, target_size)

    with tempfile.TemporaryDirectory() as tmpdir:
        decoy_path = os.path.join(tmpdir, 'decoy.png')
        target_path = os.path.join(tmpdir, 'target.png')
        decoy_img.save(decoy_path)
        Image.fromarray(target_arr).save(target_path)

        # Resolve the directory containing the generator scripts. Using abspath ensures
        # we don't end up with a relative path (e.g. 'adversarial_generators') which,
        # when combined with cwd=tmpdir in subprocess.run, would incorrectly look for
        # the scripts inside the temporary directory instead of the project.
        script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'adversarial_generators')
        if not os.path.isdir(script_dir):
            raise RuntimeError(f'Generator scripts directory not found: {script_dir}')
        if method == 'bicubic':
            script = 'bicubic_gen_payload.py'
            cmd = [
                'python3', os.path.join(script_dir, script),
                '--decoy', decoy_path,
                '--target', target_path,
                '--lam', str(lam),
                '--eps', str(eps),
                '--gamma', str(gamma),
                '--dark-frac', str(dark_frac if dark_frac is not None else 0.3)
            ]
            pattern = 'adv*.png'
        elif method == 'bilinear':
            script = 'bilinear_gen_payload.py'
            cmd = [
                'python3', os.path.join(script_dir, script),
                '--decoy', decoy_path,
                '--target', target_path,
                '--lam', str(lam),
                '--eps', str(eps),
                '--gamma', str(gamma),
                '--dark-frac', str(dark_frac if dark_frac is not None else 0.3)
            ]
            pattern = 'adv_bilinear*.png'
        else:
            script = 'nearest_gen_payload.py'
            cmd = [
                'python3', os.path.join(script_dir, script),
                '--decoy', decoy_path,
                '--target', target_path,
                '--lam', str(lam),
                '--eps', str(eps),
                '--gamma', str(gamma),
                '--offset', str(offset if offset is not None else 2)
            ]
            pattern = 'advNN*.png'

        result = subprocess.run(cmd, cwd=tmpdir, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f'Generator failed (exit {result.returncode}).\nCommand: {" ".join(cmd)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}')

        adv_files = glob.glob(os.path.join(tmpdir, pattern))
        if not adv_files:
            raise RuntimeError('No adversarial image generated.')
        adv_img = Image.open(adv_files[0]).convert('RGB')

    return Image.fromarray(target_arr), adv_img


# ---------- Streamlit UI ----------

st.title('Anamorpher Streamlit App')

uploaded = st.file_uploader('Upload image (auto-cropped to square)', type=['png', 'jpg', 'jpeg'])

text = st.text_input('Hidden text', 'secret')
method = st.selectbox('Method', ['bicubic', 'bilinear', 'nearest'])

if method in ['bicubic', 'bilinear']:
    lam = st.number_input('λ (lam)', value=0.25 if method == 'bicubic' else 1.0, step=0.01)
    eps = st.number_input('ε (eps)', value=0.0, step=0.01)
    gamma = st.number_input('γ (gamma)', value=1.0 if method == 'bicubic' else 0.9, step=0.1)
    dark_frac = st.slider('dark_frac', 0.0, 1.0, 0.3)
    offset = None
else:
    lam = st.number_input('λ (lam)', value=0.25, step=0.01)
    eps = st.number_input('ε (eps)', value=0.0, step=0.01)
    gamma = st.number_input('γ (gamma)', value=1.0, step=0.1)
    offset = st.number_input('offset', min_value=0, max_value=10, value=2)
    dark_frac = None

if uploaded and st.button('Generate'):
    try:
        decoy = Image.open(uploaded).convert('RGB')
        decoy = prepare_decoy_image(decoy)
        target_img, adv_img = generate_adversarial(decoy, text, method, lam, eps, gamma, dark_frac, offset)

        col1, col2 = st.columns(2)
        col1.image(decoy, caption='Original', use_container_width=True)
        col2.image(adv_img, caption='Adversarial', use_container_width=True)


        st.subheader('Target Text Image')
        st.image(target_img, caption='Target', use_container_width=True)

        st.subheader('Downscaled Preview')
        preview = adv_img.resize((adv_img.width // 4, adv_img.height // 4), Image.LANCZOS)
        st.image(preview, caption='Adversarial (downscaled)', use_container_width=True)


        buf = BytesIO()
        adv_img.save(buf, format='PNG')
        st.download_button('Download adversarial image', buf.getvalue(), file_name='adversarial.png', mime='image/png')
    except Exception as e:
        st.error(f'Error: {e}')
