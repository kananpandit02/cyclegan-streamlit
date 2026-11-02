# Photo-to-Ghibli-Style Translation with CycleGAN

An image-to-image translation project training a CycleGAN for unpaired translation between real-world photographs and an illustrated visual domain, with a Streamlit interface for experimentation.

## Overview
Paired training images are not always available for artistic translation. CycleGAN learns mappings between two image domains using adversarial and cycle-consistency objectives.

## Approach
`Photos ⇄ CycleGAN generators/discriminators ⇄ Illustration domain`
- Two generators learn mappings between image domains
- Discriminators encourage outputs to resemble the target domain
- Cycle-consistency loss helps preserve content through round-trip translation
- Identity loss can help maintain color/content characteristics
- Streamlit provides an interactive application layer

## Tech stack
Python · PyTorch · CycleGAN · GANs · Streamlit · Computer Vision

## Run locally
```bash
git clone https://github.com/kananpandit02/cyclegan-streamlit.git
cd cyclegan-streamlit
pip install -r requirements.txt
streamlit run app.py
```
Review repository dependencies and model weights; GPU availability may be required for inference.

## Repository
[View source code](https://github.com/kananpandit02/cyclegan-streamlit)

**Kanan Pandit** · [Portfolio](https://kananpanditportfolio.netlify.app/) · [LinkedIn](https://linkedin.com/in/kananpandit02)
