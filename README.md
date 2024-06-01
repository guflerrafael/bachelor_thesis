# Hybrid Voice Disorder Detection Sytem
This repository includes all the necessary code and data for classifying audio samples as either healthy or pathological voices. Detailed explanations of each part of the code are provided within the code itself, complemented by a more comprehensive discussion in the associated bachelor thesis.

For code usage, `main_script.py` serves as a guide. Although the descriptions may not always be elaborate, the script is well-organized and straightforward, ensuring that users can easily execute the code.

### Important considerations include:
- `requirements.txt` lists all the dependencies required. Ensure your environment is compatible with these dependencies.
- The zip file containing the voice samples should be placed in the `data` directory and named as specified in the script.
- The device used for feature extraction has to be defined in `feature_extraction.py`.
- Currently, the Jupyter notebook is not functional due to crashes occurring during model training. If certain parts of the code are unnecessary (e.g., if model training has already been completed), those sections should be commented out to avoid execution errors.
