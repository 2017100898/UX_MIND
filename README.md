## MIND (Multimodal INteractive Dashboard)

0. Download the model folder from [Google Drive](https://drive.google.com/drive/folders/1djwUiAWDnatcuyIDgYtblJyOTAx_YTBW?usp=sharing) and place it in the cloned UX_MIND folder.
1. Install the package through requirements.txt.
    * A version compatibility issue exists between the current packages.
    * Therefore, it is recommended that you check and install the imported packages rather than installing all of the packages using txt file.
2. Run Flask using the prompt window.
    * When using real-time EEG (you need emotive lab stream layer (lsl)): `python app.py`
    * When using saved EEG data: `python app_loaded.py`
3. Open a new prompt window and run `streamlit run dash.py`
 