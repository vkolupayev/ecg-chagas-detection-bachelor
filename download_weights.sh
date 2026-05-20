# python -m venv hf_venv
# source hf_venv/bin/activate
# pip install huggingface-hub
# export HF_TOKEN='PASTE THE TOKEN IF NEW'
# hf auth login --token $HF_TOKEN
# hf download Edoardo-BS/hubert-ecg-small --local-dir ./pretrained_models/hubert-ecg-small
# Or with wget:
wget -c https://huggingface.co/PKUDigitalHealth/ECGFounder/blob/main/12_lead_ECGFounder.pth -P pretrained_models/
wget -c https://huggingface.co/Edoardo-BS/hubert-ecg-small/blob/main/model.safetensors -P pretrained_models/hubert-ecg-small/
wget -c https://huggingface.co/Edoardo-BS/hubert-ecg-small/blob/main/config.json -P pretrained_models/hubert-ecg-small/