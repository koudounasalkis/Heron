# Heron - HiErarchical Residual audio-text fusiON

This repository contains the code for the paper **"When Audio Matters: A Lightweight, Hierarchical Fusion Model for Speech and Non-Verbal Emotion Recognition"**, submitted at ICASSP 2026.

> Recognizing human emotion is a complex multimodal task. While text provides a strong semantic baseline, audio and non-verbal vocalizations (NVVs) like laughter or sighs offer crucial paralinguistic cues that can disambiguate or even alter meaning.  
In this work, we demonstrate that audio's primary value lies in disambiguating lexically neutral or ambiguous text. 
We propose *HERON (HiErarchical Residual audio-text fusiON)*, a lightweight architecture that first unifies audio representation from speech and NVV backbones, then injects this auditory context into a text model via residual cross-attention.  
We analyze two paradigms: in a parameter-efficient frozen setting, HERON matches the strongest unimodal baseline with only 7.6M trainable parameters. 
When fully fine-tuned, it establishes a new state-of-the-art with a +3\% absolute F1-score improvement.

The code to reproduce the experiments will be made available upon acceptance.

## Table of Contents

- [Training setup](#training-setup)
- [SpeechLLMs](#speechllms)
- [License](#license)

## Training setup
The model was trained with the following configuration and hyperparameters.

**Architecture:** For all backbone models ([RoBERTa](https://huggingface.co/FacebookAI/roberta-base), [HuBERT](https://huggingface.co/facebook/hubert-base-ls960}), [voc2vec](https://huggingface.co/alkiskoudounas/voc2vec-hubert-ls-pt)), we extract the hidden states from the final transformer layer, which we empirically determined to be the optimal setting. The final classification head is a 2-layer MLP with a hidden dimension of 256 and a dropout rate of 0.1.

**Training:** The model is trained for a maximum of 20 epochs, with an early stopping mechanism that halts training if the validation F1-score does not improve for 5 consecutive epochs. We use the AdamW optimizer with a learning rate of 5e−5 and a standard cross-entropy loss function. A ReduceLROnPlateau scheduler adjusts the learning rate based on the validation loss. We use a batch size of 8 with 2 gradient accumulation steps, resulting in an effective batch size of 16.

**Regularization:** We employ modality dropout with a probability of 0.2 during training. This method randomly sets a modality's entire feature representation to a zero vector to prevent the model from becoming overly reliant on the dominant text signal. Consequently, the model is forced to learn meaningful patterns from the audio cues alone, which improves the fusion module's robustness.

**Settings:** We evaluate two distinct training settings: (1) Frozen Backbones, where only the fusion module and the final MLP head are trained, and (2) Full Fine-tuning, where all parameters of the model, including the three backbones, are fine-tuned end-to-end.

**Hardware:** All experiments are conducted on a single NVIDIA® RTX A6000 GPU.

## SpeechLLMs
We utilize the following SpeechLLMs as strong multimodal baselines:
- Phi4-Multimodal-Instruct: [HF Checkpoint](https://huggingface.co/microsoft/Phi-4-multimodal-instruct)
- MERaLiON-2-10B: [HF Checkpoint](https://huggingface.co/MERaLiON/MERaLiON-2-10B)
- Qwen2Audio-7B-Instruct: [HF Checkpoint](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct)
- Qwen2.5Omni-7B: [HF Checkpoint](https://huggingface.co/Qwen/Qwen2.5-Omni-7B)

**Prompt:** We use the following prompts for the emotion recognition task.

The system prompt is as follows:
    
    ROLE
    You are an expert at recognizing human emotions from speech and nonverbal audio cues.

    TASK
    Given an audio clip of a person speaking, your task is to classify the emotional tone conveyed.
    Choose the most appropriate emotion from the following set: 
    {'sad', 'neutral', 'disgusted', 'surprised', 'angry', 'happy', 'other', 'fearful'}.

    INSTRUCTIONS
    1. Carefully listen to the audio clip and analyze the vocal characteristics.
    2. Consider factors such as pitch, tone, pace, volume, and any nonverbal sounds.
    3. Read the text transcription of the audio to understand the context.
    4. Select the single emotion that best represents the overall emotional tone of the audio.
    5. Respond with only the single emotion label, without any additional text or explanation.

    Output Format
    Provide only the single emotion label as your response, without any additional explanation or context.

The user prompt is as follows:

    Read the following text transcription to help you understand the context: <transcription>. 
    Classify the emotion conveyed in the following audio clip.

For few shot settings, we provide 5 examples of audio clips with their corresponding transcriptions and emotion labels.

## License

This code and the models are released under the Apache 2.0 license. See the [LICENSE](LICENSE) file for more details.
