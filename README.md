# Sky LiveKit Agent Perplexica

Sky LiveKit Agent Perplexica is a fully local, free solution that integrates LiveKit with advanced internet search capabilities. It leverages a local Perplexica instance with function calling to retrieve and summarise search results in natural, conversational language, complete with source attribution.

## Features

- **Local Search:** Conducts internet searches entirely on local services.
- **Free of API Costs:** No need for external API calls or subscription fees.
- **Speech Recognition:** Uses [Speaches](https://github.com/speaches-ai/speaches) for accurate speech transcription.
- **Language Processing:** Employs Ollama (Qwen 2.5) for robust language processing.
- **Text-to-Speech:** Uses [Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI) with custom TTS implementation.

## Required Components

- [Perplexica](https://github.com/ItzCrazyKns/Perplexica) - Local search engine
- Ollama - for local models
- [LiveKit Server](https://docs.livekit.io/home/self-hosting/local/) - WebRTC server
- [Voice Assistant Frontend](https://github.com/livekit-examples/voice-assistant-frontend) - LiveKit frontend interface
- [Speaches](https://github.com/speaches-ai/speaches) - Speech recognition
- [Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI) - Text-to-speech service
- The custom tts.py in this repository
