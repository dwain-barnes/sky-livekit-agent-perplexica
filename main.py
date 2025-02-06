# main.py

import logging
import os
from dotenv import load_dotenv
import aiohttp
from typing import Annotated
import re 
# LiveKit Agent Imports
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.pipeline import AgentCallContext, VoicePipelineAgent

# Plugin Imports
from livekit.plugins import openai, silero, turn_detector

# Import the TTS class with Kokoro support
from livekit.plugins.openai.tts import TTS

# Load environment variables
load_dotenv(dotenv_path=".env.local")

logger = logging.getLogger("voice-agent")
logger.setLevel(logging.DEBUG)

class AssistantFnc(llm.FunctionContext):
    """
    Defines a set of functions that the assistant can execute.
    Includes an 'internet_search' function that calls a local Perplexica instance
    and formats results for natural TTS output.
    """

    def _format_for_speech(self, message: str, sources: list) -> str:
        """
        Formats search results in a natural, conversational way suitable for TTS.
        """
        # Clean the main message of any markdown or special characters
        clean_message = re.sub(r'\*+', '', message)
        clean_message = re.sub(r'[\[\]\(\)\{\}]', '', clean_message)
        
        # Format the response in a conversational way
        speech_text = f"Based on my search, {clean_message}"
        
        # Add source attribution in a natural way
        if sources:
            speech_text += "\n\nThis information comes from "
            source_titles = []
            for source in sources[:2]:  # Limit to top 2 sources for brevity
                title = source.get("metadata", {}).get("title", "")
                if title:
                    clean_title = re.sub(r'\*+', '', title)
                    clean_title = re.sub(r'[\[\]\(\)\{\}]', '', clean_title)
                    source_titles.append(clean_title)
            
            if source_titles:
                if len(source_titles) == 1:
                    speech_text += f"an article titled {source_titles[0]}"
                else:
                    speech_text += f"articles including {source_titles[0]} and {source_titles[1]}"
        
        return speech_text

    @llm.ai_callable()
    async def internet_search(
        self,
        query: Annotated[
            str,
            llm.TypeInfo(
                description="The search query for performing an internet search using the Perplexica API."
            ),
        ],
    ):
        """
        Performs an internet search using a local Perplexica instance and formats results
        for natural speech output.
        """
        agent = AgentCallContext.get_current().agent

        # Send a natural-sounding filler message
        if not agent.chat_ctx.messages or agent.chat_ctx.messages[-1].role != "assistant":
            filler_message = f"Let me look that up for you."
            logger.info(f"Sending filler message: {filler_message}")
            await agent.say(filler_message, add_to_chat_ctx=True)

        logger.info(f"Performing internet search for query: {query}")

        search_url = "http://localhost:3001/api/search"
        payload = {
            "chatModel": {
                "provider": "ollama",
                "model": "qwen2.5:14b-instruct-8k"
            },
            "embeddingModel": {
                "provider": "ollama",
                "model": "nomic-embed-text:latest"
            },
            "optimizationMode": "speed",
            "focusMode": "webSearch",
            "query": query,
            "history": []
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(search_url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    message = data.get("message", "")
                    sources = data.get("sources", [])

                    # Format the results for natural speech
                    speech_text = self._format_for_speech(message, sources)
                    
                    # Keep a more detailed version for chat context
                    chat_text = speech_text + "\n\nSources:\n" + "\n".join(
                        f"{i+1}. {source.get('metadata', {}).get('title', 'No Title')} - {source.get('metadata', {}).get('url', '')}"
                        for i, source in enumerate(sources)
                    )

                    logger.info(f"Search results formatted for speech: {speech_text}")
                    return {"query": query, "speech_results": speech_text, "chat_results": chat_text}
                else:
                    error_msg = f"I'm sorry, but I wasn't able to complete the search successfully."
                    logger.error(f"Search failed with status code: {response.status}")
                    return {"query": query, "speech_results": error_msg, "chat_results": error_msg}


def prewarm(proc: JobProcess):
    """
    Runs once before the agent starts to load heavy models (e.g., VAD).
    """
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            """You are Sky, an advanced AI assistant created by Dwain Barnes. You run entirely locally using containerized services including Faster Whisper for speech recognition, Ollama for language processing, and Kokoro-82M for text-to-speech synthesis. You can search the internet using the function internet_search and provide a summary of the search for the user.

Core Attributes:
- You are helpful, friendly, and engaging while maintaining professionalism.
- You communicate clearly and directly.
- You have a warm, natural voice powered by Kokoro-82M TTS.
- You process and respond to both text and voice inputs seamlessly.
- You can search the internet using Searxng

Voice Output Guidelines:
- Never read out special characters, formatting symbols, or markdown syntax (like asterisks, brackets, etc.)
- When reading search results or any text, ignore and skip over any *asterisks*, [brackets], or other formatting symbols
- Present information in natural, conversational language
- Treat any asterisks or special characters as formatting instructions only, not as text to be spoken
- When seeing asterisks around words (like *this*), simply read the word naturally without emphasis or mentioning the asterisks

Technical Capabilities:
- Speech Recognition via Faster Whisper
- Language Processing powered by Ollama
- Local Voice Synthesis using Kokoro-82M TTS
- Local operation through containerized services

Interaction Style:
- Engaging and helpful
- Adaptive response length and detail
- Maintains context and remembers conversation details

Response Format:
- Clear, well-structured text
- Natural and properly paced voice responses
- No special characters or formatting symbols in spoken output"""
        ),
    )

    # Connect to the room and auto-subscribe to audio.
    logger.info(f"Connecting to room: {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for at least one participant in the room.
    participant = await ctx.wait_for_participant()
    logger.info(f"Participant joined: {participant.identity}")

    # 1) Speech-to-Text (STT) with OpenAI + FasterWhisper.
    stt_plugin = openai.STT.with_faster_whisper(
        model="Systran/faster-distil-whisper-large-v3"
    )

    # 2) Language Model (LLM) from a custom local endpoint.
    llm_plugin = openai.LLM(
        base_url="http://localhost:11434/v1",  # Example local endpoint.
        api_key=os.environ.get("12343"),        # Your custom API key.
        model="qwen2.5:14b-instruct-8k",
    )

    # 3) Text-to-Speech (TTS) using Kokoro.
    tts_plugin = TTS.create_kokoro_client(
        model="kokoro",                  # Local placeholder model name.
        voice="af_sky",                  # Example voice.
        speed=1.0,
        base_url="http://localhost:8880/v1",  # Kokoro TTS endpoint.
        api_key="not-needed",            # Typically not needed for local Kokoro.
    )

    # 4) Create the function calling context (adds internet search functionality).
    fnc_ctx = AssistantFnc()

    # 5) Create the VoicePipelineAgent with the function calling context.
    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=stt_plugin,
        llm=llm_plugin,
        tts=tts_plugin,
        fnc_ctx=fnc_ctx,
        chat_ctx=initial_ctx,
        turn_detector=turn_detector.EOUModel(),  # End-of-utterance model.
    )

    # Start the agent on the room with the participant.
    agent.start(ctx.room, participant)

    # Greet the participant.
    await agent.say("Hey, how can I help you today?", allow_interruptions=True)


if __name__ == "__main__":
    # Run the app with your worker configuration.
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
