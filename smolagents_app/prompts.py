"""Prompt templates for the smolagents runner."""

FINAL_ANSWER_TOOL_DESCRIPTION = "Provides a final answer to the given problem with optional formatting."

WEB_SEARCH_TOOL_DESCRIPTION = (
    "Performs a duckduckgo web search based on your query (think a Google search) "
    "then returns the top search results."
)

MULTIMODAL_TOOL_DESCRIPTION = """
Unified tool for processing audio, video, and image files.
Supports transcription, analysis, content extraction, captioning, and cross-modal tasks.
Handles common formats: mp3, wav, m4a, mp4, avi, mov, jpg, png, gif, etc.
OpenAI mode uses Responses API for images and the Audio Transcriptions endpoint for audio/video.
NOTE: PDFs are NOT supported by this tool (use Docling for PDF processing).
""".strip()

FORMAT_PROMPT_TEMPLATE = """Extract the exact answer from the response below.

Question: {question}
Response: {response}

Provide your reasoning, then the exact answer."""

GAIA_SYSTEM_PROMPT = """You are a general AI assistant. I will ask you a question. Report your thoughts.
IMPORTANT:
- In the last step of your reasoning, if you think your reasoning is not able to answer the question, answer the question directly with your internal reasoning, without using the visit_webpage tool.
- Finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.
"""

MULTIMODAL_TASK_PROMPTS = {
    "analyze": {
        "audio": "Analyze this audio file. Describe the content, identify sounds, speech, music, and any notable audio characteristics.",
        "video": "Analyze this video comprehensively. Describe visual content, actions, audio elements, and their relationship.",
        "image": "Analyze this image in detail. Describe objects, scenes, text, colors, composition, and any notable features.",
    },
    "transcribe": {
        "audio": "Transcribe all speech in this audio file accurately. Include speaker changes if multiple speakers.",
        "video": "Transcribe all speech and dialogue in this video. Note visual context when relevant.",
        "image": "Extract and transcribe any text visible in this image using OCR.",
    },
    "extract": {
        "audio": "Extract key information, topics, or specific content from this audio.",
        "video": "Extract key visual and audio information from this video.",
        "image": "Extract all text, objects, and important visual elements from this image.",
    },
    "caption": {
        "audio": "Generate descriptive captions for this audio content.",
        "video": "Generate detailed captions describing both visual and audio elements of this video.",
        "image": "Generate a comprehensive caption describing this image.",
    },
    "summarize": {
        "audio": "Provide a concise summary of the main points in this audio.",
        "video": "Summarize the key visual and audio content of this video.",
        "image": "Summarize the main elements and content of this image.",
    },
    "search": {
        "audio": "Make this audio content searchable by extracting keywords, topics, and semantic information.",
        "video": "Extract searchable content from both visual and audio elements of this video.",
        "image": "Extract searchable keywords and descriptions from this image.",
    },
}

GAIA_MEDIA_CONTEXT_TEMPLATE = (
    "Attached media file: {file_path} ({modality}).\n"
    "Use the multimodal_processor tool on this file_path before answering."
)

GAIA_DOCUMENT_CONTEXT_TEMPLATE = "Document context from {filename}:\n{content}"

GAIA_CONTEXT_QUESTION_TEMPLATE = "{context}\n\nQuestion: {question}"
