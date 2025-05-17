# %%
from openai import OpenAI, AsyncOpenAI
from neo.types.errors import ModelServiceError
from neo.types import contents as C
from io import BytesIO
from neo.utils.file_handling import base64_str_to_binary, fetch_url_as_base64_str


async def aspeech2text(
    audio: C.AudioContent | C.AudioTextContent,
    custom_api_key: str = None,
    model: str = "gpt-4o-transcribe",
    language: str = None,
    prompt: str = None,
) -> C.TextContent:
    """
    Transcribe audio to text using OpenAI's Whisper model.

    Parameters
    ----------
    file : bytes
        The audio binary to transcribe.
    custom_api_key : str, optional
        The custom API key for OpenAI, by default None
    model : str, optional
        The model to use for transcription, by default "gpt-4o-transcribe"
    prompt : str, optional
        The prompt to use for transcription, by default None

    Returns
    -------
    TextContent
        The transcribed text content.

    Raises
    ------
    ModelServiceError
        If there is an error during the transcription process.

    """
    if isinstance(audio.data, str):
        fetched = fetch_url_as_base64_str(audio.data)
        binary = base64_str_to_binary(fetched["data"])
        audio.mime_type = fetched["mime_type"]
        audio.file_name = fetched["file_name"]
    else:
        binary = audio.data

    try:
        client = AsyncOpenAI(api_key=custom_api_key)

        audio_file = BytesIO(binary)
        audio_file.name = audio.file_name
        audio_file.seek(0)
        transcript = await client.audio.transcriptions.create(
            model=model,
            file=audio_file,
            prompt=prompt,
            response_format="json",
            language=language,
            temperature=0,
        )
        return C.TextContent(data=transcript.text)
    except Exception as e:
        raise ModelServiceError(e)


# %%
