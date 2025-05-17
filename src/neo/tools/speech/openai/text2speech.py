from openai import AsyncOpenAI
from neo.types.errors import ModelServiceError
from neo.types.contents import AudioContent


async def atext2speech(
    text: str,
    custom_api_key: str = None,
    model: str = "gpt-4o-mini-tts",
    voice: str = "alloy",
    output_file: str = None,
    instructions: str = "",
) -> AudioContent | None:
    """
    Convert text to speech using OpenAI's TTS API.

    Parameters
    ----------
    text : str
        The text to convert to speech.
    custom_api_key : str, optional
        Custom OpenAI API key. If None, uses default API key configuration.
    model : str, default="gpt-4o-mini-tts"
        The TTS model to use.
    voice : str, default="alloy"
        Voice style to use for speech generation. Options include:
        'alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'.
    output_file : str, optional
        If provided, saves audio directly to this file path instead of returning content.
    instructions : str, default=""
        Additional instructions to control the style of the generated speech.

    Returns
    -------
    AudioContent or None
        An AudioContent object containing the generated audio if output_file is None,
        otherwise returns None after saving to specified file.

    Raises
    ------
    ModelServiceError
        If there's an issue with the OpenAI API request.

    Notes
    -----
    This function provides two modes of operation:
    1. Return AudioContent directly for in-memory processing
    2. Stream results directly to a file when output_file is specified

    """

    client = AsyncOpenAI(api_key=custom_api_key)

    try:
        if output_file is None:
            data = await client.audio.speech.create(
                model=model,
                voice=voice,
                input=text,
                instructions=instructions,
                response_format="mp3",
            )
            mime = "audio/mp3"
            c = AudioContent(data=data, mime_type=mime)
            return c

        async with client.audio.speech.with_streaming_response.create(
            model=model,
            voice=voice,
            input=text,
            instructions=instructions,
        ) as response:
            response.stream_to_file(output_file)

    except Exception as e:
        raise ModelServiceError(e)
