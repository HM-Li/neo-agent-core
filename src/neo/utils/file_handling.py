import base64
import io
import re
from urllib.parse import unquote
from typing import Tuple

import requests
from pydub import AudioSegment


def fetch_url_as_base64_str(url: str) -> dict:
    """
    Download the content of a URL and return it as bytes.

    Args:
        url (str): The URL to download.

    Returns:
        base64 encoded and utf-8 decoded str, mime_type
    """
    response = requests.get(url, timeout=10)
    response.raise_for_status()  # Raise an error for bad responses

    # Return raw bytes instead of base64 encoding
    data = response.content
    mime = response.headers.get("Content-Type")

    # get file name
    file_name = None
    cd = response.headers.get("Content-Disposition")
    if cd:
        file_name = re.findall('filename="?([^"]+)"?', cd)
        if file_name:
            file_name = file_name[0]
    # Fallback to last segment of URL
    if not file_name:
        file_name = unquote(url.split("/")[-1]) or "downloaded_file"

    if mime is None:
        raise ValueError("Could not determine MIME type from the response headers.")

    # Content-Type can include encoding (audio/wav; charset=UTF-8)
    mime = mime.split(";")[0].strip()

    data = binary_to_base64_str(data)

    return {
        "data": data,
        "mime_type": mime,
        "file_name": file_name,
    }


def load_file_as_base64_str(file_path: str) -> str:
    """
    Load a file and return its content as a base64 encoded string.

    Args:
        file_path (str): The path to the file.

    Returns:
        str: The base64 encoded string of the file content.
    """
    with open(file_path, "rb") as f:
        data = f.read()
    return binary_to_base64_str(data)


from pypdf import PdfReader


def extract_text_from_pdf(data: bytes | str) -> str:
    """
    Extract text from a PDF file.

    Args:
        data (bytes): The PDF file content in bytes.

    Returns:
        str: The extracted text.
    """
    if isinstance(data, str):
        # If the data is a base64 encoded string, decode it
        data = base64.b64decode(data, validate=True)

    reader = PdfReader(io.BytesIO(data))

    # concatenate all the pages
    texts = "\n".join([page.extract_text() for page in reader.pages])
    return texts


def binary_to_base64_str(data: bytes) -> str:
    """
    Convert binary data to a base64 encoded string.

    Args:
        data (bytes): The binary data.

    Returns:
        str: The base64 encoded string.
    """
    return base64.b64encode(data).decode("utf-8")


def base64_str_to_binary(data: str) -> bytes:
    """
    Convert a base64 encoded string to binary data.

    Args:
        data (str): The base64 encoded string.

    Returns:
        bytes: The binary data.
    """
    return base64.b64decode(data, validate=True)


def reformat_audio_bytes(
    audio_data: bytes | str, mime_type: str, target_format="wav"
) -> dict:
    """
    Convert audio bytes to WAV format.

    Args:
        audio_bytes (bytes | str): The audio data in raw bytes or base64 encoded string.
        mime_type (str): The MIME type of the audio data.

    Returns:
        Tuple[bytes, str]: The converted audio data in bytes and its MIME type.
    """
    # Load the audio using pydub
    format_name = mime_type.split("/")[-1]

    if isinstance(audio_data, str):
        audio_data = base64_str_to_binary(audio_data)
    elif not isinstance(audio_data, (bytes, bytearray)):
        raise ValueError(
            "audio_bytes must be a bytes object or a base64 encoded string"
        )

    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_data), format=format_name)
    except Exception as e:
        raise ValueError(f"Unsupported or invalid audio format: {format_name}") from e

    # Convert to WAV
    wav_io = io.BytesIO()
    audio.export(wav_io, format=target_format)
    wav_io.seek(0)

    return {"data": wav_io.read(), "mime_type": f"audio/{target_format}"}
