# %%
from io import BytesIO

from openai import AsyncOpenAI
from PIL import Image

from neo.types.contents import ImageContent
from neo.types.errors import ModelServiceError
from neo.utils import str_starts_withs
from neo.utils.file_handling import base64_str_to_binary, fetch_url_as_base64_str


def load_img(url: str = None, file_path: str = None, open_image: bool = False) -> bytes:
    """
    Load image from url or file path
    """
    if not url and not file_path:
        raise ValueError("Either url or file_path must be provided")

    if url:
        fetched = fetch_url_as_base64_str(url)
        bytes_data = fetched["data"]
    else:
        with open(file_path, "rb") as f:
            bytes_data = f.read()

    if open_image:
        img = Image.open(BytesIO(bytes_data))
        return img
    return f


def is_png(f):
    """
    Check if image is png
    """
    img = Image.open(f)
    if img.format == "PNG":
        return True
    return False


# %%
def convert2png(f: bytes, open_image=False):
    """
    Convert image to png
    """
    img = Image.open(BytesIO(f))
    f = BytesIO()
    img.save(f, format="PNG")
    if open_image:
        return img
    return f


import io
import math

from PIL import Image


def reduce_size(f: bytes, target=2e6):
    """Save the image as JPEG with the given name at best quality that makes less than "target" bytes"""
    # Min and Max quality
    Qmin, Qmax = 5, 95
    # Highest acceptable quality found
    Qacc = -1

    im = Image.open(BytesIO(f))
    fmt = im.format
    while Qmin <= Qmax:
        m = math.floor((Qmin + Qmax) / 2)

        # Encode into memory and get size
        buffer = io.BytesIO()
        if fmt == "JPEG":
            im.save(buffer, format=fmt, quality=m)
        else:
            w, h = im.size
            w = int(w * m / 100)
            h = int(h * m / 100)
            im.resize((w, h)).save(buffer, format=fmt)
        s = buffer.getbuffer().nbytes
        if s <= target:
            Qacc = m
            Qmin = m + 1
        elif s > target:
            Qmax = m - 1

    # Write to disk at the defined quality
    if Qacc > -1:
        f = BytesIO()
        if fmt == "JPEG":
            im.save(f, format=fmt, quality=Qacc)
        else:
            w, h = im.size
            w = int(w * Qacc / 100)
            h = int(h * Qacc / 100)
            im.resize((w, h)).save(f, format=fmt)
    return f


async def agenerate_image(
    prompt: str,
    custom_api_key: str = None,
    model: str = "gpt-image-1",
    size: str = "1024x1024",
    quality: str = "auto",
    n: int = 1,
):
    """
    Generate an image using OpenAI's DALL-E model.
    Parameters
    ----------
    prompt : str
        The prompt to generate the image from.
    custom_api_key : str, optional
        The custom API key for OpenAI, by default None
    model : str, optional
        The model to use for image generation, by default "gpt-image-1"
    size : str, optional
        The size of the generated image, by default "1024x1024"
    quality : str, optional
        The quality of the generated image, by default "auto"
    n : int, optional
        The number of images to generate, by default 1
    Returns
    -------
    str
        The URL of the generated image.
    """
    client = AsyncOpenAI(api_key=custom_api_key)

    try:
        response = await client.images.generate(
            model=model,
            prompt=prompt,
            size=size,
            quality=quality,
            n=n,
            # response_format="b64_json",  # new model only support b64, older model by default url
            output_format="png",
        )
    except Exception as e:
        raise ModelServiceError(e)

    if response.data[0].url:
        c = ImageContent(data=response.data[0].url)
    else:
        data = base64_str_to_binary(response.data[0].b64_json)
        mime = "image/png"
        c = ImageContent(data=data, mime_type=mime, file_name="image.png")
    return c


async def acreate_variation(
    image: str | bytes,
    custom_api_key: str = None,
    model: str = "dall-e-2",
    n: int = 1,
    size: str = "1024x1024",
):
    """
    Create a variation of an image using OpenAI's DALL-E model.
    Parameters
    ----------
    image : str | bytes
        The image to create a variation of. Can be a URL or a file path.
    custom_api_key : str, optional
        The custom API key for OpenAI, by default None
    model : str, optional
        The model to use for image generation, by default "dall-e-2"
    n : int, optional
        The number of variations to create, by default 1
    size : str, optional
        The size of the generated image, by default "1024x1024"
    Returns
    -------
    str
        The URL of the generated image.
    """
    if isinstance(image, str):
        url = file_path = None
        if str_starts_withs(image, ["http", "www", "https"]):
            url = image
        else:
            file_path = image
        image = load_img(url=url, file_path=file_path, open_image=False)

    if not is_png(image):
        image = convert2png(image)

    image = reduce_size(image)

    client = AsyncOpenAI(api_key=custom_api_key)

    try:
        response = await client.images.create_variation(
            model=model, image=image, n=n, size=size
        )
    except Exception as e:
        raise ModelServiceError(e)

    return response.data[0].url


# %%
# %%
