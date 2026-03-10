def is_jpeg(data):
    """Return whether the provided bytes look like a JPEG image."""
    jpeg_signature = b'\xFF\xD8\xFF'
    return data.startswith(jpeg_signature)

def is_png(data):
    """Return whether the provided bytes look like a PNG image."""
    png_signature = b'\x89\x50\x4E\x47\x0D\x0A\x1A\x0A'
    return data.startswith(png_signature)


def get_image_type_from_bytes(data):
    """Infer the image MIME subtype from raw bytes."""
    if is_jpeg(data):
        return "jpeg"
    elif is_png(data):
        return "png"
    else:
        raise ValueError("Image type not supported, only jpeg and png supported.")