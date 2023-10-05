import requests

response = requests.post(
    url='http://d2ae-35-230-66-89.ngrok.io/text-to-image',
    json={'prompt': 'High quality photo of a racing car on a track'})

from PIL import Image
import io

image = Image.open(io.BytesIO(response.content))
image.show()
