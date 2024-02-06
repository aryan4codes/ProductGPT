from openai import OpenAI
import os

# Ensure you have set your OPENAI_API_KEY environment variable
api_key = os.getenv('OPENAI_API_KEY')
if api_key is None:
    raise ValueError("API key not found. Make sure you've set the OPENAI_API_KEY environment variable.")

client = OpenAI(api_key="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")
response = client.chat.completions.create(
  model="gpt-4-vision-preview",
  messages=[
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Can you find the object in first image present in the second image? IF yes, can you tell the locations of the object in the second image?",
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "https://storage.sg.content-cdn.io/cdn-cgi/image/width=1000,height=1500,quality=75,format=auto/in-resources/8845e144-8902-4204-b80f-9dc7dc2f4bcb/Images/ProductImages/Source/3014259%20Vaseline%20Intensive%20Care%20Body%20Lotion%20300ml.jpg",},
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
          },
        },
      ],
    }
  ],
  max_tokens=300,
)
print(response.choices[0])