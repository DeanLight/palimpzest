import base64
import requests
import json
import shutil
import os
import logging
import sys


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def make_payload(base64_image):
    payload = {
      "model": "gpt-4-vision-preview",
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": '''
              You are a image analysis bot.  Analyze the supplied image and return an complete description of its contents including 
              a description of the general scene as well as a list of all of the people, animals, and objects you see and what they are doing.
              '''
            },
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
              }
            }
          ]
        }
      ],
      "max_tokens": 4000
    }
    return payload

def do_image_analysis(api_key, image_bytes):
    # Getting the base64 string
    base64_image = image_bytes

    payload = make_payload(base64_image)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    # Your JSON blob
    json_blob = response.json()
    #print(json_blob)
    # Accessing the content
    content_str = json_blob['choices'][0]['message']['content']

    # Converting the string back to a JSON object
    #content_json = json.loads(content_str)

    #print(content_json)
    return content_str
