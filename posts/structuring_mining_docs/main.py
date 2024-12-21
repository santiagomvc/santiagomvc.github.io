# import libraries
import anthropic
import base64
from dotenv import load_dotenv
import fitz  # PyMuPDF
import io
from PIL import Image
from openai import OpenAI


# Load environment variables
load_dotenv("../../.env")

# Initialize OpenAI client
client = OpenAI()

# First fetch the file
with open("data/AVISO WEB No 027-2024 PAR NOBSA.pdf", "rb") as f:
    pdf_data = base64.standard_b64encode(f.read()).decode("utf-8")

# Convert base64 back to bytes
pdf_bytes = base64.b64decode(pdf_data)

# Open PDF from memory
pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

# Convert each page to image
images = []
for page_num in range(pdf_document.page_count):
    page = pdf_document[page_num]
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    images.append(img)

pdf_document.close()

# Convert images to base64 strings
image_base64s = []
for img in images:
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    image_base64s.append(img_str)

# Prepare the messages with images
messages = [
    {
        "role": "system",
        "content": """You are an assistant that analyzes documents. 
        Please provide your analysis in the following JSON format:
        {
            "document_type": "string",
            "main_topic": "string",
            "key_points": ["string"],
            "dates_mentioned": ["string"],
            "entities_mentioned": ["string"]
        }"""
    },
    {
        "role": "user", 
        "content": [
            {
                "type": "text",
                "text": "Please analyze these document images and provide the information in the specified JSON format."
            }
        ]
    }
]

# Add each image to the content array
for img_str in image_base64s:
    messages[1]["content"].append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{img_str}"
        }
    })

# Send request to OpenAI
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    max_tokens=1000,
    response_format={"type": "json_object"}
)

# Store structured response
structured_analysis = response.choices[0].message.content


# # Finally send the API request
# client = anthropic.Anthropic()
# message = client.beta.messages.create(
#     model="claude-3-5-sonnet-20241022",
#     betas=["pdfs-2024-09-25"],
#     max_tokens=1024,
#     messages=[
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "document",
#                     "source": {
#                         "type": "base64",
#                         "media_type": "application/pdf",
#                         "data": pdf_data
#                     }
#                 },
#                 {
#                     "type": "text",
#                     "text": "Summarize the document in a few sentences."
#                 }
#             ]
#         }
#     ],
# )

# print(message.content)

