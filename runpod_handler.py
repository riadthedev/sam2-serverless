import runpod
from sam2_processor import embed_image

def handler(event):
    if 'input' not in event:
        return {"error": "No input provided"}

    action = event['input'].get('action', 'embed_image')

    if action == 'embed_image':
        # Return only the image embedding for client-side decoding
        return embed_image(event)
    else:
        return {"error": f"Unknown action: {action}"}
    

runpod.serverless.start({"handler": handler})
