
import os
import time
import requests
import json
#for rabiitMQ
import pika
#For managing audio file
import librosa
#For Pytorch
import torch
#Importing  openai
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
#Importing Wav2Vec
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

tec1 = "whisper"
tec2 = "wav2vec"
donwloads_folder= "/tmp/python-data"
donwloads_url= os.environ.get('DONWLOAD_URL',"http://localhost:8080/api/files/")
rabbitmq_url = os.environ.get('RABBITMQ_URL', 'amqp://localhost?connection_attempts=10&retry_delay=10')
print(donwloads_url)
def process_whisper(filepath):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )


    # Loading the audio file
    audio, rate = librosa.load(filepath)

    result = pipe(audio)
    return result["text"]

def process_wav2vec(filepath):
    # Loading the audio file
    audio, rate = librosa.load(filepath, sr = 16000)

    # Importing Wav2Vec pretrained model
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    # Taking an input value
    input_values = processor(audio, return_tensors = "pt").input_values

    # Storing logits (non-normalized prediction values)
    logits = model(input_values).logits

    # Storing predicted ids
    prediction = torch.argmax(logits, dim = -1)

    # Passing the prediction to the tokenzer decode to get the transcription
    transcription = processor.decode(prediction[0])

    return transcription

def process_audio(file_Path, technology):
    if tec1 == technology:
        return process_whisper(file_Path)
    elif tec2 == technology:
        return process_wav2vec(file_Path)
    else:
        return None

def consume_and_respond(ch, method, properties, body):
    try:
        startTime = time.time()

        # Deserializar el mensaje como un diccionario
        message = eval(body.decode("utf-8"))

        # Extraer nombreAudio y el texto del diccionario
        technology = message.get("technology")
        audioName = message.get("audioName")

        filePath = download_file(audioName)

        if filePath:
            transcription = process_audio(filePath, technology)
            response = {
                "respuesta": transcription if transcription is not None else "No se ha podido transcribir."
            }
        else:
            response = {
                "respuesta": "No se ha podido descargar el archivo."
            }

        # Publicar la transcripción como respuesta
        channel.basic_publish(
            exchange='',
            routing_key=properties.reply_to,
            properties=pika.BasicProperties(correlation_id=properties.correlation_id),
            body=json.dumps(response)
        )
         # Confirmar que se ha procesado el mensaje
        ch.basic_ack(delivery_tag=method.delivery_tag)

        endTime = time.time()
        elapsedTime = endTime - startTime
        print("Tiempo necesario:", elapsedTime)        
    except Exception as e:
        print("Error procesando audio:", str(e))
      
def download_file(file_name):
    file_path = os.path.join(donwloads_folder, file_name)
    url = donwloads_url+file_name

    response = requests.get(url)
    
    if response.status_code == 200:
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print(f"Archivo descargado como '{file_name}'")
        return file_path
    else:
        print("Error al descargar el archivo:", response.status_code)
        return None

os.makedirs(donwloads_folder, exist_ok=True)

connection = pika.BlockingConnection(pika.URLParameters(rabbitmq_url))
channel = connection.channel()

channel.basic_consume(queue='audio_queue', on_message_callback=consume_and_respond)

try:
    print('Esperando mensajes...')
    channel.start_consuming()
except KeyboardInterrupt:
    print("Proceso interrumpido")
finally:
    connection.close()