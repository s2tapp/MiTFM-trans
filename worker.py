import os
import time
import requests
import json
import logging
#for rabiitMQ
import pika
#For managing audio file
import librosa
#For Pytorch
import torch
#Importing  openai
from transformers import WhisperForConditionalGeneration, WhisperProcessor
#Importing Wav2Vec
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Configuración del registro
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='audio_processing.log', filemode='a')
logger = logging.getLogger(__name__)

tech1 = "Whisper"
tech2 = "Wav2vec"

khz16 = 16000 
donwloads_folder= "/tmp/python-data"
donwloads_url= os.environ.get('DONWLOAD_URL',"http://localhost:8080/api/files/")
rabbitmq_url = os.environ.get('RABBITMQ_URL', 'amqp://localhost?connection_attempts=10&retry_delay=10')


model_whisper_id = "openai/whisper-large-v3"
model_wav2vec_id = "facebook/wav2vec2-large-960h-lv60-self"
base_dir = os.path.expanduser("~/.cache/huggingface/hub/")
base_dir_whisper = base_dir + "models--openai--whisper-large-v3" 
base_dir_wav2vec = base_dir + "models--facebook--wav2vec2-large-960h-lv60-self" 

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def process_whisper(audio):
    try:
        if os.path.exists(base_dir_whisper):
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            model = WhisperForConditionalGeneration.from_pretrained(model_whisper_id, torch_dtype=torch_dtype, local_files_only=True).to(device)
            processor = WhisperProcessor.from_pretrained(model_whisper_id)

            input_features = processor(audio, return_tensors="pt", sampling_rate=khz16)
            
            outputs = model.generate(**input_features, output_scores=True, return_dict_in_generate=True)

            predicted_ids = outputs.sequences

            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

            #Cáculo de confianza (non-normalized score values)
            scores = outputs.scores

            confidences = []
            for score in scores:
                probabilities = torch.softmax(score, dim=-1)
                max_probabilities, _ = torch.max(probabilities, dim=-1)
                confidences.append(max_probabilities.mean().item())

            # Confianza media a través de todos los pasos de generación
            confianza_media = round(sum(confidences) / len(confidences), 3)
            # Devolver la transcripción y la confianza
            return transcription, confianza_media
        else:
            logger.info(f"Descargando modelo {model_whisper_id}...")
            WhisperForConditionalGeneration.from_pretrained(model_whisper_id, force_download=True)
            return "No se ha podido realizar la transcripción. Inténtelo más tarde.", None
    except Exception as e:
        logger.error(f"Error en el procesamiento con Whisper: {str(e)}")
        return "Error durante la transcripción.", None

def process_wav2vec(audio):
    try:
        if os.path.exists(base_dir_wav2vec):
            model = Wav2Vec2ForCTC.from_pretrained(model_wav2vec_id, local_files_only=True).to(device)
            processor = Wav2Vec2Processor.from_pretrained(model_wav2vec_id)

            input_values = processor(audio, return_tensors = "pt", sampling_rate=khz16)["input_values"].to(device)

            # guardando logits (non-normalized prediction values)
            logits = model(input_values).logits

            predicted_ids = torch.argmax(logits, dim = -1)

            transcription = processor.batch_decode(predicted_ids)[0]

            # Cáculo de probabilidades
            probabilities = torch.softmax(logits, dim=-1)

            # Extraer la probabilidad máxima para cada token generado
            max_probabilities, _ = torch.max(probabilities, dim=-1)

            # Calcular la confianza media
            confianza_media = round( max_probabilities.mean().item(), 3)  

            return transcription, confianza_media
        else:
            logger.info(f"Descargando modelo {model_wav2vec_id}...")
            Wav2Vec2ForCTC.from_pretrained(model_wav2vec_id, force_download=True)
            return "No se ha podido realizar la transcripción. Inténtelo más tarde.", None
    except Exception as e:
        logger.error(f"Error en el procesamiento con wav2vec: {str(e)}")
        return "Error durante la transcripción.", None
        

def load_audio(file_Path):
    # Cargamos el fichero
    speech, sr = librosa.load(file_Path)
    if sr != khz16:
        speech = librosa.resample(speech, orig_sr=sr, target_sr=khz16)
    return speech

def process_audio(file_path, technology):
    # Cargando el fichero de audio
    audio = load_audio(file_path)

    if tech1 == technology:
        return process_whisper(audio)
    elif tech2 == technology:
        return process_wav2vec(audio)
    else:
        return None

def consume_and_respond(ch, method, properties, body):
    try:
        # Deserializar el mensaje como un diccionario
        message = json.loads(body.decode("utf-8"))
        logger.info(f"Mensaje recibido: {message}")

        # Extraer audioName y la tecnología del diccionario
        technology = message.get("technology")
        audioName = message.get("audioName")

        logger.info(f"Procesando audio '{audioName}' usando tecnología '{technology}'")

        filePath = download_file(audioName)
        
        if filePath:
            startTime = time.time()
            transcription, confianza_media = process_audio(filePath, technology)

            endTime = time.time()
            elapsedTime = round(endTime - startTime, 3)
            response = {
                "respuesta": transcription if transcription is not None else "No se ha podido transcribir.",
                "confianza": confianza_media,
                "tiempoProceso": elapsedTime
            }
            logger.info(f"Transcripción: {transcription}, Confianza: {confianza_media}, Tiempo de proceso: {elapsedTime} segundos")
        else:
            response = {
                "respuesta": "No se ha podido descargar el archivo."
            }            
            logger.error("No se ha podido descargar el archivo.")

    except Exception as e:
        logger.error(f"Error procesando audio: {str(e)}")
        response = {
            "respuesta": "Error procesando audio"
        }
    finally:
        # Publicar la transcripción o el error como respuesta
        ch.basic_publish(
            exchange='',
            routing_key=properties.reply_to,
            properties=pika.BasicProperties(correlation_id=properties.correlation_id),
            body=json.dumps(response)
        )
        # Confirmar que se ha procesado el mensaje
        ch.basic_ack(delivery_tag=method.delivery_tag)

def download_file(file_name):
    file_path = os.path.join(donwloads_folder, file_name)
    url = donwloads_url + file_name

    try:
        response = requests.get(url)
        response.raise_for_status()  # Lanza una excepción para códigos de error HTTP
       
        with open(file_path, 'wb') as f:
            f.write(response.content)
            logger.info(f"Archivo descargado como '{file_name}'")
            return file_path
    except requests.exceptions.RequestException as e:
        logger.error(f"Error al descargar el archivo: {response.status_code}")
        return None

os.makedirs(donwloads_folder, exist_ok=True)

connection = pika.BlockingConnection(pika.URLParameters(rabbitmq_url))
channel = connection.channel()

# asegura la creación de la cola
channel.queue_declare(queue='audio_queue', durable=True)

channel.basic_consume(queue='audio_queue', on_message_callback=consume_and_respond)

try:
    logger.info('Esperando mensajes...')
    channel.start_consuming()
except KeyboardInterrupt:
    logger.info("Proceso interrumpido")
finally:
    connection.close()
    logger.info("Conexión cerrada")