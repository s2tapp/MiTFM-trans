import os
import time
import requests
import json
import logging


import pika     #for rabiitMQ
import librosa  #For managing audio file
import torch    #For Pytorch

#Importing  Transformers
from transformers import WhisperForConditionalGeneration, WhisperProcessor  #openai whisper
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor  # wav2vec

# VALORES GLOBALES, OBTENIDOS DEL ENTORNO
khz16 = 16000 
log_file= os.environ.get('LOG_FILE', 's2t-processing.log')
log_format='%(asctime)s - %(levelname)s - S2T-CONVERTER-WORKER:%(message)s'
audio_folder= os.path.expanduser(os.environ.get('AUDIO_FOLDER', '/tmp/audio-uploads'))
base_dir = os.path.expanduser(os.environ.get('HF_HOME', '~/.cache/huggingface/')) + 'hub/'
#base_dir_whisper = base_dir + "models--openai--whisper-large-v3" 
#base_dir_wav2vec = base_dir + "models--facebook--wav2vec2-large-960h-lv60-self" 
downloads_url= os.environ.get('DOWNLOAD_URL',"http://backend:8080/api/files/")
rabbitmq_url = os.environ.get('RABBITMQ_URL', 'amqp://rabbitmq?connection_attempts=10&retry_delay=10')

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Configuración del registro
logging.basicConfig(level=logging.INFO, format=log_format, filename=log_file, filemode='a')
logger = logging.getLogger(__name__)

#
# AVISOS INICIALES
logger.info("Inicializando Servicio SpeechToText")
logger.info("URL de gestion de cola de peticiones [%s]", rabbitmq_url)
logger.info("URL de Descarga de ficheros de audio [%s]", downloads_url)
logger.info("Directorio de almacenamiento de descargas [%s]", audio_folder)
logger.info("Directorio de almacenamiento de modelos [%s]", base_dir)
logger.info("Dispositivo de cálculo [%s]", device)


tech1 = "Whisper"
model_whisper_id = "openai/whisper-large-v3"
try:
    logger.info(f"Descargando/Inicializando modelo {model_whisper_id}...")
    wshmodel = WhisperForConditionalGeneration.from_pretrained(model_whisper_id, torch_dtype=torch_dtype).to(device)
    logger.info(f"Inicializando Procesador WHISPER {model_whisper_id}...")
    wshprocessor = WhisperProcessor.from_pretrained(model_whisper_id)

    # if os.path.exists(base_dir_whisper):
    #     logger.info(f"Inicializando Modelo WHISPER {model_whisper_id}...")
    #     wshmodel = WhisperForConditionalGeneration.from_pretrained(model_whisper_id, torch_dtype=torch_dtype, local_files_only=True).to(device)
    #     logger.info(f"Inicializando Procesador WHISPER {model_whisper_id}...")
    #     wshprocessor = WhisperProcessor.from_pretrained(model_whisper_id)
    # else:
    #     logger.info(f"Descargando modelo {model_whisper_id}...")
    #     wshmodel = WhisperForConditionalGeneration.from_pretrained(model_whisper_id, force_download=True).to(device)
    #     logger.info(f"Inicializando Procesador WHISPER {model_whisper_id}...")
    #     wshprocessor = WhisperProcessor.from_pretrained(model_whisper_id)
except Exception as e:
        logger.error(f"Error al Inicializar Modelo WHISPER: [[{str(e)}]]")


tech2 = "Wav2vec"
model_wav2vec_id = "facebook/wav2vec2-large-960h-lv60-self"
try:
    logger.info(f"Descargando modelo {model_wav2vec_id}...")
    w2vmodel = Wav2Vec2ForCTC.from_pretrained(model_wav2vec_id).to(device)
    logger.info(f"Inicializando Procesador WAV2VEC {model_wav2vec_id}...")
    w2vprocessor = Wav2Vec2Processor.from_pretrained(model_wav2vec_id)

    # if os.path.exists(base_dir_wav2vec):
    #     logger.info(f"Inicializando Modelo WAV2VEC {model_wav2vec_id}...")
    #     w2vmodel = Wav2Vec2ForCTC.from_pretrained(model_wav2vec_id, local_files_only=True).to(device)
    #     logger.info(f"Inicializando Procesador WAV2VEC {model_wav2vec_id}...")
    #     w2vprocessor = Wav2Vec2Processor.from_pretrained(model_wav2vec_id)
    # else:
    #     logger.info(f"Descargando modelo {model_wav2vec_id}...")
    #     w2vmodel = Wav2Vec2ForCTC.from_pretrained(model_wav2vec_id, force_download=True).to(device)
    #     logger.info(f"Inicializando Procesador WAV2VEC {model_wav2vec_id}...")
    #     w2vprocessor = Wav2Vec2Processor.from_pretrained(model_wav2vec_id)

except Exception as e:
        logger.error(f"Error al Inicializar Modelo WAV2VEC: [[{str(e)}]]")



def process_whisper(audio):
    try:
        ifht = wshprocessor(audio, return_tensors="pt", sampling_rate=khz16)
        input_features = ifht.to(torch.half).to(device)
        
        outputs = wshmodel.generate(**input_features, output_scores=True, return_dict_in_generate=True)
#       outputs = model.generate(**ifht, output_scores=True, return_dict_in_generate=True)

        predicted_ids = outputs.sequences
        transcription = wshprocessor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

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

    except Exception as e:
        logger.error(f"Error al intentar transcripción con WHISPER [[{str(e)}]]")
        return "¡¡¡ Error al intentar transcripción con WHISPER !!!.", None

def process_wav2vec(audio):
    try:
        input_values = w2vprocessor(audio, return_tensors = "pt", sampling_rate=khz16)["input_values"].to(device)

        # guardando logits (non-normalized prediction values)
        logits = w2vmodel(input_values).logits
        predicted_ids = torch.argmax(logits, dim = -1)

        transcription = w2vprocessor.batch_decode(predicted_ids)[0]

        # Cáculo de probabilidades
        probabilities = torch.softmax(logits, dim=-1)

        # Extraer la probabilidad máxima para cada token generado
        max_probabilities, _ = torch.max(probabilities, dim=-1)

        # Calcular la confianza media
        confianza_media = round( max_probabilities.mean().item(), 3)  
        return transcription, confianza_media

    except Exception as e:
        logger.error(f"Error al intentar transcripción con WAV2VEC [[{str(e)}]]")
        return "¡¡¡ Error al intentar transcripción con WAV2VEC !!!", None
        

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
    file_path = os.path.join(audio_folder, file_name)
    url = downloads_url + file_name

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

# Preparación del bucle central de procesamiento
os.makedirs(audio_folder, exist_ok=True)

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