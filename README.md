# TuVozATexto (Procesamiento de Audio)

Este repositorio contiene un script en Python para procesar archivos de audio utilizando dos tecnologías de vanguardia: Whisper y Wav2vec. El script descarga archivos de audio de una URL especificada, los procesa y devuelve la transcripción del audio junto con la confianza en la transcripción y el tiempo de procesamiento.

**Nota:** Antes de utilizar este script, asegúrate de ejecutar primero el proyecto [MiTFM-back](https://github.com/uva2023/MiTFM-back), que proporciona la API necesaria para descargar los archivos de audio y comunicarse con el script de procesamiento.

## Requisitos

- Python 3.x
- Paquetes Python: `requests`, `librosa`, `torch`, `transformers`, `pika`, `ffmpeg`, `accelerate`

## Instalación

1. Clona este repositorio:

   ```bash
   git clone hhttps://github.com/uva2023/MiTFM-back.git
   ```

2. Instala las dependencias:

   ```bash
    pip install torch
    pip install pika
    pip install requests
    pip install ffmpeg
    pip install librosa
    pip install transformers
    pip install accelerate
   ```

## Uso

1. Configura las variables de entorno necesarias en tu sistema o define valores predeterminados en el script:

   - `DONWLOAD_URL`: URL base para descargar archivos de audio.
   - `RABBITMQ_URL`: URL de RabbitMQ para la comunicación con la cola de mensajes.

2. Ejecuta el script `worker.py`:

   ```bash
   python3 worker.py
   ```

El script estará a la espera de mensajes en la cola 'audio_queue' de RabbitMQ. Cuando se reciba un mensaje, descargará el archivo de audio especificado, lo procesará utilizando la tecnología especificada (Whisper o Wav2vec), y responderá con la transcripción, la confianza en la transcripción y el tiempo de procesamiento.

## Contribución

Siéntete libre de abrir un issue o enviar un pull request con cualquier mejora o corrección.

## Licencia

Este proyecto está bajo la licencia Haz_con_ello_lo_que_quieras_pero_no_me_marees License.
