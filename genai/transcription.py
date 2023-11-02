# Classe que recebe um nome de arquivo no construtor e transcreve o seu conteúdo
import openai
import os
import speech_recognition as sr
from pydub import AudioSegment
from pydub.playback import play
from vosk import Model, KaldiRecognizer
import json
import logging
import io

lista_modos = ["openai", "google", "vosk"]


class Transcription:
    PRICE_PER_MINUTE_USD = 0.006

    def __init__(self, nome_arquivo : str, modo : str = "openai"):
        self.nome_arquivo = nome_arquivo
        self.last_transcription_cost = 0
        self.total_cost = 0
        if modo in lista_modos:
            self.modo = modo
        else:
            raise Exception("Modo inválido")
    
    def __converter_mp3_para_wav(self, caminho_arquivo : str):
        audio = AudioSegment.from_mp3(caminho_arquivo)
        caminho_wav = caminho_arquivo.replace(".mp3", ".wav")

        logging.info(f"Convertendo o arquivo '{caminho_arquivo}' para o formato wav em {caminho_wav}...")

        audio.export(caminho_wav, format="wav")

        logging.info("Arquivo convertido com sucesso.")

        return caminho_wav

    def __obter_transcricao_audio_openai(self, nome_arquivo : str) -> str:
        audio = AudioSegment.from_wav(nome_arquivo)

        response = ""

        # PyDub handles time in milliseconds
        chunk_size_in_minutes = 1 * 60 * 3000 #3 minutes chunk

        # chunkenize thw audio and transcribe until the end of the audio
        for i in range(0, len(audio), chunk_size_in_minutes):
            chunk = audio[i:i+chunk_size_in_minutes]
            # save the chunk in a buffer
            buffer = io.BytesIO()
            # you need to set the name with the extension
            buffer.name = f"chunk{i}.wav"
            chunk.export(buffer, format="wav")

            chunk_result = openai.Audio.transcribe("whisper-1", buffer)
            logging.info(chunk_result["text"])
            response += chunk_result["text"]

        self.last_transcription_cost = (len(audio) / (60*1000))*Transcription.PRICE_PER_MINUTE_USD
        self.total_cost += self.last_transcription_cost

        return response

    def __obter_transcricao_audio_google(self, nome_arquivo : str) -> str:
        recognizer = sr.Recognizer()
        with sr.AudioFile(nome_arquivo) as source:
            audio = recognizer.record(source)
            texto = recognizer.recognize_google(audio, language='pt-BR')
            return texto

    def __obter_transcricao_audio_vosk(self, nome_arquivo : str) -> str:
        model = Model("./vosk-model-pt-fb-v0.1.1-20220516_2113")

        recognizer = KaldiRecognizer(model, 16000)

        wf = open(nome_arquivo, "rb")
        
        texto = ""

        while True:
            data = wf.read(4000)
            if len(data) == 0:
                break
            if recognizer.AcceptWaveform(data):
                resultado = recognizer.Result()
                if resultado:
                    # le como json
                    resultado = json.loads(resultado)
                    if resultado["text"] and resultado["text"] != "<UNK>":
                        print(resultado["text"], end="")
                        texto += resultado["text"]

        resultado_final = recognizer.FinalResult()
        texto += resultado_final["text"]

        return texto

    def obter_transcricao_audio(self) -> str:
        # verifica se o arquivo é mp3 e, se for, primeiro converte para wav
        if self.nome_arquivo.endswith(".mp3"):
            nome_arquivo_wav = self.__converter_mp3_para_wav(self.nome_arquivo)
        elif self.nome_arquivo.endswith(".wav"):
            nome_arquivo_wav = self.nome_arquivo
        else:
            raise Exception("Arquivo inválido")

        if self.modo == "openai":
            return self.__obter_transcricao_audio_openai(nome_arquivo_wav)
        elif self.modo == "google":
            return self.__obter_transcricao_audio_google(nome_arquivo_wav)
        elif self.modo == "vosk":
            return self.__obter_transcricao_audio_vosk(nome_arquivo_wav)
        else:
            raise Exception(f"O modo '{self.modo}' não é válido.")
