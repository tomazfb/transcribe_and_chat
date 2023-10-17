# Classe que recebe um nome de arquivo no construtor e transcreve o seu conteúdo
import openai
import os
import speech_recognition as sr
from pydub import AudioSegment
from pydub.playback import play
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
import json
import logging

lista_modos = ["openai", "google", "vosk"]


class Transcricao:
    def __init__(self, nome_arquivo : str, modo : str = "openai"):
        self.nome_arquivo = nome_arquivo
        if modo in lista_modos:
            self.modo = modo
        else:
            raise Exception("Modo inválido")
    
    @staticmethod
    def __converter_mp3_para_wav(caminho_arquivo : str):
        audio = AudioSegment.from_mp3(caminho_arquivo)
        caminho_wav = caminho_arquivo.replace(".mp3", ".wav")

        logging.info(f"Convertendo o arquivo '{caminho_arquivo}' para o formato wav em {caminho_wav}...")

        audio.export(caminho_wav, format="wav")

        logging.info("Arquivo convertido com sucesso.")

        return caminho_wav

    @staticmethod
    def __obter_transcricao_audio_openai(nome_arquivo : str) -> str:
        audio = AudioSegment.from_wav(nome_arquivo)

        response = ""
        
        # PyDub handles time in milliseconds
        chunk_size_in_minutes = 2 * 60 * 1000

        # chunkenize thw audio and transcribe until the end of the audio
        for i in range(0, len(audio), chunk_size_in_minutes):
            chunk = audio[i:i+chunk_size_in_minutes]
            # save the chunk as a wav file
            chunk_name = f"chunk{i}.wav"
            chunk.export(chunk_name, format="wav")
            with open(chunk_name, "rb") as audio_file:
                # transcribe the audio file in chunks and print result while transcribing
                chunk_result = openai.Audio.transcribe("whisper-1", audio_file)
                logging.info(chunk_result["text"])
                response += chunk_result["text"]

            # delete the chunk file
            os.remove(chunk_name)

        return response

    @staticmethod
    def __obter_transcricao_audio_google(nome_arquivo : str) -> str:
        recognizer = sr.Recognizer()
        with sr.AudioFile(nome_arquivo) as source:
            audio = recognizer.record(source)
            texto = recognizer.recognize_google(audio, language='pt-BR')
            return texto

    @staticmethod
    def __obter_transcricao_audio_vosk(nome_arquivo : str) -> str:
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
            nome_arquivo_wav = Transcricao.__converter_mp3_para_wav(self.nome_arquivo)
        elif self.nome_arquivo.endswith(".wav"):
            nome_arquivo_wav = self.nome_arquivo
        else:
            raise Exception("Arquivo inválido")

        if self.modo == "openai":
            return Transcricao.__obter_transcricao_audio_openai(nome_arquivo_wav)
        elif self.modo == "google":
            return Transcricao.__obter_transcricao_audio_google(nome_arquivo_wav)
        elif self.modo == "vosk":
            return Transcricao.__obter_transcricao_audio_vosk(nome_arquivo_wav)
        else:
            raise Exception(f"O modo '{self.modo}' não é válido.")
