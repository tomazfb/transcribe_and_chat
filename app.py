from transcricao import Transcricao
from chat_with_embeddings import ChatWithEmbeddings
import os
import sys
from langchain.callbacks import get_openai_callback

def main(args):
    # o argumento deve ser um arquivo.mp3 ou arquivo.wav
    if len(args) != 2:
        # imprime a mensagem de uso permitindo mp3 ou wav
        print("Uso: python app.py <arquivo_entrada>")
        return

    nome_arquivo = args[1]

    if not os.path.isfile(nome_arquivo):
        print(f"O arquivo '{nome_arquivo}' não foi encontrado.")
        return

    extension_lowercase = nome_arquivo.split(".")[-1].lower()

    if extension_lowercase.endswith("mp3") or extension_lowercase.endswith("wav"):
        t = Transcricao(nome_arquivo)
        transcricao = t.obter_transcricao_audio()
        # escreve o resultado no caminho
        # do arquivo com o nome do arquivo sem extensão
        # e com extensão .txt
        # exemplo: arquivo.mp3 -> arquivo.txt
        # exemplo: arquivo.wav -> arquivo.txt
        arquivo_transcricao = nome_arquivo.split(".")[0] + "_transcricao.txt"
        with open(arquivo_transcricao, "w") as f:
            f.write(transcricao)

        print(f"A transcrição foi salva em: {arquivo_transcricao}")
    elif extension_lowercase.endswith("txt") or extension_lowercase.endswith("xlsx") or extension_lowercase.endswith("csv"):
        with get_openai_callback() as cb:
            if extension_lowercase.endswith("txt"):
                loader = ChatWithEmbeddings.create_text_loader(nome_arquivo)
            elif extension_lowercase.endswith("csv"):
                loader = ChatWithEmbeddings.create_csv_loader(nome_arquivo)
            else: # extension_lowercase.endswith("xlsx"):
                loader = ChatWithEmbeddings.create_unstructured_excel_loader(nome_arquivo)
            
            c = ChatWithEmbeddings(loader)
            
            #pergunta ao usuário a frase
            frase = input(f"Digite o prompt para interagir com {nome_arquivo}: \n")
            print("")
            while frase != "q":
                resposta = c.chat(frase)
                if resposta and resposta["result"]:
                    print(resposta["result"])
                    print("--- Total Token and Cost Tracking ---")
                    print(cb)
                    print("---")
                frase = input(f"Digite o prompt ou 'q' para sair: \n")

    else:
        raise Exception("O arquivo deve ser um arquivo .mp3, .wav, .txt ou .xlsx")

if __name__ == "__main__":
    main(sys.argv)





