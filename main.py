import texto_extracao as te
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
import re
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Carregar o tokenizer e o modelo treinado
path_modelo = "./t5_small_finetuned"
modelo = TFAutoModelForSeq2SeqLM.from_pretrained(path_modelo)
tokenizer = AutoTokenizer.from_pretrained(path_modelo)


def dividir_texto_em_sentencas(texto, num_sentencas=4):
    """
    Função que divide o texto em trechos contendo até num_sentences sentenças
    (baseado em pontos finais).
    """
    # Expressão regular para identificar sentenças (terminadas em ponto final)
    final_de_frase = re.compile(r"([.!?])")

    sentencas = []
    pedaco_atual = ""
    count_sentencas = 0

    # Iterar sobre cada caractere no texto
    for i, char in enumerate(texto):
        pedaco_atual += char
        if final_de_frase.match(char):  # Se encontrarmos um ponto final
            count_sentencas += 1

        # Quando encontrarmos duas sentenças completas (dois pontos finais)
        if count_sentencas == num_sentencas:
            sentencas.append(pedaco_atual.strip())  # Adicionar o trecho
            pedaco_atual = ""  # Reiniciar o trecho
            count_sentencas = 0 # Reiniciar contador de sentenças

    # Adicionar o último pedaço, se houver
    if pedaco_atual:
        sentencas.append(pedaco_atual.strip())

    return sentencas


def resumir_texto(texto, max_input_length=512, max_output_length=300, num_sentencas=2):
    """
    Função que divide o texto em partes de até num_sentences sentenças,
    faz um resumo de cada parte e junta os resumos.
    """
    resumos = []

    # Dividir o texto em partes baseadas em duas sentenças
    pedacos_de_texto = dividir_texto_em_sentencas(texto, num_sentencas)

    # Fazer um resumo para cada parte do texto
    for pedaco in pedacos_de_texto:
        texto_com_comando = "summarize: " + pedaco
        resumo = resumir_pedaco(texto_com_comando, max_input_length, max_output_length)
        resumos.append(resumo)

    # Juntar todos os resumos gerados
    return " ".join(resumos)


# Função para gerar resumos longos
def resumir_pedaco(texto, max_input_length=512, max_output_length=300):
    inputs = tokenizer(
        texto,
        return_tensors="tf",
        max_length=max_input_length,
        truncation=True,
        padding="max_length",
    )

    if inputs["input_ids"].shape[1] == 0:
        print("Erro: Nenhum token foi gerado pelo tokenizer.")
        return ""

    try:
        ids_resumo = modelo.generate(
            inputs["input_ids"],
            max_length=max_output_length,
            num_beams=4,
            length_penalty=1.0,
            early_stopping=True,
        )

        print(
            ids_resumo.shape
        )  # Formato do batch (x, y) do resumo gerado onde x é a quantidade
        # de amostras analizadas e y é a quantidade de tokens referente a amostra x.

        # Verificar os IDs do resumo gerado
        if ids_resumo.shape[1] == 0:
            print("Erro: Nenhum resumo foi gerado.")
            return ""

        print(f"Resumo IDs gerados: {ids_resumo}")

        return tokenizer.decode(ids_resumo[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Erro ao gerar o resumo: {e}")
        return ""  # Retorna uma string vazia se houver erro


def salvar_resumo_em_arquivo(resumo, nome_arquivo="resumo_gerado.txt"):
    """
    Função que salva o resumo gerado em um arquivo de texto formatado.
    """
    final_de_frase = re.compile(r"([.!?])")
    count_cols = 0
    try:
        with open(nome_arquivo, "w", encoding="utf-8") as file:
            file.write("Resumo Gerado:\n")
            file.write("=" * 40 + "\n")
            for i,char in enumerate(resumo):
                file.write(resumo[i])
                count_cols += 1
                if count_cols == 130:
                    file.write("\n")
                    count_cols = 0
            file.write("=" * 40 + "\n")
        print(f"Resumo salvo com sucesso no arquivo '{nome_arquivo}'!")
    except Exception as e:
        print(f"Erro ao salvar o resumo: {e}")


# Função principal
def main():
    url = input("Digite a URL: ")

    # Extrair o conteúdo HTML
    conteudo_html = te.obter_conteudo_pagina(url)
    soup = te.parse_conteudo_html(conteudo_html)
    texto = te.extrair_texto_relevante(soup)

    print(
        f"\nTexto extraído:\n{texto[:500]}..."
    )  # Mostra os primeiros 500 caracteres do texto

    # Resumir o texto extraído (limitar a 2000 caracteres)
    resumo = resumir_texto(texto)

    if resumo:
        salvar_resumo_em_arquivo(resumo)
    else:
        print("Erro ao gerar o resumo.")


main()
